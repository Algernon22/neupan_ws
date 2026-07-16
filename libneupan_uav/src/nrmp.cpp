#include "neupan_uav/nrmp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef NEUPAN_UAV_WITH_OSQP
#include <osqp/osqp.h>
#endif

namespace neupan_uav {

namespace {

constexpr double kPatternEps = 1.0e-12;

bool allFinite(const Eigen::MatrixXd& matrix) {
  for (Eigen::Index i = 0; i < matrix.size(); ++i) {
    if (!std::isfinite(matrix.data()[i])) return false;
  }
  return true;
}

bool allFinite(const Eigen::VectorXd& vector) {
  for (Eigen::Index i = 0; i < vector.size(); ++i) {
    if (!std::isfinite(vector(i))) return false;
  }
  return true;
}

double squaredNonnegativeSqrtWeight(double weight) {
  const double scale = std::sqrt(std::max(weight, 0.0));
  return scale * scale;
}

void validateVectorSize(const Eigen::VectorXd& vector, int size,
                        const char* name) {
  if (vector.size() != size) {
    throw std::invalid_argument(std::string(name) + " must have " +
                                std::to_string(size) + " elements");
  }
  if (!allFinite(vector)) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

void validateBoundVectorSize(const Eigen::VectorXd& vector, int size,
                             const char* name) {
  if (vector.size() != size) {
    throw std::invalid_argument(std::string(name) + " must have " +
                                std::to_string(size) + " elements");
  }
  for (Eigen::Index i = 0; i < vector.size(); ++i) {
    if (std::isnan(vector(i))) {
      throw std::invalid_argument(std::string(name) + " must not contain NaN");
    }
  }
}

void validateMatrixShape(const Eigen::MatrixXd& matrix, int rows, int cols,
                         const char* name) {
  if (matrix.rows() != rows || matrix.cols() != cols) {
    throw std::invalid_argument(std::string(name) + " must have shape " +
                                std::to_string(rows) + "x" +
                                std::to_string(cols));
  }
  if (!allFinite(matrix)) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

NrmpResult placeholderSolve(const NrmpInput& input) {
  NrmpResult result;
  result.control = input.desired_control;
  result.status = 0;
  result.iterations = 0;
  result.status_text = "placeholder";
  return result;
}

}  // namespace

#ifdef NEUPAN_UAV_WITH_OSQP

namespace {

struct Layout {
  int x_offset = 0;
  int u_offset = 0;
  int d_offset = 0;
  int e_offset = 0;
  int var_dim = 0;
  int row_count = 0;
};

struct Shape {
  int T = 0;
  int state_dim = 0;
  int control_dim = 0;
  int geom_dim = 0;
  int point_dim = 0;
  int max_num = 0;
  bool no_obs = false;
  bool enable_control_smoothing = false;

  std::vector<int> finite_speed_rows;
  std::vector<int> finite_acce_rows;
  std::vector<int> finite_tracking_speed_rows;
  std::vector<int> dyn_A_rows;
  std::vector<int> dyn_A_cols;
  std::vector<int> dyn_B_rows;
  std::vector<int> dyn_B_cols;
};

struct Slots {
  std::vector<int> P_x_diag;
  std::vector<int> P_u_diag;
  std::vector<int> P_u_smooth_offdiag;
  std::vector<int> P_e_diag;

  std::vector<int> A_dyn_x_next;
  std::vector<int> A_dyn_A;
  std::vector<int> A_dyn_B;
  std::vector<int> A_init_state;
  std::vector<int> A_accel_curr;
  std::vector<int> A_accel_prev;
  std::vector<int> A_speed;
  std::vector<int> A_tracking_speed;
  std::vector<int> A_d_bounds;
  std::vector<int> A_e_nonneg;
  std::vector<int> A_obs_e;
  std::vector<int> A_obs_d;
  std::vector<int> A_obs_fa_x;

  std::vector<int> row_dyn;
  std::vector<int> row_init_state;
  std::vector<int> row_accel;
  std::vector<int> row_speed;
  std::vector<int> row_tracking_speed;
  std::vector<int> row_d_bounds;
  std::vector<int> row_e_nonneg;
  std::vector<int> row_obs;
};

enum class SlotKind : uint8_t {
  None,
  P_x_diag,
  P_u_diag,
  P_u_smooth_offdiag,
  P_e_diag,
  A_dyn_x_next,
  A_dyn_A,
  A_dyn_B,
  A_init_state,
  A_accel_curr,
  A_accel_prev,
  A_speed,
  A_tracking_speed,
  A_d_bounds,
  A_e_nonneg,
  A_obs_e,
  A_obs_d,
  A_obs_fa_x,
};

struct SlotRef {
  SlotKind kind = SlotKind::None;
  int a = 0;
  int b = 0;
  int c = 0;
};

struct Triplet {
  int row = 0;
  int col = 0;
  SlotRef slot;
};

struct CscMatrix {
  int rows = 0;
  int cols = 0;
  std::vector<OSQPInt> p;
  std::vector<OSQPInt> i;
  std::vector<OSQPFloat> x;
};

bool isAcceptedOsqpStatus(int status_val) {
  return status_val == OSQP_SOLVED || status_val == OSQP_SOLVED_INACCURATE;
}

std::vector<double> finiteBoundRows(const Eigen::VectorXd& bound, int dim,
                                    std::vector<int>& rows) {
  rows.clear();
  std::vector<double> copy(static_cast<std::size_t>(bound.size()), 0.0);
  for (Eigen::Index i = 0; i < bound.size(); ++i) copy[i] = bound(i);
  for (int i = 0; i < dim && i < static_cast<int>(copy.size()); ++i) {
    if (std::isfinite(copy[static_cast<std::size_t>(i)])) rows.push_back(i);
  }
  return copy;
}

std::vector<double> copyMatrixRowMajor(const Eigen::MatrixXd& matrix) {
  std::vector<double> out(
      static_cast<std::size_t>(matrix.rows() * matrix.cols()), 0.0);
  for (Eigen::Index r = 0; r < matrix.rows(); ++r) {
    for (Eigen::Index c = 0; c < matrix.cols(); ++c) {
      out[static_cast<std::size_t>(r * matrix.cols() + c)] = matrix(r, c);
    }
  }
  return out;
}

std::vector<double> copyVector(const Eigen::VectorXd& vector) {
  std::vector<double> out(static_cast<std::size_t>(vector.size()), 0.0);
  for (Eigen::Index i = 0; i < vector.size(); ++i) out[i] = vector(i);
  return out;
}

void nonzeroPattern(const std::vector<double>& matrix, int rows, int cols,
                    std::vector<int>& out_rows, std::vector<int>& out_cols) {
  out_rows.clear();
  out_cols.clear();
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      if (std::abs(matrix[static_cast<std::size_t>(r * cols + c)]) >
          kPatternEps) {
        out_rows.push_back(r);
        out_cols.push_back(c);
      }
    }
  }
}

Layout makeLayout(const Shape& s) {
  Layout l;
  const int x_var_dim = s.state_dim * (s.T + 1);
  const int u_var_dim = s.control_dim * s.T;
  const int d_var_dim = s.no_obs ? 0 : s.T;
  const int e_var_dim = s.no_obs ? 0 : s.max_num * s.T;

  l.x_offset = 0;
  l.u_offset = l.x_offset + x_var_dim;
  l.d_offset = l.u_offset + u_var_dim;
  l.e_offset = l.d_offset + d_var_dim;
  l.var_dim = l.e_offset + e_var_dim;

  l.row_count = s.T * s.state_dim;
  l.row_count += s.geom_dim;
  if (s.T > 1) {
    l.row_count +=
        (s.T - 1) * static_cast<int>(s.finite_acce_rows.size());
  }
  l.row_count += s.T * static_cast<int>(s.finite_speed_rows.size());
  l.row_count +=
      (s.T + 1) * static_cast<int>(s.finite_tracking_speed_rows.size());
  if (!s.no_obs) {
    l.row_count += s.T;
    l.row_count += s.T * s.max_num;
    l.row_count += s.T * s.max_num;
  }
  return l;
}

int xIndex(const Shape& s, const Layout& l, int t, int i) {
  return l.x_offset + t * s.state_dim + i;
}

int uIndex(const Shape& s, const Layout& l, int t, int i) {
  return l.u_offset + t * s.control_dim + i;
}

int dIndex(const Layout& l, int t) { return l.d_offset + t; }

int eIndex(const Shape& s, const Layout& l, int t, int k) {
  return l.e_offset + t * s.max_num + k;
}

void initSlots(const Shape& s, Slots& slots) {
  auto fill = [](std::vector<int>& v, int n) { v.assign(std::max(0, n), -1); };

  fill(slots.P_x_diag, (s.T + 1) * s.state_dim);
  fill(slots.P_u_diag, s.T * s.control_dim);
  fill(slots.P_u_smooth_offdiag, std::max(s.T - 1, 0) * s.control_dim);
  fill(slots.P_e_diag, s.no_obs ? 0 : s.T * s.max_num);

  fill(slots.A_dyn_x_next, s.T * s.state_dim);
  fill(slots.A_dyn_A, s.T * static_cast<int>(s.dyn_A_rows.size()));
  fill(slots.A_dyn_B, s.T * static_cast<int>(s.dyn_B_rows.size()));
  fill(slots.A_init_state, s.geom_dim);
  fill(slots.A_accel_curr, std::max(s.T - 1, 0) * s.control_dim);
  fill(slots.A_accel_prev, std::max(s.T - 1, 0) * s.control_dim);
  fill(slots.A_speed, s.T * s.control_dim);
  fill(slots.A_tracking_speed,
       (s.T + 1) * static_cast<int>(s.finite_tracking_speed_rows.size()));
  fill(slots.A_d_bounds, s.no_obs ? 0 : s.T);
  fill(slots.A_e_nonneg, s.no_obs ? 0 : s.T * s.max_num);
  fill(slots.A_obs_e, s.no_obs ? 0 : s.T * s.max_num);
  fill(slots.A_obs_d, s.no_obs ? 0 : s.T * s.max_num);
  fill(slots.A_obs_fa_x, s.no_obs ? 0 : s.T * s.max_num * s.point_dim);

  fill(slots.row_dyn, s.T * s.state_dim);
  fill(slots.row_init_state, s.geom_dim);
  fill(slots.row_accel, std::max(s.T - 1, 0) * s.control_dim);
  fill(slots.row_speed, s.T * s.control_dim);
  fill(slots.row_tracking_speed,
       (s.T + 1) * static_cast<int>(s.finite_tracking_speed_rows.size()));
  fill(slots.row_d_bounds, s.no_obs ? 0 : s.T);
  fill(slots.row_e_nonneg, s.no_obs ? 0 : s.T * s.max_num);
  fill(slots.row_obs, s.no_obs ? 0 : s.T * s.max_num);
}

void assignSlot(Slots& slots, const Shape& s, const SlotRef& ref,
                int data_idx) {
  switch (ref.kind) {
    case SlotKind::P_x_diag:
      slots.P_x_diag[ref.a * s.state_dim + ref.b] = data_idx;
      break;
    case SlotKind::P_u_diag:
      slots.P_u_diag[ref.a * s.control_dim + ref.b] = data_idx;
      break;
    case SlotKind::P_u_smooth_offdiag:
      slots.P_u_smooth_offdiag[ref.a * s.control_dim + ref.b] = data_idx;
      break;
    case SlotKind::P_e_diag:
      slots.P_e_diag[ref.a * s.max_num + ref.b] = data_idx;
      break;
    case SlotKind::A_dyn_x_next:
      slots.A_dyn_x_next[ref.a * s.state_dim + ref.b] = data_idx;
      break;
    case SlotKind::A_dyn_A:
      slots.A_dyn_A[ref.a * static_cast<int>(s.dyn_A_rows.size()) + ref.b] =
          data_idx;
      break;
    case SlotKind::A_dyn_B:
      slots.A_dyn_B[ref.a * static_cast<int>(s.dyn_B_rows.size()) + ref.b] =
          data_idx;
      break;
    case SlotKind::A_init_state:
      slots.A_init_state[ref.a] = data_idx;
      break;
    case SlotKind::A_accel_curr:
      slots.A_accel_curr[ref.a * s.control_dim + ref.b] = data_idx;
      break;
    case SlotKind::A_accel_prev:
      slots.A_accel_prev[ref.a * s.control_dim + ref.b] = data_idx;
      break;
    case SlotKind::A_speed:
      slots.A_speed[ref.a * s.control_dim + ref.b] = data_idx;
      break;
    case SlotKind::A_tracking_speed:
      slots.A_tracking_speed
          [ref.a * static_cast<int>(s.finite_tracking_speed_rows.size()) +
           ref.b] = data_idx;
      break;
    case SlotKind::A_d_bounds:
      slots.A_d_bounds[ref.a] = data_idx;
      break;
    case SlotKind::A_e_nonneg:
      slots.A_e_nonneg[ref.a * s.max_num + ref.b] = data_idx;
      break;
    case SlotKind::A_obs_e:
      slots.A_obs_e[ref.a * s.max_num + ref.b] = data_idx;
      break;
    case SlotKind::A_obs_d:
      slots.A_obs_d[ref.a * s.max_num + ref.b] = data_idx;
      break;
    case SlotKind::A_obs_fa_x:
      slots.A_obs_fa_x[(ref.a * s.max_num + ref.b) * s.point_dim + ref.c] =
          data_idx;
      break;
    case SlotKind::None:
      break;
  }
}

void addTriplet(std::vector<Triplet>& out, int row, int col, SlotKind kind,
                int a = 0, int b = 0, int c = 0) {
  out.push_back(Triplet{row, col, SlotRef{kind, a, b, c}});
}

CscMatrix buildCsc(int rows, int cols, std::vector<Triplet> triplets,
                   Slots& slots, const Shape& shape) {
  std::stable_sort(triplets.begin(), triplets.end(),
                   [](const Triplet& a, const Triplet& b) {
                     if (a.col != b.col) return a.col < b.col;
                     return a.row < b.row;
                   });

  for (std::size_t idx = 1; idx < triplets.size(); ++idx) {
    if (triplets[idx - 1].row == triplets[idx].row &&
        triplets[idx - 1].col == triplets[idx].col) {
      throw std::runtime_error(
          "duplicate sparse matrix entry while building NRMP OSQP structure");
    }
  }

  CscMatrix m;
  m.rows = rows;
  m.cols = cols;
  m.p.assign(static_cast<std::size_t>(cols + 1), 0);
  m.i.resize(triplets.size());
  m.x.assign(triplets.size(), 0.0);

  for (const auto& t : triplets) {
    if (t.row < 0 || t.row >= rows || t.col < 0 || t.col >= cols) {
      throw std::runtime_error("sparse matrix entry is out of bounds");
    }
    ++m.p[static_cast<std::size_t>(t.col + 1)];
  }
  for (int c = 0; c < cols; ++c) {
    m.p[static_cast<std::size_t>(c + 1)] +=
        m.p[static_cast<std::size_t>(c)];
  }

  for (std::size_t idx = 0; idx < triplets.size(); ++idx) {
    m.i[idx] = static_cast<OSQPInt>(triplets[idx].row);
    assignSlot(slots, shape, triplets[idx].slot, static_cast<int>(idx));
  }
  return m;
}

void buildStructure(const Shape& s, const Layout& l, Slots& slots,
                    CscMatrix& P, CscMatrix& A) {
  initSlots(s, slots);
  std::vector<Triplet> p_triplets;
  std::vector<Triplet> a_triplets;

  for (int t = 0; t <= s.T; ++t) {
    for (int i = 0; i < s.state_dim; ++i) {
      const int idx = xIndex(s, l, t, i);
      addTriplet(p_triplets, idx, idx, SlotKind::P_x_diag, t, i);
    }
  }
  for (int t = 1; t < s.T; ++t) {
    for (int i = 0; i < s.control_dim; ++i) {
      addTriplet(p_triplets, uIndex(s, l, t - 1, i),
                 uIndex(s, l, t, i), SlotKind::P_u_smooth_offdiag, t - 1,
                 i);
    }
  }
  for (int t = 0; t < s.T; ++t) {
    for (int i = 0; i < s.control_dim; ++i) {
      const int idx = uIndex(s, l, t, i);
      addTriplet(p_triplets, idx, idx, SlotKind::P_u_diag, t, i);
    }
  }
  if (!s.no_obs) {
    for (int t = 0; t < s.T; ++t) {
      for (int k = 0; k < s.max_num; ++k) {
        const int idx = eIndex(s, l, t, k);
        addTriplet(p_triplets, idx, idx, SlotKind::P_e_diag, t, k);
      }
    }
  }

  int row = 0;
  for (int t = 0; t < s.T; ++t) {
    for (int i = 0; i < s.state_dim; ++i) {
      slots.row_dyn[t * s.state_dim + i] = row;
      addTriplet(a_triplets, row, xIndex(s, l, t + 1, i),
                 SlotKind::A_dyn_x_next, t, i);
      ++row;
    }
    for (int slot_i = 0; slot_i < static_cast<int>(s.dyn_A_rows.size());
         ++slot_i) {
      addTriplet(a_triplets,
                 slots.row_dyn[t * s.state_dim + s.dyn_A_rows[slot_i]],
                 xIndex(s, l, t, s.dyn_A_cols[slot_i]),
                 SlotKind::A_dyn_A, t, slot_i);
    }
    for (int slot_i = 0; slot_i < static_cast<int>(s.dyn_B_rows.size());
         ++slot_i) {
      addTriplet(a_triplets,
                 slots.row_dyn[t * s.state_dim + s.dyn_B_rows[slot_i]],
                 uIndex(s, l, t, s.dyn_B_cols[slot_i]),
                 SlotKind::A_dyn_B, t, slot_i);
    }
  }

  for (int i = 0; i < s.geom_dim; ++i) {
    slots.row_init_state[i] = row;
    addTriplet(a_triplets, row, xIndex(s, l, 0, i),
               SlotKind::A_init_state, i);
    ++row;
  }

  if (s.T > 1) {
    for (int t = 1; t < s.T; ++t) {
      for (int i : s.finite_acce_rows) {
        slots.row_accel[(t - 1) * s.control_dim + i] = row;
        addTriplet(a_triplets, row, uIndex(s, l, t, i),
                   SlotKind::A_accel_curr, t - 1, i);
        addTriplet(a_triplets, row, uIndex(s, l, t - 1, i),
                   SlotKind::A_accel_prev, t - 1, i);
        ++row;
      }
    }
  }

  for (int t = 0; t < s.T; ++t) {
    for (int i : s.finite_speed_rows) {
      slots.row_speed[t * s.control_dim + i] = row;
      addTriplet(a_triplets, row, uIndex(s, l, t, i), SlotKind::A_speed, t,
                 i);
      ++row;
    }
  }

  const int track_width =
      static_cast<int>(s.finite_tracking_speed_rows.size());
  for (int t = 0; t <= s.T; ++t) {
    for (int slot_i = 0; slot_i < track_width; ++slot_i) {
      const int bound_i = s.finite_tracking_speed_rows[slot_i];
      slots.row_tracking_speed[t * track_width + slot_i] = row;
      addTriplet(a_triplets, row, xIndex(s, l, t, 4 + bound_i),
                 SlotKind::A_tracking_speed, t, slot_i);
      ++row;
    }
  }

  if (!s.no_obs) {
    for (int t = 0; t < s.T; ++t) {
      slots.row_d_bounds[t] = row;
      addTriplet(a_triplets, row, dIndex(l, t), SlotKind::A_d_bounds, t);
      ++row;
    }
    for (int t = 0; t < s.T; ++t) {
      for (int k = 0; k < s.max_num; ++k) {
        slots.row_e_nonneg[t * s.max_num + k] = row;
        addTriplet(a_triplets, row, eIndex(s, l, t, k),
                   SlotKind::A_e_nonneg, t, k);
        ++row;
      }
    }
    for (int t = 0; t < s.T; ++t) {
      for (int k = 0; k < s.max_num; ++k) {
        slots.row_obs[t * s.max_num + k] = row;
        addTriplet(a_triplets, row, eIndex(s, l, t, k), SlotKind::A_obs_e,
                   t, k);
        addTriplet(a_triplets, row, dIndex(l, t), SlotKind::A_obs_d, t, k);
        for (int j = 0; j < s.point_dim; ++j) {
          addTriplet(a_triplets, row, xIndex(s, l, t + 1, j),
                     SlotKind::A_obs_fa_x, t, k, j);
        }
        ++row;
      }
    }
  }

  if (row != l.row_count) {
    throw std::runtime_error("internal NRMP OSQP row count mismatch");
  }

  P = buildCsc(l.var_dim, l.var_dim, std::move(p_triplets), slots, s);
  A = buildCsc(l.row_count, l.var_dim, std::move(a_triplets), slots, s);
}

void setSlot(std::vector<OSQPFloat>& data, const std::vector<int>& slots,
             int idx, double value) {
  if (idx < 0 || idx >= static_cast<int>(slots.size())) return;
  const int slot = slots[static_cast<std::size_t>(idx)];
  if (slot >= 0) data[static_cast<std::size_t>(slot)] =
      static_cast<OSQPFloat>(value);
}

void setRow(std::vector<OSQPFloat>& l, std::vector<OSQPFloat>& u,
            const std::vector<int>& rows, int idx, double low, double high) {
  if (idx < 0 || idx >= static_cast<int>(rows.size())) return;
  const int row = rows[static_cast<std::size_t>(idx)];
  if (row >= 0) {
    l[static_cast<std::size_t>(row)] = static_cast<OSQPFloat>(low);
    u[static_cast<std::size_t>(row)] = static_cast<OSQPFloat>(high);
  }
}

double matrixValue(const Eigen::MatrixXd& matrix, int row, int col) {
  return matrix(row, col);
}

}  // namespace

class NRMP::Backend {
 public:
  explicit Backend(const NrmpConfig& config) {
    if (config.receding <= 0 || config.state_dim <= 0 ||
        config.control_dim <= 0 || config.geom_dim <= 0 ||
        config.point_dim <= 0 || config.max_num < 0) {
      throw std::invalid_argument("invalid NRMP OSQP dimensions");
    }
    validateMatrixShape(config.dynamics_A, config.state_dim, config.state_dim,
                        "NrmpConfig::dynamics_A");
    validateMatrixShape(config.dynamics_B, config.state_dim, config.control_dim,
                        "NrmpConfig::dynamics_B");
    validateVectorSize(config.dynamics_C, config.state_dim,
                       "NrmpConfig::dynamics_C");
    validateBoundVectorSize(config.speed_bound, config.control_dim,
                            "NrmpConfig::speed_bound");
    validateBoundVectorSize(config.acce_bound, config.control_dim,
                            "NrmpConfig::acce_bound");
    if (config.tracking_speed_bound.size() != 0) {
      validateBoundVectorSize(config.tracking_speed_bound, 4,
                              "NrmpConfig::tracking_speed_bound");
    }

    A_const_ = copyMatrixRowMajor(config.dynamics_A);
    B_const_ = copyMatrixRowMajor(config.dynamics_B);
    C_const_ = copyVector(config.dynamics_C);
    speed_bound_ = finiteBoundRows(config.speed_bound, config.control_dim,
                                   shape_.finite_speed_rows);
    acce_bound_ = finiteBoundRows(config.acce_bound, config.control_dim,
                                  shape_.finite_acce_rows);
    Eigen::VectorXd tracking =
        config.tracking_speed_bound.size() == 0
            ? Eigen::VectorXd::Constant(4,
                                        std::numeric_limits<double>::infinity())
            : config.tracking_speed_bound;
    tracking_speed_bound_ = finiteBoundRows(
        tracking, 4, shape_.finite_tracking_speed_rows);

    shape_.T = config.receding;
    shape_.state_dim = config.state_dim;
    shape_.control_dim = config.control_dim;
    shape_.geom_dim = config.geom_dim;
    shape_.point_dim = config.point_dim;
    shape_.max_num = config.max_num;
    shape_.no_obs = config.no_obs || config.max_num <= 0;
    shape_.enable_control_smoothing = config.enable_control_smoothing;

    nonzeroPattern(A_const_, shape_.state_dim, shape_.state_dim,
                   shape_.dyn_A_rows, shape_.dyn_A_cols);
    nonzeroPattern(B_const_, shape_.state_dim, shape_.control_dim,
                   shape_.dyn_B_rows, shape_.dyn_B_cols);

    layout_ = makeLayout(shape_);
    buildStructure(shape_, layout_, slots_, P_, A_mat_);
    q_.assign(static_cast<std::size_t>(layout_.var_dim), 0.0);
    l_.assign(static_cast<std::size_t>(layout_.row_count), 0.0);
    u_.assign(static_cast<std::size_t>(layout_.row_count), 0.0);

    applyConstantValues();
    setupSolver(config.solver_options);
  }

  ~Backend() {
    if (solver_) {
      osqp_cleanup(solver_);
      solver_ = nullptr;
    }
  }

  Backend(const Backend&) = delete;
  Backend& operator=(const Backend&) = delete;

  NrmpResult solve(const NrmpInput& input) {
    validateSolveInput(input);
    fillValues(input);

    OSQPInt update_status = osqp_update_data_mat(
        solver_, P_.x.data(), nullptr, static_cast<OSQPInt>(P_.x.size()),
        A_mat_.x.data(), nullptr, static_cast<OSQPInt>(A_mat_.x.size()));
    if (update_status == 0) {
      update_status =
          osqp_update_data_vec(solver_, q_.data(), l_.data(), u_.data());
    }
    if (update_status != 0) {
      throw std::runtime_error("OSQP update failed with code " +
                               std::to_string(static_cast<int>(update_status)));
    }

    const OSQPInt solve_status = osqp_solve(solver_);
    if (solve_status != 0) {
      throw std::runtime_error("OSQP solve failed with code " +
                               std::to_string(static_cast<int>(solve_status)));
    }
    ++solve_count_;

    if (!solver_->solution || !solver_->solution->x) {
      throw std::runtime_error("OSQP returned an empty solution");
    }
    const int status_val =
        solver_->info ? static_cast<int>(solver_->info->status_val) : 0;
    if (!isAcceptedOsqpStatus(status_val)) {
      throw std::runtime_error(
          std::string("Solver osqp returned status ") +
          (solver_->info ? solver_->info->status : "unknown"));
    }

    return buildResult();
  }

 private:
  void setupSolver(const NrmpSolverOptions& options) {
    OSQPSettings settings;
    osqp_set_default_settings(&settings);
    settings.verbose = options.verbose ? 1 : 0;
    settings.warm_starting = options.warm_starting ? 1 : 0;
    settings.polishing = options.polishing ? 1 : 0;
    settings.allocate_solution = 1;
    settings.eps_abs = options.eps_abs;
    settings.eps_rel = options.eps_rel;
    settings.max_iter = static_cast<OSQPInt>(options.max_iter);

    OSQPCscMatrix P_csc;
    OSQPCscMatrix A_csc;
    OSQPCscMatrix_set_data(&P_csc, P_.rows, P_.cols,
                           static_cast<OSQPInt>(P_.x.size()), P_.x.data(),
                           P_.i.data(), P_.p.data());
    OSQPCscMatrix_set_data(&A_csc, A_mat_.rows, A_mat_.cols,
                           static_cast<OSQPInt>(A_mat_.x.size()),
                           A_mat_.x.data(), A_mat_.i.data(), A_mat_.p.data());

    const OSQPInt status =
        osqp_setup(&solver_, &P_csc, q_.data(), &A_csc, l_.data(), u_.data(),
                   layout_.row_count, layout_.var_dim, &settings);
    if (status != 0 || solver_ == nullptr) {
      throw std::runtime_error("OSQP setup failed with code " +
                               std::to_string(static_cast<int>(status)));
    }
    ++setup_count_;
  }

  void validateSolveInput(const NrmpInput& input) const {
    validateMatrixShape(input.nominal_states, shape_.state_dim, shape_.T + 1,
                        "NrmpInput::nominal_states");
    validateMatrixShape(input.reference_states, shape_.state_dim, shape_.T + 1,
                        "NrmpInput::reference_states");
    validateMatrixShape(input.reference_controls, shape_.control_dim, shape_.T,
                        "NrmpInput::reference_controls");

    if (input.state_weights.size() != 1 &&
        input.state_weights.size() != shape_.state_dim) {
      throw std::invalid_argument(
          "NrmpInput::state_weights must be scalar or length state_dim");
    }
    if (!allFinite(input.state_weights)) {
      throw std::invalid_argument("NrmpInput::state_weights must be finite");
    }

    if (!shape_.no_obs) {
      validateMatrixShape(input.fa_batch, shape_.T * shape_.max_num,
                          shape_.point_dim, "NrmpInput::fa_batch");
      if ((input.fb_batch.rows() != shape_.T ||
           input.fb_batch.cols() != shape_.max_num) &&
          (input.fb_batch.rows() != shape_.T * shape_.max_num ||
           input.fb_batch.cols() != 1)) {
        throw std::invalid_argument(
            "NrmpInput::fb_batch must have shape T x max_num or "
            "(T*max_num) x 1");
      }
      if (!allFinite(input.fb_batch)) {
        throw std::invalid_argument("NrmpInput::fb_batch must be finite");
      }
    }
  }

  void applyConstantValues() {
    for (int idx : slots_.A_dyn_x_next) {
      if (idx >= 0) A_mat_.x[static_cast<std::size_t>(idx)] = 1.0;
    }
    for (int idx : slots_.A_init_state) {
      if (idx >= 0) A_mat_.x[static_cast<std::size_t>(idx)] = 1.0;
    }
    for (int idx : slots_.A_accel_curr) {
      if (idx >= 0) A_mat_.x[static_cast<std::size_t>(idx)] = 1.0;
    }
    for (int idx : slots_.A_accel_prev) {
      if (idx >= 0) A_mat_.x[static_cast<std::size_t>(idx)] = -1.0;
    }
    for (int idx : slots_.A_speed) {
      if (idx >= 0) A_mat_.x[static_cast<std::size_t>(idx)] = 1.0;
    }
    for (int idx : slots_.A_tracking_speed) {
      if (idx >= 0) A_mat_.x[static_cast<std::size_t>(idx)] = 1.0;
    }
    for (int idx : slots_.A_d_bounds) {
      if (idx >= 0) A_mat_.x[static_cast<std::size_t>(idx)] = 1.0;
    }
    for (int idx : slots_.A_e_nonneg) {
      if (idx >= 0) A_mat_.x[static_cast<std::size_t>(idx)] = 1.0;
    }
    for (int idx : slots_.A_obs_e) {
      if (idx >= 0) A_mat_.x[static_cast<std::size_t>(idx)] = 1.0;
    }
    for (int idx : slots_.A_obs_d) {
      if (idx >= 0) A_mat_.x[static_cast<std::size_t>(idx)] = -1.0;
    }

    for (int t = 0; t + 1 < shape_.T; ++t) {
      for (int i : shape_.finite_acce_rows) {
        setRow(l_, u_, slots_.row_accel, t * shape_.control_dim + i,
               -acce_bound_[static_cast<std::size_t>(i)],
               acce_bound_[static_cast<std::size_t>(i)]);
      }
    }
    for (int t = 0; t < shape_.T; ++t) {
      for (int i : shape_.finite_speed_rows) {
        setRow(l_, u_, slots_.row_speed, t * shape_.control_dim + i,
               -speed_bound_[static_cast<std::size_t>(i)],
               speed_bound_[static_cast<std::size_t>(i)]);
      }
    }
    const int track_width =
        static_cast<int>(shape_.finite_tracking_speed_rows.size());
    for (int t = 0; t <= shape_.T; ++t) {
      for (int slot_i = 0; slot_i < track_width; ++slot_i) {
        const int bound_i = shape_.finite_tracking_speed_rows[slot_i];
        setRow(l_, u_, slots_.row_tracking_speed, t * track_width + slot_i,
               -tracking_speed_bound_[static_cast<std::size_t>(bound_i)],
               tracking_speed_bound_[static_cast<std::size_t>(bound_i)]);
      }
    }
    if (!shape_.no_obs) {
      for (int row : slots_.row_e_nonneg) {
        if (row >= 0) {
          l_[static_cast<std::size_t>(row)] = 0.0;
          u_[static_cast<std::size_t>(row)] = OSQP_INFTY;
        }
      }
      for (int row : slots_.row_obs) {
        if (row >= 0) u_[static_cast<std::size_t>(row)] = OSQP_INFTY;
      }
    }
  }

  void fillValues(const NrmpInput& input) {
    std::fill(q_.begin(), q_.end(), 0.0);
    std::fill(P_.x.begin(), P_.x.end(), 0.0);

    std::vector<double> q_s(static_cast<std::size_t>(shape_.state_dim), 0.0);
    if (input.state_weights.size() == 1) {
      std::fill(q_s.begin(), q_s.end(), input.state_weights(0));
    } else {
      for (int i = 0; i < shape_.state_dim; ++i) {
        q_s[static_cast<std::size_t>(i)] = input.state_weights(i);
      }
    }

    for (int t = 0; t <= shape_.T; ++t) {
      for (int i = 0; i < shape_.state_dim; ++i) {
        const double weight = q_s[static_cast<std::size_t>(i)];
        double diag = 2.0 * weight * weight + input.bk;
        double linear = -2.0 * weight * weight *
                            matrixValue(input.reference_states, i, t) -
                        input.bk * matrixValue(input.nominal_states, i, t);
        if (shape_.state_dim > shape_.geom_dim && t == 0 &&
            i >= shape_.geom_dim) {
          diag += input.bk;
          linear -= input.bk * matrixValue(input.nominal_states, i, 0);
        }
        setSlot(P_.x, slots_.P_x_diag, t * shape_.state_dim + i, diag);
        q_[static_cast<std::size_t>(xIndex(shape_, layout_, t, i))] =
            static_cast<OSQPFloat>(linear);
      }
    }

    const double p_u_diag = 2.0 * input.p_u * input.p_u;
    std::vector<double> u_diag(
        static_cast<std::size_t>(shape_.T * shape_.control_dim), p_u_diag);
    for (int t = 0; t < shape_.T; ++t) {
      for (int i = 0; i < shape_.control_dim; ++i) {
        q_[static_cast<std::size_t>(uIndex(shape_, layout_, t, i))] =
            static_cast<OSQPFloat>(-2.0 * input.p_u * input.p_u *
                                   matrixValue(input.reference_controls, i, t));
      }
    }

    const double smooth_du_sq =
        squaredNonnegativeSqrtWeight(input.smooth_du);
    const double smooth_u0_sq =
        squaredNonnegativeSqrtWeight(input.smooth_u0);

    if (smooth_du_sq > 0.0 && shape_.T > 1) {
      for (int t = 0; t + 1 < shape_.T; ++t) {
        for (int i = 0; i < shape_.control_dim; ++i) {
          u_diag[static_cast<std::size_t>(t * shape_.control_dim + i)] +=
              smooth_du_sq;
          u_diag[static_cast<std::size_t>((t + 1) * shape_.control_dim + i)] +=
              smooth_du_sq;
          setSlot(P_.x, slots_.P_u_smooth_offdiag,
                  t * shape_.control_dim + i, -smooth_du_sq);
        }
      }
    }

    if (smooth_u0_sq > 0.0 && shape_.T > 0) {
      for (int i = 0; i < shape_.control_dim; ++i) {
        u_diag[static_cast<std::size_t>(i)] += smooth_u0_sq;
        q_[static_cast<std::size_t>(uIndex(shape_, layout_, 0, i))] -=
            static_cast<OSQPFloat>(smooth_u0_sq * input.seed_control(i));
      }
    }

    for (int t = 0; t < shape_.T; ++t) {
      for (int i = 0; i < shape_.control_dim; ++i) {
        setSlot(P_.x, slots_.P_u_diag, t * shape_.control_dim + i,
                u_diag[static_cast<std::size_t>(t * shape_.control_dim + i)]);
      }
    }

    if (!shape_.no_obs) {
      for (int t = 0; t < shape_.T; ++t) {
        q_[static_cast<std::size_t>(dIndex(layout_, t))] =
            static_cast<OSQPFloat>(-input.eta);
        setRow(l_, u_, slots_.row_d_bounds, t, input.d_min, input.d_max);
        for (int k = 0; k < shape_.max_num; ++k) {
          setSlot(P_.x, slots_.P_e_diag, t * shape_.max_num + k,
                  input.ro_obs);
        }
      }
    }

    for (int t = 0; t < shape_.T; ++t) {
      for (int slot_i = 0; slot_i < static_cast<int>(shape_.dyn_A_rows.size());
           ++slot_i) {
        const int r = shape_.dyn_A_rows[slot_i];
        const int c = shape_.dyn_A_cols[slot_i];
        setSlot(A_mat_.x, slots_.A_dyn_A,
                t * static_cast<int>(shape_.dyn_A_rows.size()) + slot_i,
                -A_const_[static_cast<std::size_t>(r * shape_.state_dim + c)]);
      }
      for (int slot_i = 0; slot_i < static_cast<int>(shape_.dyn_B_rows.size());
           ++slot_i) {
        const int r = shape_.dyn_B_rows[slot_i];
        const int c = shape_.dyn_B_cols[slot_i];
        setSlot(
            A_mat_.x, slots_.A_dyn_B,
            t * static_cast<int>(shape_.dyn_B_rows.size()) + slot_i,
            -B_const_[static_cast<std::size_t>(r * shape_.control_dim + c)]);
      }
      for (int i = 0; i < shape_.state_dim; ++i) {
        const int row =
            slots_.row_dyn[static_cast<std::size_t>(t * shape_.state_dim + i)];
        l_[static_cast<std::size_t>(row)] =
            static_cast<OSQPFloat>(C_const_[static_cast<std::size_t>(i)]);
        u_[static_cast<std::size_t>(row)] =
            static_cast<OSQPFloat>(C_const_[static_cast<std::size_t>(i)]);
      }
    }

    for (int i = 0; i < shape_.geom_dim; ++i) {
      const double value = matrixValue(input.nominal_states, i, 0);
      const int row = slots_.row_init_state[static_cast<std::size_t>(i)];
      l_[static_cast<std::size_t>(row)] = static_cast<OSQPFloat>(value);
      u_[static_cast<std::size_t>(row)] = static_cast<OSQPFloat>(value);
    }

    if (!shape_.no_obs) {
      for (int t = 0; t < shape_.T; ++t) {
        for (int k = 0; k < shape_.max_num; ++k) {
          const int obs_idx = t * shape_.max_num + k;
          const int row =
              slots_.row_obs[static_cast<std::size_t>(obs_idx)];
          const double fb =
              input.fb_batch.rows() == shape_.T
                  ? input.fb_batch(t, k)
                  : input.fb_batch(obs_idx, 0);
          l_[static_cast<std::size_t>(row)] =
              static_cast<OSQPFloat>(fb);
          for (int j = 0; j < shape_.point_dim; ++j) {
            setSlot(A_mat_.x, slots_.A_obs_fa_x,
                    obs_idx * shape_.point_dim + j,
                    input.fa_batch(obs_idx, j));
          }
        }
      }
    }
  }

  NrmpResult buildResult() const {
    NrmpResult result;
    result.state_trajectory.resize(shape_.state_dim, shape_.T + 1);
    result.control_trajectory.resize(shape_.control_dim, shape_.T);
    const OSQPFloat* sol = solver_->solution->x;

    for (int i = 0; i < shape_.state_dim; ++i) {
      for (int t = 0; t <= shape_.T; ++t) {
        result.state_trajectory(i, t) =
            static_cast<double>(sol[xIndex(shape_, layout_, t, i)]);
      }
    }
    for (int i = 0; i < shape_.control_dim; ++i) {
      for (int t = 0; t < shape_.T; ++t) {
        result.control_trajectory(i, t) =
            static_cast<double>(sol[uIndex(shape_, layout_, t, i)]);
      }
    }
    if (shape_.control_dim == 4 && shape_.T > 0) {
      result.control = result.control_trajectory.col(0);
    }

    if (!shape_.no_obs) {
      result.nominal_distance.resize(shape_.T);
      for (int t = 0; t < shape_.T; ++t) {
        result.nominal_distance(t) = static_cast<double>(sol[dIndex(layout_, t)]);
      }
    }

    result.status = solver_->info ? static_cast<int>(solver_->info->status_val) : 0;
    result.status_text = solver_->info ? solver_->info->status : "";
    result.iterations = solver_->info ? static_cast<int>(solver_->info->iter) : 0;
    result.solve_sec = solver_->info ? static_cast<double>(solver_->info->solve_time) : 0.0;
    result.run_time_sec = solver_->info ? static_cast<double>(solver_->info->run_time) : 0.0;
    result.setup_count = setup_count_;
    result.solve_count = solve_count_;
    return result;
  }

  Shape shape_;
  Layout layout_;
  Slots slots_;
  CscMatrix P_;
  CscMatrix A_mat_;
  std::vector<OSQPFloat> q_;
  std::vector<OSQPFloat> l_;
  std::vector<OSQPFloat> u_;

  std::vector<double> A_const_;
  std::vector<double> B_const_;
  std::vector<double> C_const_;
  std::vector<double> speed_bound_;
  std::vector<double> acce_bound_;
  std::vector<double> tracking_speed_bound_;

  OSQPSolver* solver_ = nullptr;
  int setup_count_ = 0;
  int solve_count_ = 0;
};

#else

class NRMP::Backend {
 public:
  explicit Backend(const NrmpConfig&) {
    throw std::runtime_error(
        "libneupan_uav was built without OSQP support; NRMP backend is unavailable");
  }
};

#endif

NRMP::NRMP() = default;

NRMP::NRMP(NrmpConfig config) : config_(std::move(config)) {
  backend_ = std::make_unique<Backend>(config_);
}

NRMP::~NRMP() = default;
NRMP::NRMP(NRMP&&) noexcept = default;
NRMP& NRMP::operator=(NRMP&&) noexcept = default;

bool NRMP::hasBackend() const { return backend_ != nullptr; }

const NrmpConfig& NRMP::config() const { return config_; }

NrmpResult NRMP::solve(const NrmpInput& input) {
  if (!backend_) return placeholderSolve(input);
#ifdef NEUPAN_UAV_WITH_OSQP
  return backend_->solve(input);
#else
  throw std::runtime_error(
      "libneupan_uav was built without OSQP support; NRMP solve is unavailable");
#endif
}

}  // namespace neupan_uav
