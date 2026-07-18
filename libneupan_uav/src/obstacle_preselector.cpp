#include "neupan_uav/obstacle_preselector.hpp"

#include "detail/box_geometry.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

namespace neupan_uav {

namespace {

using Clock = std::chrono::steady_clock;

double elapsedSeconds(const Clock::time_point& start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

bool allFinite(const Eigen::MatrixXd& matrix) {
  for (Eigen::Index i = 0; i < matrix.size(); ++i) {
    if (!std::isfinite(matrix.data()[i])) return false;
  }
  return true;
}

void validatePointMatrix(const PointMatrix& points, const char* name) {
  if (points.rows() != 3) {
    throw std::invalid_argument(std::string(name) + " must have shape 3xN");
  }
  if (!allFinite(points)) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

void validateVelocities(const PointMatrix& points,
                        const PointMatrix& velocities) {
  if (velocities.size() != 0 &&
      (velocities.rows() != 3 || velocities.cols() != points.cols())) {
    throw std::invalid_argument(
        "obstacle velocities must be empty or have shape 3xN matching points");
  }
  if (velocities.size() != 0 && !allFinite(velocities)) {
    throw std::invalid_argument("obstacle velocities must be finite");
  }
}

Eigen::Vector3d nonnegativeVector(const Eigen::Vector3d& value,
                                  const char* name) {
  if (!value.allFinite() || (value.array() < 0.0).any()) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and non-negative");
  }
  return value;
}

Eigen::Vector3d positiveVector(const Eigen::Vector3d& value,
                               const char* name) {
  if (!value.allFinite() || (value.array() <= 0.0).any()) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and positive");
  }
  return value;
}

double finiteNonnegative(double value, const char* name) {
  if (!std::isfinite(value) || value < 0.0) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and non-negative");
  }
  return value;
}

double finitePositive(double value, const char* name) {
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and positive");
  }
  return value;
}

double ratio(double value, const char* name) {
  if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and within [0, 1]");
  }
  return value;
}

ObstaclePreselectorConfig normalizeConfig(ObstaclePreselectorConfig config) {
  config.dt = finitePositive(config.dt, "ObstaclePreselectorConfig::dt");
  config.body_half_extent =
      nonnegativeVector(config.body_half_extent,
                        "ObstaclePreselectorConfig::body_half_extent");
  config.corridor_margin =
      nonnegativeVector(config.corridor_margin,
                        "ObstaclePreselectorConfig::corridor_margin");
  config.exact_margin =
      nonnegativeVector(config.exact_margin,
                        "ObstaclePreselectorConfig::exact_margin");
  config.diversity_voxel =
      positiveVector(config.diversity_voxel,
                     "ObstaclePreselectorConfig::diversity_voxel");
  config.hard_distance =
      finiteNonnegative(config.hard_distance,
                        "ObstaclePreselectorConfig::hard_distance");
  config.temporal_radius =
      finiteNonnegative(config.temporal_radius,
                        "ObstaclePreselectorConfig::temporal_radius");
  config.nearest_ratio =
      ratio(config.nearest_ratio, "ObstaclePreselectorConfig::nearest_ratio");
  config.temporal_ratio =
      ratio(config.temporal_ratio, "ObstaclePreselectorConfig::temporal_ratio");
  config.diversity_ratio =
      ratio(config.diversity_ratio, "ObstaclePreselectorConfig::diversity_ratio");
  const double sum =
      config.nearest_ratio + config.temporal_ratio + config.diversity_ratio;
  if (sum > 1.0 + 1.0e-6) {
    throw std::invalid_argument(
        "ObstaclePreselectorConfig quota ratios must sum to <= 1");
  }
  if (config.per_step < 0) {
    throw std::invalid_argument(
        "ObstaclePreselectorConfig::per_step must be non-negative");
  }
  return config;
}

Eigen::Matrix3d makeRotation(double roll, double pitch, double yaw) {
  const double cr = std::cos(roll);
  const double sr = std::sin(roll);
  const double cp = std::cos(pitch);
  const double sp = std::sin(pitch);
  const double cy = std::cos(yaw);
  const double sy = std::sin(yaw);

  Eigen::Matrix3d rotation;
  rotation << cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
              sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
              -sp, cp * sr, cp * cr;
  return rotation;
}

std::vector<Eigen::Matrix3d> buildRotations(
    const Trajectory& nominal_states,
    const Eigen::MatrixXd& attitude_horizon) {
  const Eigen::Index steps = nominal_states.cols();
  std::vector<Eigen::Matrix3d> rotations;
  rotations.reserve(static_cast<std::size_t>(steps));

  if (attitude_horizon.size() != 0) {
    if (attitude_horizon.rows() != 3 || attitude_horizon.cols() != steps) {
      throw std::invalid_argument(
          "attitude_horizon must have shape 3xN matching nominal states");
    }
    if (!allFinite(attitude_horizon)) {
      throw std::invalid_argument("attitude_horizon must be finite");
    }
    for (Eigen::Index t = 0; t < steps; ++t) {
      rotations.push_back(makeRotation(attitude_horizon(0, t),
                                       attitude_horizon(1, t),
                                       attitude_horizon(2, t)));
    }
    return rotations;
  }

  for (Eigen::Index t = 0; t < steps; ++t) {
    const double yaw = nominal_states.rows() >= 4 ? nominal_states(3, t) : 0.0;
    rotations.push_back(makeRotation(0.0, 0.0, yaw));
  }
  return rotations;
}

std::array<int, 3> quotaCounts(int total, double nearest_ratio,
                               double temporal_ratio,
                               double diversity_ratio) {
  total = std::max(0, total);
  std::array<double, 3> ratios{
      std::clamp(nearest_ratio, 0.0, 1.0),
      std::clamp(temporal_ratio, 0.0, 1.0),
      std::clamp(diversity_ratio, 0.0, 1.0),
  };
  std::array<double, 3> raw{};
  std::array<int, 3> counts{};
  double raw_sum = 0.0;
  int count_sum = 0;
  for (int i = 0; i < 3; ++i) {
    raw[i] = ratios[i] * static_cast<double>(total);
    raw_sum += raw[i];
    counts[i] = static_cast<int>(std::floor(raw[i]));
    count_sum += counts[i];
  }

  const int target =
      std::min(total, static_cast<int>(std::floor(raw_sum + 1.0e-9)));
  int remaining = target - count_sum;
  std::array<int, 3> order{0, 1, 2};
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    const double ra = raw[a] - counts[a];
    const double rb = raw[b] - counts[b];
    if (ra == rb) return a < b;
    return ra > rb;
  });
  for (int i = 0; i < remaining; ++i) counts[order[i]] += 1;
  return counts;
}

std::vector<int> validUniqueIndices(const std::vector<int>& input, int size) {
  std::vector<int> out;
  if (size <= 0) return out;
  out.reserve(input.size());
  std::vector<uint8_t> seen(static_cast<std::size_t>(size), 0);
  for (int idx : input) {
    if (idx < 0 || idx >= size || seen[static_cast<std::size_t>(idx)]) {
      continue;
    }
    seen[static_cast<std::size_t>(idx)] = 1;
    out.push_back(idx);
  }
  return out;
}

void sortByScoreStable(std::vector<int>& indices,
                       const std::vector<double>& score) {
  std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) {
    return score[static_cast<std::size_t>(a)] <
           score[static_cast<std::size_t>(b)];
  });
}

int appendRanked(std::vector<int>& selected, std::vector<uint8_t>& selected_mask,
                 const std::vector<int>& candidates, int target_max,
                 int limit = -1) {
  if (target_max <= 0 || static_cast<int>(selected.size()) >= target_max) {
    return static_cast<int>(selected.size());
  }
  int room = target_max - static_cast<int>(selected.size());
  if (limit >= 0) room = std::min(room, std::max(0, limit));
  if (room <= 0) return static_cast<int>(selected.size());

  for (int idx : candidates) {
    if (idx < 0 || idx >= static_cast<int>(selected_mask.size()) ||
        selected_mask[static_cast<std::size_t>(idx)] != 0) {
      continue;
    }
    selected_mask[static_cast<std::size_t>(idx)] = 1;
    selected.push_back(idx);
    if (--room <= 0 || static_cast<int>(selected.size()) >= target_max) break;
  }
  return static_cast<int>(selected.size());
}

struct VoxelKey {
  int x = 0;
  int y = 0;
  int z = 0;

  bool operator==(const VoxelKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelHash {
  std::size_t operator()(const VoxelKey& key) const {
    std::size_t h = static_cast<std::size_t>(static_cast<uint32_t>(key.x));
    h ^= static_cast<std::size_t>(static_cast<uint32_t>(key.y)) +
         0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= static_cast<std::size_t>(static_cast<uint32_t>(key.z)) +
         0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
  }
};

VoxelKey voxelKey(const Eigen::Vector3d& point, const Eigen::Vector3d& voxel) {
  return {
      static_cast<int>(std::floor(point.x() / voxel.x())),
      static_cast<int>(std::floor(point.y() / voxel.y())),
      static_cast<int>(std::floor(point.z() / voxel.z())),
  };
}

PointMatrix columnsByIndex(const PointMatrix& source,
                           const std::vector<int>& indices) {
  PointMatrix out(3, static_cast<Eigen::Index>(indices.size()));
  for (Eigen::Index col = 0; col < out.cols(); ++col) {
    out.col(col) = source.col(indices[static_cast<std::size_t>(col)]);
  }
  return out;
}

Eigen::Vector3d pointAtStep(const PointMatrix& points,
                            const PointMatrix& velocities,
                            bool has_velocities, int point_index, int step,
                            double dt) {
  Eigen::Vector3d point = points.col(point_index);
  if (has_velocities) {
    point += static_cast<double>(step) * dt * velocities.col(point_index);
  }
  return point;
}

std::vector<int> fallbackPointIndices(
    const PointFlowConfig& config, const Trajectory& nominal_states,
    const PointMatrix& obstacle_points,
    const Eigen::MatrixXd& attitude_horizon) {
  const int point_count = static_cast<int>(obstacle_points.cols());
  const int keep_count =
      std::min(static_cast<int>(config.dune_max_num), point_count);
  std::vector<int> indices(static_cast<std::size_t>(point_count));
  std::iota(indices.begin(), indices.end(), 0);
  if (config.dune_max_num == 0 ||
      point_count <= static_cast<int>(config.dune_max_num)) {
    return indices;
  }

  const Eigen::Matrix3d rotation =
      buildRotations(nominal_states, attitude_horizon).front();
  const Eigen::Vector3d origin = nominal_states.col(0).head<3>();
  const Eigen::Vector3d half = config.body_half_extent.cwiseMax(0.0);
  std::vector<double> score(static_cast<std::size_t>(point_count), 0.0);
  for (int i = 0; i < point_count; ++i) {
    const Eigen::Vector3d local = rotation.transpose() *
                                  (obstacle_points.col(i) - origin);
    score[static_cast<std::size_t>(i)] =
        detail::pointToBoxClearanceSquared(local, half);
  }
  sortByScoreStable(indices, score);
  indices.resize(static_cast<std::size_t>(keep_count));
  return indices;
}

}  // namespace

ObstaclePreselector::ObstaclePreselector(std::size_t max_points)
    : ObstaclePreselector([&] {
        ObstaclePreselectorConfig config;
        config.max_points = max_points;
        return config;
      }()) {}

ObstaclePreselector::ObstaclePreselector(ObstaclePreselectorConfig config)
    : config_(normalizeConfig(std::move(config))) {}

ObstacleSelection ObstaclePreselector::select(const PointMatrix& points) {
  return select(points, emptyPointMatrix());
}

ObstacleSelection ObstaclePreselector::select(
    const PointMatrix& points, const PointMatrix& velocities) {
  return selectByColumnLimit(points, velocities);
}

ObstacleSelection ObstaclePreselector::selectByColumnLimit(
    const PointMatrix& points, const PointMatrix& velocities) const {
  validatePointMatrix(points, "obstacle points");
  validateVelocities(points, velocities);

  const Eigen::Index take =
      config_.max_points == 0
          ? points.cols()
          : std::min<Eigen::Index>(
                points.cols(), static_cast<Eigen::Index>(config_.max_points));

  ObstacleSelection out;
  out.profile.input_obstacle_count = static_cast<std::size_t>(points.cols());
  out.profile.corridor_obstacle_count = static_cast<std::size_t>(points.cols());
  out.profile.preselected_obstacle_count = static_cast<std::size_t>(take);
  out.profile.preselect_max_count = config_.max_points;
  out.points = take == 0 ? emptyPointMatrix() : points.leftCols(take);
  if (velocities.size() != 0) {
    out.velocities =
        take == 0 ? emptyPointMatrix() : velocities.leftCols(take);
  }
  out.tags.assign(static_cast<std::size_t>(take), 0);
  return out;
}

ObstacleSelection ObstaclePreselector::selectWithNominalTrajectory(
    const Trajectory& nominal_states, const PointMatrix& points,
    const PointMatrix& velocities, const Eigen::MatrixXd& attitude_horizon) {
  const auto total_start = Clock::now();
  if (nominal_states.rows() < 4 || nominal_states.cols() <= 0) {
    throw std::invalid_argument(
        "nominal_states must have shape state_dim x steps with at least 4 rows");
  }
  if (!allFinite(nominal_states)) {
    throw std::invalid_argument("nominal_states must be finite");
  }
  validatePointMatrix(points, "obstacle points");
  validateVelocities(points, velocities);
  const bool has_velocities = velocities.size() != 0;

  ObstacleSelection out;
  PlannerProfile& profile = out.profile;
  profile.input_obstacle_count = static_cast<std::size_t>(points.cols());
  profile.corridor_obstacle_count = static_cast<std::size_t>(points.cols());
  profile.preselected_obstacle_count = static_cast<std::size_t>(points.cols());
  profile.preselect_max_count = config_.max_points;

  const int point_count = static_cast<int>(points.cols());
  const int target_max = std::min(static_cast<int>(config_.max_points),
                                  point_count);
  if (point_count == 0 || !config_.enabled || target_max <= 0) {
    out.points = points;
    if (has_velocities) out.velocities = velocities;
    previous_points_ = emptyPointMatrix();
    profile.preselect_sec = elapsedSeconds(total_start);
    return out;
  }

  const int steps = static_cast<int>(nominal_states.cols());
  const std::vector<Eigen::Matrix3d> rotations =
      buildRotations(nominal_states, attitude_horizon);

  Eigen::Vector3d corridor_pad;
  if (attitude_horizon.size() != 0) {
    Eigen::Vector3d max_rotated = Eigen::Vector3d::Zero();
    for (const Eigen::Matrix3d& rotation : rotations) {
      max_rotated =
          max_rotated.cwiseMax(rotation.cwiseAbs() * config_.body_half_extent);
    }
    corridor_pad = max_rotated + config_.corridor_margin;
  } else {
    const double horiz_radius =
        std::hypot(config_.body_half_extent.x(),
                   config_.body_half_extent.y());
    corridor_pad =
        Eigen::Vector3d(horiz_radius + config_.corridor_margin.x(),
                        horiz_radius + config_.corridor_margin.y(),
                        config_.body_half_extent.z() +
                            config_.corridor_margin.z());
  }

  auto section_start = Clock::now();
  Eigen::Vector3d cmin = nominal_states.topRows<3>().rowwise().minCoeff();
  Eigen::Vector3d cmax = nominal_states.topRows<3>().rowwise().maxCoeff();
  cmin -= corridor_pad;
  cmax += corridor_pad;
  const double horizon_sec =
      std::max(0.0, static_cast<double>(steps - 1) * config_.dt);

  std::vector<int> corridor_indices;
  corridor_indices.reserve(static_cast<std::size_t>(point_count));
  for (int i = 0; i < point_count; ++i) {
    bool keep = true;
    for (int r = 0; r < 3; ++r) {
      const double p = points(r, i);
      if (!has_velocities || horizon_sec <= 0.0) {
        if (p < cmin(r) || p > cmax(r)) keep = false;
      } else {
        const double endp = p + velocities(r, i) * horizon_sec;
        if (std::max(p, endp) < cmin(r) ||
            std::min(p, endp) > cmax(r)) {
          keep = false;
        }
      }
      if (!keep) break;
    }
    if (keep) corridor_indices.push_back(i);
  }
  profile.preselect_corridor_sec = elapsedSeconds(section_start);
  profile.corridor_obstacle_count = corridor_indices.size();

  if (corridor_indices.empty()) {
    out.points = emptyPointMatrix();
    if (has_velocities) out.velocities = emptyPointMatrix();
    profile.preselected_obstacle_count = 0;
    previous_points_ = emptyPointMatrix();
    profile.preselect_sec = elapsedSeconds(total_start);
    return out;
  }

  const int corridor_count = static_cast<int>(corridor_indices.size());
  const int select_target = std::min(target_max, corridor_count);
  const Eigen::Vector3d inflated =
      config_.body_half_extent + config_.exact_margin;

  section_start = Clock::now();
  std::vector<double> dist_min_sq(static_cast<std::size_t>(corridor_count),
                                  std::numeric_limits<double>::infinity());
  std::vector<Eigen::Vector3d> representative_local(
      static_cast<std::size_t>(corridor_count), Eigen::Vector3d::Zero());
  for (int ci = 0; ci < corridor_count; ++ci) {
    const int point_idx = corridor_indices[static_cast<std::size_t>(ci)];
    for (int t = 0; t < steps; ++t) {
      const Eigen::Vector3d point =
          pointAtStep(points, velocities, has_velocities, point_idx, t,
                      config_.dt);
      const Eigen::Vector3d local =
          rotations[static_cast<std::size_t>(t)].transpose() *
          (point - nominal_states.col(t).head<3>());
      const double d2 =
          detail::pointToBoxClearanceSquared(local, inflated);
      if (d2 < dist_min_sq[static_cast<std::size_t>(ci)]) {
        dist_min_sq[static_cast<std::size_t>(ci)] = d2;
        representative_local[static_cast<std::size_t>(ci)] = local;
      }
    }
  }
  profile.preselect_distance_sec = elapsedSeconds(section_start);

  section_start = Clock::now();
  const double hard_sq = config_.hard_distance * config_.hard_distance;
  std::vector<int> hard_indices;
  std::vector<int> global_order(static_cast<std::size_t>(corridor_count));
  std::iota(global_order.begin(), global_order.end(), 0);
  sortByScoreStable(global_order, dist_min_sq);
  for (int i = 0; i < corridor_count; ++i) {
    if (dist_min_sq[static_cast<std::size_t>(i)] <= hard_sq) {
      hard_indices.push_back(i);
    }
  }
  sortByScoreStable(hard_indices, dist_min_sq);
  profile.hard_count = hard_indices.size();

  std::vector<int> protected_flat;
  if (config_.per_step > 0) {
    const int per_step = std::min(config_.per_step, corridor_count);
    for (int t = 0; t < steps; ++t) {
      std::vector<std::pair<double, int>> row;
      row.reserve(static_cast<std::size_t>(corridor_count));
      for (int ci = 0; ci < corridor_count; ++ci) {
        const int point_idx = corridor_indices[static_cast<std::size_t>(ci)];
        const Eigen::Vector3d point =
            pointAtStep(points, velocities, has_velocities, point_idx, t,
                        config_.dt);
        const Eigen::Vector3d local =
            rotations[static_cast<std::size_t>(t)].transpose() *
            (point - nominal_states.col(t).head<3>());
        row.emplace_back(detail::pointToBoxClearanceSquared(local, inflated),
                         ci);
      }
      std::stable_sort(row.begin(), row.end(),
                       [](const auto& a, const auto& b) {
                         return a.first < b.first;
                       });
      for (int k = 0; k < per_step; ++k) protected_flat.push_back(row[k].second);
    }
  }

  std::vector<int> protected_indices =
      validUniqueIndices(protected_flat, corridor_count);
  sortByScoreStable(protected_indices, dist_min_sq);
  std::vector<int> nearest_concat = hard_indices;
  nearest_concat.insert(nearest_concat.end(), protected_indices.begin(),
                        protected_indices.end());
  std::vector<int> nearest_indices =
      validUniqueIndices(nearest_concat, corridor_count);
  sortByScoreStable(nearest_indices, dist_min_sq);

  std::vector<int> continuity_indices;
  if (config_.temporal_enabled && previous_points_.cols() > 0) {
    const double radius_sq = config_.temporal_radius * config_.temporal_radius;
    std::vector<std::pair<std::tuple<double, double, int>, int>> hits;
    for (int ci = 0; ci < corridor_count; ++ci) {
      const int point_idx = corridor_indices[static_cast<std::size_t>(ci)];
      double best_sq = std::numeric_limits<double>::infinity();
      for (Eigen::Index pj = 0; pj < previous_points_.cols(); ++pj) {
        best_sq = std::min(
            best_sq,
            (points.col(point_idx) - previous_points_.col(pj)).squaredNorm());
      }
      if (best_sq <= radius_sq) {
        hits.push_back(
            {{dist_min_sq[static_cast<std::size_t>(ci)], best_sq, ci}, ci});
      }
    }
    std::stable_sort(hits.begin(), hits.end(),
                     [](const auto& a, const auto& b) {
                       return a.first < b.first;
                     });
    for (const auto& hit : hits) continuity_indices.push_back(hit.second);
  }
  profile.continuity_hits = continuity_indices.size();

  std::vector<int> diversity_indices;
  if (config_.diversity_enabled && !global_order.empty()) {
    std::unordered_set<VoxelKey, VoxelHash> seen;
    for (int idx : global_order) {
      const VoxelKey key =
          voxelKey(representative_local[static_cast<std::size_t>(idx)],
                   config_.diversity_voxel);
      if (seen.insert(key).second) diversity_indices.push_back(idx);
    }
  }
  profile.diversity_candidates = diversity_indices.size();

  const double temporal_ratio_eff =
      config_.temporal_enabled ? config_.temporal_ratio : 0.0;
  const double diversity_ratio_eff =
      config_.diversity_enabled ? config_.diversity_ratio : 0.0;
  const std::array<int, 3> quotas =
      quotaCounts(select_target, config_.nearest_ratio, temporal_ratio_eff,
                  diversity_ratio_eff);
  profile.nearest_quota = static_cast<std::size_t>(quotas[0]);
  profile.temporal_quota = static_cast<std::size_t>(quotas[1]);
  profile.diversity_quota = static_cast<std::size_t>(quotas[2]);

  std::vector<uint8_t> nearest_mask(static_cast<std::size_t>(corridor_count), 0);
  std::vector<uint8_t> temporal_mask(static_cast<std::size_t>(corridor_count), 0);
  std::vector<uint8_t> diversity_mask(static_cast<std::size_t>(corridor_count), 0);
  for (int idx : nearest_indices) nearest_mask[static_cast<std::size_t>(idx)] = 1;
  for (int idx : continuity_indices) {
    temporal_mask[static_cast<std::size_t>(idx)] = 1;
  }
  for (int idx : diversity_indices) {
    diversity_mask[static_cast<std::size_t>(idx)] = 1;
  }

  std::vector<int> chosen_indices;
  if (corridor_count <= select_target) {
    chosen_indices.resize(static_cast<std::size_t>(corridor_count));
    std::iota(chosen_indices.begin(), chosen_indices.end(), 0);
  } else {
    std::vector<uint8_t> selected_mask(
        static_cast<std::size_t>(corridor_count), 0);
    appendRanked(chosen_indices, selected_mask, nearest_indices, select_target,
                 quotas[0]);
    int temporal_selected_now = 0;
    for (int idx : chosen_indices) {
      temporal_selected_now +=
          temporal_mask[static_cast<std::size_t>(idx)] != 0 ? 1 : 0;
    }
    appendRanked(chosen_indices, selected_mask, continuity_indices,
                 select_target, std::max(0, quotas[1] - temporal_selected_now));
    int diversity_selected_now = 0;
    for (int idx : chosen_indices) {
      diversity_selected_now +=
          diversity_mask[static_cast<std::size_t>(idx)] != 0 ? 1 : 0;
    }
    appendRanked(chosen_indices, selected_mask, diversity_indices,
                 select_target,
                 std::max(0, quotas[2] - diversity_selected_now));
    appendRanked(chosen_indices, selected_mask, global_order, select_target);
  }

  profile.preselect_select_sec = elapsedSeconds(section_start);
  profile.preselected_obstacle_count = chosen_indices.size();
  out.points.resize(3, static_cast<Eigen::Index>(chosen_indices.size()));
  if (has_velocities) {
    out.velocities.resize(3, static_cast<Eigen::Index>(chosen_indices.size()));
  }
  out.tags.assign(chosen_indices.size(), 0);
  for (Eigen::Index out_i = 0;
       out_i < static_cast<Eigen::Index>(chosen_indices.size()); ++out_i) {
    const int ci = chosen_indices[static_cast<std::size_t>(out_i)];
    const int src = corridor_indices[static_cast<std::size_t>(ci)];
    out.points.col(out_i) = points.col(src);
    if (has_velocities) out.velocities.col(out_i) = velocities.col(src);

    unsigned char tag = 0;
    if (temporal_mask[static_cast<std::size_t>(ci)] != 0) {
      tag |= kTemporalSelectionTag;
      ++profile.continuity_selected;
    }
    if (diversity_mask[static_cast<std::size_t>(ci)] != 0) {
      tag |= kDiversitySelectionTag;
      ++profile.diversity_selected;
    }
    if (nearest_mask[static_cast<std::size_t>(ci)] != 0) {
      ++profile.nearest_selected;
    }
    if ((nearest_mask[static_cast<std::size_t>(ci)] == 0) &&
        (temporal_mask[static_cast<std::size_t>(ci)] == 0) &&
        (diversity_mask[static_cast<std::size_t>(ci)] == 0)) {
      ++profile.fill_selected;
    }
    out.tags[static_cast<std::size_t>(out_i)] = tag;
  }
  profile.preselect_sec = elapsedSeconds(total_start);

  if (config_.enabled && config_.max_points > 0 && out.points.cols() > 0) {
    previous_points_ = out.points;
  } else {
    previous_points_ = emptyPointMatrix();
  }
  return out;
}

void ObstaclePreselector::reset() { previous_points_ = emptyPointMatrix(); }

void ObstaclePreselector::setPreviousPoints(const PointMatrix& points) {
  validatePointMatrix(points, "previous points");
  previous_points_ = points.cols() == 0 ? emptyPointMatrix() : points;
}

PointFlowBuilder::PointFlowBuilder(PointFlowConfig config)
    : config_(std::move(config)) {
  if (config_.receding < 0) {
    throw std::invalid_argument("PointFlowConfig::receding must be non-negative");
  }
  config_.dt = finitePositive(config_.dt, "PointFlowConfig::dt");
  config_.body_half_extent =
      nonnegativeVector(config_.body_half_extent,
                        "PointFlowConfig::body_half_extent");
}

PointFlowResult PointFlowBuilder::generate(
    const Trajectory& nominal_states, const PointMatrix& obstacle_points,
    const PointMatrix& velocities, const Eigen::MatrixXd& attitude_horizon) const {
  const int steps = config_.receding + 1;
  if (steps <= 0) {
    throw std::invalid_argument("PointFlowConfig::receding must be non-negative");
  }
  if (nominal_states.rows() < 3 || nominal_states.cols() < steps) {
    throw std::invalid_argument(
        "nominal_states must have at least 3 rows and receding+1 columns");
  }
  if (!allFinite(nominal_states)) {
    throw std::invalid_argument("nominal_states must be finite");
  }
  validatePointMatrix(obstacle_points, "obstacle points");
  validateVelocities(obstacle_points, velocities);
  const bool has_velocities = velocities.size() != 0;

  const Trajectory active_nominal = nominal_states.leftCols(steps);
  const std::vector<int> indices =
      fallbackPointIndices(config_, active_nominal, obstacle_points,
                           attitude_horizon);
  const bool fallback_applied =
      static_cast<Eigen::Index>(indices.size()) < obstacle_points.cols();

  PointFlowResult result;
  result.fallback_applied = fallback_applied;
  result.selected_points =
      fallback_applied ? columnsByIndex(obstacle_points, indices)
                       : obstacle_points;
  if (has_velocities) {
    result.selected_velocities =
        fallback_applied ? columnsByIndex(velocities, indices) : velocities;
  }

  const int point_count = static_cast<int>(result.selected_points.cols());
  result.point_flow.reserve(static_cast<std::size_t>(steps));
  result.rotations = buildRotations(active_nominal, attitude_horizon);
  result.obstacle_points_by_step.reserve(static_cast<std::size_t>(steps));

  for (int t = 0; t < steps; ++t) {
    PointMatrix step_points(3, point_count);
    PointMatrix flow(3, point_count);
    const Eigen::Vector3d origin = active_nominal.col(t).head<3>();
    const Eigen::Matrix3d& rotation =
        result.rotations[static_cast<std::size_t>(t)];
    for (int i = 0; i < point_count; ++i) {
      const Eigen::Vector3d point =
          pointAtStep(result.selected_points, result.selected_velocities,
                      has_velocities, i, t, config_.dt);
      step_points.col(i) = point;
      flow.col(i) = rotation.transpose() * (point - origin);
    }
    result.obstacle_points_by_step.push_back(std::move(step_points));
    result.point_flow.push_back(std::move(flow));
  }
  return result;
}

}  // namespace neupan_uav
