#pragma once

#include "neupan_uav/types.hpp"

#include <Eigen/Dense>

#include <memory>
#include <string>

namespace neupan_uav {

struct NrmpSolverOptions {
  bool verbose = false;
  bool warm_starting = true;
  bool polishing = false;
  double eps_abs = 1.0e-4;
  double eps_rel = 1.0e-4;
  int max_iter = 4000;
};

struct NrmpConfig {
  int receding = 0;
  int state_dim = 0;
  int control_dim = 0;
  int geom_dim = 0;
  int point_dim = 3;
  int max_num = 0;
  bool no_obs = true;
  bool enable_control_smoothing = false;

  Eigen::MatrixXd dynamics_A;
  Eigen::MatrixXd dynamics_B;
  Eigen::VectorXd dynamics_C;
  Eigen::VectorXd speed_bound;
  Eigen::VectorXd acce_bound;
  Eigen::VectorXd tracking_speed_bound;
  NrmpSolverOptions solver_options;
};

struct NrmpInput {
  Control seed_control = Control::Zero();
  Control desired_control = Control::Zero();

  Trajectory nominal_states;
  Trajectory reference_states;
  Eigen::MatrixXd reference_controls;
  Eigen::VectorXd state_weights;
  double p_u = 1.0;
  double eta = 10.0;
  double d_min = 0.1;
  double d_max = 1.0;
  double ro_obs = 400.0;
  double bk = 0.1;
  double smooth_du = 0.0;
  double smooth_u0 = 0.0;
  Eigen::MatrixXd fa_batch;
  Eigen::MatrixXd fb_batch;
};

struct NrmpResult {
  Control control = Control::Zero();
  Trajectory state_trajectory;
  Eigen::MatrixXd control_trajectory;
  Eigen::RowVectorXd nominal_distance;
  int status = 0;
  int status_val = 0;
  int iterations = 0;
  double solve_sec = 0.0;
  double run_time_sec = 0.0;
  int setup_count = 0;
  int solve_count = 0;
  std::string status_text;
};

class NRMP {
 public:
  NRMP();
  explicit NRMP(NrmpConfig config);
  ~NRMP();

  NRMP(NRMP&&) noexcept;
  NRMP& operator=(NRMP&&) noexcept;
  NRMP(const NRMP&) = delete;
  NRMP& operator=(const NRMP&) = delete;

  bool hasBackend() const;
  const NrmpConfig& config() const;
  NrmpResult solve(const NrmpInput& input);

 private:
  class Backend;

  NrmpConfig config_;
  std::unique_ptr<Backend> backend_;
};

}  // namespace neupan_uav
