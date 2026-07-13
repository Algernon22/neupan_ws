#pragma once

#include <Eigen/Dense>

#include <cstddef>
#include <limits>
#include <string>

namespace neupan_uav {

using Scalar = double;
using Control = Eigen::Matrix<Scalar, 4, 1>;
using PointMatrix = Eigen::Matrix<Scalar, 3, Eigen::Dynamic>;
using Trajectory = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

inline Control zeroControl() { return Control::Zero(); }

inline PointMatrix emptyPointMatrix() { return PointMatrix(3, 0); }

struct PlannerProfile {
  double forward_sec = 0.0;
  double preselect_sec = 0.0;
  double preselect_corridor_sec = 0.0;
  double preselect_distance_sec = 0.0;
  double preselect_select_sec = 0.0;
  double dune_sec = 0.0;
  double dune_inference_sec = 0.0;
  double dune_select_sec = 0.0;
  double nrmp_sec = 0.0;
  std::size_t input_obstacle_count = 0;
  std::size_t corridor_obstacle_count = 0;
  std::size_t preselected_obstacle_count = 0;
  std::size_t preselect_max_count = 0;
  std::size_t hard_count = 0;
  std::size_t nearest_quota = 0;
  std::size_t nearest_selected = 0;
  std::size_t temporal_quota = 0;
  std::size_t continuity_hits = 0;
  std::size_t continuity_selected = 0;
  std::size_t diversity_quota = 0;
  std::size_t diversity_candidates = 0;
  std::size_t diversity_selected = 0;
  std::size_t fill_selected = 0;
  std::size_t dune_selected_count = 0;
  int pan_iterations = 0;
  int pan_iteration_limit = 0;
  int osqp_status = 0;
  int osqp_iteration_count = 0;
  double osqp_solve_sec = 0.0;
  double osqp_run_time_sec = 0.0;
  double farfield_sec = 0.0;
  bool farfield_active = false;
  double farfield_near_m = 0.0;
  double farfield_far_m = 0.0;
  double farfield_offset_m = 0.0;
  double farfield_target_offset_m = 0.0;
  int farfield_center_count = 0;
  int farfield_left_count = 0;
  int farfield_right_count = 0;
  int farfield_release_streak = 0;
};

struct PlannerInput {
  Eigen::VectorXd state;
  PointMatrix obstacle_points = emptyPointMatrix();
  PointMatrix obstacle_velocities = emptyPointMatrix();
  double stamp_sec = 0.0;
  bool valid = true;
  bool stale = false;
};

struct PlannerOutput {
  Control command = Control::Zero();
  Trajectory trajectory;
  Trajectory reference;
  Eigen::MatrixXd control_trajectory;
  Eigen::RowVectorXd nominal_distance;
  bool ready = true;
  std::string reason = "planner_ok";
  bool arrive = false;
  bool stop = false;
  double min_distance = std::numeric_limits<double>::infinity();
  PlannerProfile profile;

  // The command used as this planning cycle's warm-start seed. It must be the
  // command that was actually published/applied in the previous cycle.
  Control seed_control = Control::Zero();
};

}  // namespace neupan_uav
