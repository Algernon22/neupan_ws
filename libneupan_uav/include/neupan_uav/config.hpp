#pragma once

#include "neupan_uav/dune.hpp"
#include "neupan_uav/farfield_guide.hpp"
#include "neupan_uav/nrmp.hpp"
#include "neupan_uav/obstacle_preselector.hpp"
#include "neupan_uav/robot_model.hpp"
#include "neupan_uav/types.hpp"

#include <Eigen/Dense>

#include <limits>
#include <string>
#include <vector>

namespace neupan_uav {

enum class RknnRunnerMode {
  kDisabled,
  kMock,
  kRuntime,
};

struct PanConfig {
  int receding = 10;
  double step_time = 0.1;
  int iter_num = 2;
  double iter_threshold = 0.1;
  int dune_max_num = 0;
  int nrmp_max_num = 0;
  RknnRunnerMode rknn_mode = RknnRunnerMode::kDisabled;
  std::string rknn_metadata_path;
  std::string rknn_core_mask = "CORE_0_1_2";
  bool rknn_require_device = true;
  DunePostprocessorConfig dune;
  NrmpConfig nrmp;
  PointFlowConfig point_flow;
  bool has_nrmp_config = false;
  bool has_dune_config = false;
  Eigen::VectorXd state_weights;
  double p_u = 1.0;
  double eta = 10.0;
  double d_min = 0.1;
  double d_max = 1.0;
  double ro_obs = 400.0;
  double bk = 0.1;
  double smooth_du = 0.0;
  double smooth_u0 = 0.0;
};

struct InitialPathConfig {
  // Waypoints are [x, y, z, yaw]. When present, they take precedence over the
  // single-goal fallback for reference generation and arrive checks.
  std::vector<Eigen::Vector4d> waypoints;
  bool monotonic_progress = true;
};

struct PlannerConfig {
  int receding = 10;
  double step_time = 0.1;
  double ref_speed = 1.0;
  double collision_threshold = 0.1;
  double arrive_threshold = 0.2;
  bool has_goal = false;
  Eigen::Vector3d goal_position = Eigen::Vector3d::Zero();
  Control placeholder_command = Control::Zero();
  RobotModelConfig robot;
  ObstaclePreselectorConfig preselect;
  FarfieldGuideConfig farfield_guide;
  InitialPathConfig initial_path;
  PanConfig pan;
};

struct UavDynamicsConfig {
  Control max_acceleration =
      Control::Constant(std::numeric_limits<double>::infinity());
  Eigen::Vector3d velocity_time_constant =
      Eigen::Vector3d(0.35, 0.35, 0.45);
  Eigen::Vector3d velocity_gain = Eigen::Vector3d::Ones();
  double yaw_rate_time_constant = 0.30;
  double yaw_rate_gain = 1.0;
  double velocity_weight_scale = 0.35;
};

struct UavPlannerConfigSpec {
  // Contains only fields that users configure directly. Derived planner,
  // PAN, DUNE, and NRMP fields are populated by buildUavPlannerConfig().
  PlannerConfig planner;
  UavDynamicsConfig dynamics;
  double state_weight_gain = 1.0;
};

PlannerConfig buildUavPlannerConfig(UavPlannerConfigSpec spec);

}  // namespace neupan_uav
