#pragma once

#include "neupan_uav/dune.hpp"
#include "neupan_uav/farfield_guide.hpp"
#include "neupan_uav/nrmp.hpp"
#include "neupan_uav/obstacle_preselector.hpp"
#include "neupan_uav/rknn_runner.hpp"
#include "neupan_uav/robot_model.hpp"
#include "neupan_uav/types.hpp"

#include <Eigen/Dense>

#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace neupan_uav {

enum class RknnRunnerMode {
  kDisabled,
  kRuntime,
};

struct PredictionGrid {
  int horizon_steps = 10;
  double dt = 0.1;
};

struct RobotSpec {
  Eigen::Vector3d body_half_extent = Eigen::Vector3d::Zero();
  Control max_control =
      Control::Constant(std::numeric_limits<double>::infinity());
};

struct DuneOptions {
  std::string metadata_path;
  std::string core_mask = "CORE_0_1_2";
  bool require_device = true;
  double select_nearest_ratio = 0.60;
  double select_temporal_ratio = 0.20;
  double select_diversity_ratio = 0.20;
};

struct NrmpOptions {
  int max_constraints = 0;
  bool enable_control_smoothing = false;
  NrmpSolverOptions solver;
};

struct PanOptions {
  int iter_num = 2;
  double trajectory_threshold = 0.05;  // 状态/控制轨迹 RMS 变化收敛阈值
  double dune_threshold = 0.01;        // DUNE mu/lambda RMS 变化收敛阈值
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

struct PlannerOptions {
  PredictionGrid grid;
  RobotSpec robot;
  std::optional<DuneOptions> dune;
  NrmpOptions nrmp;
  ObstaclePreselectorConfig preselect;
  FarfieldGuideConfig farfield_guide;
  InitialPathConfig initial_path;
  PanOptions pan;
  double ref_speed = 1.0;
  double collision_threshold = 0.1;
  double arrive_threshold = 0.2;
  bool has_goal = false;
  Eigen::Vector3d goal_position = Eigen::Vector3d::Zero();
  Control placeholder_command = Control::Zero();
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

struct PanConfig {
  PredictionGrid grid;
  int iter_num = 2;
  double trajectory_threshold = 0.05;
  double dune_threshold = 0.01;
  RknnRunnerMode rknn_mode = RknnRunnerMode::kDisabled;
  std::string rknn_metadata_path;
  std::string rknn_core_mask = "CORE_0_1_2";
  bool rknn_require_device = true;
  std::optional<DunePostprocessorConfig> dune;
  NrmpConfig nrmp;
  PointFlowConfig point_flow;
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

class CompiledPlannerConfig {
 public:
  const PredictionGrid& grid() const { return grid_; }
  int receding() const { return grid_.horizon_steps; }
  double stepTime() const { return grid_.dt; }
  double refSpeed() const { return ref_speed_; }
  double collisionThreshold() const { return collision_threshold_; }
  double arriveThreshold() const { return arrive_threshold_; }
  bool hasGoal() const { return has_goal_; }
  const Eigen::Vector3d& goalPosition() const { return goal_position_; }
  const Control& placeholderCommand() const { return placeholder_command_; }
  const RobotModelConfig& robot() const { return robot_; }
  const ObstaclePreselectorConfig& preselect() const { return preselect_; }
  const FarfieldGuideConfig& farfieldGuide() const { return farfield_guide_; }
  const InitialPathConfig& initialPath() const { return initial_path_; }
  const PanConfig& pan() const { return pan_; }

 private:
  friend CompiledPlannerConfig compilePlannerConfig(
      const PlannerOptions& options,
      const UavDynamicsConfig& dynamics,
      double state_weight_gain,
      const std::optional<RknnMetadata>& metadata);

  PredictionGrid grid_;
  double ref_speed_ = 1.0;
  double collision_threshold_ = 0.1;
  double arrive_threshold_ = 0.2;
  bool has_goal_ = false;
  Eigen::Vector3d goal_position_ = Eigen::Vector3d::Zero();
  Control placeholder_command_ = Control::Zero();
  RobotModelConfig robot_;
  ObstaclePreselectorConfig preselect_;
  FarfieldGuideConfig farfield_guide_;
  InitialPathConfig initial_path_;
  PanConfig pan_;
};

CompiledPlannerConfig compilePlannerConfig(
    const PlannerOptions& options,
    const UavDynamicsConfig& dynamics = UavDynamicsConfig(),
    double state_weight_gain = 1.0,
    const std::optional<RknnMetadata>& metadata = std::nullopt);

}  // namespace neupan_uav
