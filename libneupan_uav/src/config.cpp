#include "neupan_uav/config.hpp"

#include <unsupported/Eigen/MatrixFunctions>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace neupan_uav {

namespace {

constexpr int kUavStateDim = 8;
constexpr int kUavControlDim = 4;
constexpr int kPointDim = 3;
constexpr int kDuneEdgeDim = 6;

bool finiteVector(const Eigen::VectorXd& vector) {
  for (Eigen::Index i = 0; i < vector.size(); ++i) {
    if (!std::isfinite(vector(i))) return false;
  }
  return true;
}

bool nonNegativeBoundVector(const Eigen::VectorXd& vector) {
  for (Eigen::Index i = 0; i < vector.size(); ++i) {
    const double value = vector(i);
    if (std::isnan(value) || value < 0.0) return false;
  }
  return true;
}

void validatePlannerOptions(const PlannerOptions& options,
                            const UavDynamicsConfig& dynamics,
                            double state_weight_gain) {
  if (options.grid.horizon_steps <= 0) {
    throw std::invalid_argument("PredictionGrid::horizon_steps must be positive");
  }
  if (!(options.grid.dt > 0.0) || !std::isfinite(options.grid.dt)) {
    throw std::invalid_argument("PredictionGrid::dt must be finite and positive");
  }
  if (options.collision_threshold < 0.0 ||
      !std::isfinite(options.collision_threshold)) {
    throw std::invalid_argument(
        "PlannerOptions::collision_threshold must be finite and non-negative");
  }
  if (!finiteVector(options.robot.body_half_extent) ||
      (options.robot.body_half_extent.array() < 0.0).any()) {
    throw std::invalid_argument(
        "RobotSpec::body_half_extent must be finite and non-negative");
  }
  if (!nonNegativeBoundVector(options.robot.max_control)) {
    throw std::invalid_argument("RobotSpec::max_control must be non-negative");
  }
  if (options.nrmp.max_constraints < 0) {
    throw std::invalid_argument(
        "NrmpOptions::max_constraints must be non-negative");
  }
  if (!(options.pan.trajectory_threshold > 0.0) ||
      !std::isfinite(options.pan.trajectory_threshold)) {
    throw std::invalid_argument(
        "PanOptions::trajectory_threshold must be finite and positive");
  }
  if (!(options.pan.dune_threshold > 0.0) ||
      !std::isfinite(options.pan.dune_threshold)) {
    throw std::invalid_argument(
        "PanOptions::dune_threshold must be finite and positive");
  }
  if (!nonNegativeBoundVector(dynamics.max_acceleration)) {
    throw std::invalid_argument(
        "UavDynamicsConfig::max_acceleration must be non-negative");
  }
  if (!finiteVector(dynamics.velocity_time_constant) ||
      (dynamics.velocity_time_constant.array() <= 0.0).any()) {
    throw std::invalid_argument(
        "UavDynamicsConfig::velocity_time_constant must be finite and positive");
  }
  if (!finiteVector(dynamics.velocity_gain)) {
    throw std::invalid_argument(
        "UavDynamicsConfig::velocity_gain must be finite");
  }
  if (!(dynamics.yaw_rate_time_constant > 0.0) ||
      !std::isfinite(dynamics.yaw_rate_time_constant)) {
    throw std::invalid_argument(
        "UavDynamicsConfig::yaw_rate_time_constant must be finite and positive");
  }
  if (!std::isfinite(dynamics.yaw_rate_gain)) {
    throw std::invalid_argument(
        "UavDynamicsConfig::yaw_rate_gain must be finite");
  }
  if (!std::isfinite(dynamics.velocity_weight_scale)) {
    throw std::invalid_argument(
        "UavDynamicsConfig::velocity_weight_scale must be finite");
  }
  if (state_weight_gain < 0.0 || !std::isfinite(state_weight_gain)) {
    throw std::invalid_argument(
        "state_weight_gain must be finite and non-negative");
  }
}

RobotModelConfig makeRobotConfig(const RobotSpec& spec) {
  RobotModelConfig robot;
  robot.body_half_extent = spec.body_half_extent;
  robot.max_control = spec.max_control;
  return robot;
}

Eigen::MatrixXd continuousA(const UavDynamicsConfig& dynamics) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(kUavStateDim, kUavStateDim);
  A(0, 4) = 1.0;
  A(1, 5) = 1.0;
  A(2, 6) = 1.0;
  A(3, 7) = 1.0;
  for (int i = 0; i < kPointDim; ++i) {
    A(4 + i, 4 + i) = -1.0 / dynamics.velocity_time_constant(i);
  }
  A(7, 7) = -1.0 / dynamics.yaw_rate_time_constant;
  return A;
}

Eigen::MatrixXd continuousB(const UavDynamicsConfig& dynamics) {
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(kUavStateDim, kUavControlDim);
  for (int i = 0; i < kPointDim; ++i) {
    B(4 + i, i) =
        dynamics.velocity_gain(i) / dynamics.velocity_time_constant(i);
  }
  B(7, 3) = dynamics.yaw_rate_gain / dynamics.yaw_rate_time_constant;
  return B;
}

void discretizeDynamics(const UavDynamicsConfig& dynamics, double dt,
                        Eigen::MatrixXd& A, Eigen::MatrixXd& B,
                        Eigen::VectorXd& C) {
  const Eigen::MatrixXd Ac = continuousA(dynamics);
  const Eigen::MatrixXd Bc = continuousB(dynamics);
  Eigen::MatrixXd aug = Eigen::MatrixXd::Zero(
      kUavStateDim + kUavControlDim, kUavStateDim + kUavControlDim);
  aug.block(0, 0, kUavStateDim, kUavStateDim) = Ac;
  aug.block(0, kUavStateDim, kUavStateDim, kUavControlDim) = Bc;
  const Eigen::MatrixXd expm = (aug * dt).exp();
  A = expm.block(0, 0, kUavStateDim, kUavStateDim);
  B = expm.block(0, kUavStateDim, kUavStateDim, kUavControlDim);
  C = Eigen::VectorXd::Zero(kUavStateDim);
}

DunePostprocessorConfig makeDuneConfig(
    const PlannerOptions& options,
    const RknnMetadata& metadata) {
  DunePostprocessorConfig dune;
  dune.receding = options.grid.horizon_steps;
  dune.point_dim = kPointDim;
  dune.dune_max_num = static_cast<std::size_t>(metadata.dune_max_num);
  dune.select_num =
      static_cast<std::size_t>(options.nrmp.max_constraints);
  dune.edge_dim = metadata.output_dim;
  dune.select_nearest_ratio = options.dune->select_nearest_ratio;
  dune.select_temporal_ratio = options.dune->select_temporal_ratio;
  dune.select_diversity_ratio = options.dune->select_diversity_ratio;
  dune.G.resize(kDuneEdgeDim, kPointDim);
  dune.G << 1.0F, 0.0F, 0.0F,
      -1.0F, 0.0F, 0.0F,
      0.0F, 1.0F, 0.0F,
      0.0F, -1.0F, 0.0F,
      0.0F, 0.0F, 1.0F,
      0.0F, 0.0F, -1.0F;
  dune.h.resize(kDuneEdgeDim);
  dune.h << static_cast<float>(options.robot.body_half_extent(0)),
      static_cast<float>(options.robot.body_half_extent(0)),
      static_cast<float>(options.robot.body_half_extent(1)),
      static_cast<float>(options.robot.body_half_extent(1)),
      static_cast<float>(options.robot.body_half_extent(2)),
      static_cast<float>(options.robot.body_half_extent(2));
  return dune;
}

NrmpConfig makeNrmpConfig(const PlannerOptions& options,
                          const UavDynamicsConfig& dynamics) {
  NrmpConfig nrmp;
  nrmp.receding = options.grid.horizon_steps;
  nrmp.state_dim = kUavStateDim;
  nrmp.control_dim = kUavControlDim;
  nrmp.geom_dim = kPointDim;
  nrmp.point_dim = kPointDim;
  nrmp.max_num = options.nrmp.max_constraints;
  nrmp.no_obs = options.nrmp.max_constraints <= 0;
  nrmp.enable_control_smoothing = options.nrmp.enable_control_smoothing;
  nrmp.solver_options = options.nrmp.solver;
  discretizeDynamics(dynamics, options.grid.dt, nrmp.dynamics_A,
                     nrmp.dynamics_B, nrmp.dynamics_C);
  nrmp.speed_bound = options.robot.max_control;
  nrmp.acce_bound = dynamics.max_acceleration * options.grid.dt;
  nrmp.tracking_speed_bound = options.robot.max_control.cwiseAbs();
  return nrmp;
}

PointFlowConfig makePointFlowConfig(const PlannerOptions& options,
                                    int dune_max_num) {
  PointFlowConfig point_flow;
  point_flow.receding = options.grid.horizon_steps;
  point_flow.dt = options.grid.dt;
  point_flow.dune_max_num = static_cast<std::size_t>(std::max(0, dune_max_num));
  point_flow.body_half_extent = options.robot.body_half_extent;
  return point_flow;
}

ObstaclePreselectorConfig makePreselectorConfig(
    const PlannerOptions& options,
    int max_points) {
  ObstaclePreselectorConfig preselect = options.preselect;
  preselect.dt = options.grid.dt;
  preselect.body_half_extent = options.robot.body_half_extent;
  preselect.max_points = static_cast<std::size_t>(std::max(0, max_points));
  return preselect;
}

}  // namespace

CompiledPlannerConfig compilePlannerConfig(
    const PlannerOptions& options,
    const UavDynamicsConfig& dynamics,
    double state_weight_gain,
    const std::optional<RknnMetadata>& metadata) {
  validatePlannerOptions(options, dynamics, state_weight_gain);

  const int dune_capacity =
      options.dune.has_value() && metadata.has_value() ? metadata->dune_max_num : 0;
  if (options.dune.has_value()) {
    if (!metadata.has_value()) {
      throw std::invalid_argument(
          "DUNE enabled but RKNN metadata is unavailable");
    }
    RknnRuntimeContract contract;
    contract.receding = options.grid.horizon_steps;
    contract.dune_max_num = metadata->dune_max_num;
    contract.output_dim = metadata->output_dim;
    contract.body_half_extent = options.robot.body_half_extent;
    metadata->validateRuntime(contract);
    if (options.nrmp.max_constraints > metadata->dune_max_num) {
      throw std::invalid_argument(
          "NRMP max_constraints exceeds DUNE capacity");
    }
  }

  CompiledPlannerConfig out;
  out.grid_ = options.grid;
  out.ref_speed_ = options.ref_speed;
  out.collision_threshold_ = options.collision_threshold;
  out.arrive_threshold_ = options.arrive_threshold;
  out.has_goal_ = options.has_goal;
  out.goal_position_ = options.goal_position;
  out.placeholder_command_ = options.placeholder_command;
  out.robot_ = makeRobotConfig(options.robot);
  out.preselect_ = makePreselectorConfig(options, dune_capacity);
  out.farfield_guide_ = options.farfield_guide;
  out.initial_path_ = options.initial_path;

  out.pan_.grid = options.grid;
  out.pan_.iter_num = options.pan.iter_num;
  out.pan_.trajectory_threshold = options.pan.trajectory_threshold;
  out.pan_.dune_threshold = options.pan.dune_threshold;
  out.pan_.nrmp = makeNrmpConfig(options, dynamics);
  out.pan_.point_flow = makePointFlowConfig(options, dune_capacity);
  out.pan_.state_weights = Eigen::VectorXd::Ones(kUavStateDim);
  out.pan_.state_weights.tail<kUavControlDim>().setConstant(
      state_weight_gain * dynamics.velocity_weight_scale);
  out.pan_.p_u = options.pan.p_u;
  out.pan_.eta = options.pan.eta;
  out.pan_.d_min = options.pan.d_min;
  out.pan_.d_max = options.pan.d_max;
  out.pan_.ro_obs = options.pan.ro_obs;
  out.pan_.bk = options.pan.bk;
  out.pan_.smooth_du = options.pan.smooth_du;
  out.pan_.smooth_u0 = options.pan.smooth_u0;

  if (options.dune.has_value()) {
    if (!options.dune->metadata_path.empty()) {
      out.pan_.rknn_mode = RknnRunnerMode::kRuntime;
      out.pan_.rknn_metadata_path = options.dune->metadata_path;
      out.pan_.rknn_core_mask = options.dune->core_mask;
      out.pan_.rknn_require_device = options.dune->require_device;
    }
    out.pan_.dune = makeDuneConfig(options, *metadata);
  }

  return out;
}

}  // namespace neupan_uav
