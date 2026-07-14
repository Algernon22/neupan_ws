#include "neupan_uav/config.hpp"

#include <unsupported/Eigen/MatrixFunctions>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

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

void validateSpec(const UavPlannerConfigSpec& spec) {
  const PlannerConfig& planner = spec.planner;
  const UavDynamicsConfig& dynamics = spec.dynamics;

  if (planner.receding <= 0) {
    throw std::invalid_argument("PlannerConfig::receding must be positive");
  }
  if (!(planner.step_time > 0.0) || !std::isfinite(planner.step_time)) {
    throw std::invalid_argument(
        "PlannerConfig::step_time must be finite and positive");
  }
  if (planner.collision_threshold < 0.0 ||
      !std::isfinite(planner.collision_threshold)) {
    throw std::invalid_argument(
        "PlannerConfig::collision_threshold must be finite and non-negative");
  }
  if (!finiteVector(planner.robot.body_half_extent) ||
      (planner.robot.body_half_extent.array() < 0.0).any()) {
    throw std::invalid_argument(
        "RobotModelConfig::body_half_extent must be finite and non-negative");
  }
  if (!nonNegativeBoundVector(planner.robot.max_control)) {
    throw std::invalid_argument(
        "RobotModelConfig::max_control must be non-negative");
  }
  if (planner.pan.dune_max_num < 0) {
    throw std::invalid_argument("PanConfig::dune_max_num must be non-negative");
  }
  if (planner.pan.nrmp_max_num < 0) {
    throw std::invalid_argument("PanConfig::nrmp_max_num must be non-negative");
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
  if (spec.state_weight_gain < 0.0 || !std::isfinite(spec.state_weight_gain)) {
    throw std::invalid_argument(
        "UavPlannerConfigSpec::state_weight_gain must be finite and non-negative");
  }
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

void configureDune(PlannerConfig& config) {
  config.pan.dune.receding = config.receding;
  config.pan.dune.point_dim = kPointDim;
  config.pan.dune.dune_max_num =
      static_cast<std::size_t>(std::max(0, config.pan.dune_max_num));
  config.pan.dune.select_num =
      static_cast<std::size_t>(std::max(0, config.pan.nrmp_max_num));
  config.pan.dune.edge_dim = kDuneEdgeDim;
  config.pan.dune.G.resize(kDuneEdgeDim, kPointDim);
  config.pan.dune.G << 1.0F, 0.0F, 0.0F,
      -1.0F, 0.0F, 0.0F,
      0.0F, 1.0F, 0.0F,
      0.0F, -1.0F, 0.0F,
      0.0F, 0.0F, 1.0F,
      0.0F, 0.0F, -1.0F;
  config.pan.dune.h.resize(kDuneEdgeDim);
  config.pan.dune.h << static_cast<float>(config.robot.body_half_extent(0)),
      static_cast<float>(config.robot.body_half_extent(0)),
      static_cast<float>(config.robot.body_half_extent(1)),
      static_cast<float>(config.robot.body_half_extent(1)),
      static_cast<float>(config.robot.body_half_extent(2)),
      static_cast<float>(config.robot.body_half_extent(2));
  config.pan.has_dune_config = config.pan.dune_max_num > 0;
}

void configureNrmp(PlannerConfig& config, const UavDynamicsConfig& dynamics,
                   double state_weight_gain) {
  config.pan.nrmp.receding = config.receding;
  config.pan.nrmp.state_dim = kUavStateDim;
  config.pan.nrmp.control_dim = kUavControlDim;
  config.pan.nrmp.geom_dim = kPointDim;
  config.pan.nrmp.point_dim = kPointDim;
  config.pan.nrmp.max_num = config.pan.nrmp_max_num;
  config.pan.nrmp.no_obs = config.pan.nrmp_max_num <= 0;
  discretizeDynamics(dynamics, config.step_time, config.pan.nrmp.dynamics_A,
                     config.pan.nrmp.dynamics_B, config.pan.nrmp.dynamics_C);
  config.pan.nrmp.speed_bound = config.robot.max_control;
  config.pan.nrmp.acce_bound = dynamics.max_acceleration * config.step_time;
  config.pan.nrmp.tracking_speed_bound = config.robot.max_control.cwiseAbs();
  config.pan.has_nrmp_config = config.pan.nrmp_max_num > 0;

  config.pan.state_weights = Eigen::VectorXd::Ones(kUavStateDim);
  config.pan.state_weights.tail<kUavControlDim>().setConstant(
      state_weight_gain * dynamics.velocity_weight_scale);
}

}  // namespace

PlannerConfig buildUavPlannerConfig(UavPlannerConfigSpec spec) {
  validateSpec(spec);

  PlannerConfig config = std::move(spec.planner);

  config.pan.receding = config.receding;
  config.pan.step_time = config.step_time;

  config.pan.point_flow.receding = config.receding;
  config.pan.point_flow.dt = config.step_time;
  config.pan.point_flow.dune_max_num =
      static_cast<std::size_t>(std::max(0, config.pan.dune_max_num));
  config.pan.point_flow.body_half_extent = config.robot.body_half_extent;

  config.preselect.max_points =
      static_cast<std::size_t>(std::max(0, config.pan.dune_max_num));

  configureDune(config);
  configureNrmp(config, spec.dynamics, spec.state_weight_gain);

  return config;
}

}  // namespace neupan_uav
