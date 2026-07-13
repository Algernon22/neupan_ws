#include "neupan_ros/config_loader.hpp"

#include "neupan_uav/rknn_runner.hpp"

#include <yaml-cpp/yaml.h>

#include <unsupported/Eigen/MatrixFunctions>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace neupan_ros {

namespace {

template <typename T>
T scalar(const YAML::Node& node, const std::string& key, T fallback) {
  const YAML::Node value = node[key];
  if (!value) return fallback;
  return value.as<T>();
}

Eigen::Vector3d vec3(const YAML::Node& node, const std::string& key,
                     const Eigen::Vector3d& fallback) {
  const YAML::Node value = node[key];
  if (!value) return fallback;
  if (!value.IsSequence() || value.size() != 3) {
    throw std::runtime_error(key + " must be a length-3 sequence");
  }
  return Eigen::Vector3d(value[0].as<double>(), value[1].as<double>(),
                         value[2].as<double>());
}

neupan_uav::Control control4(const YAML::Node& node, const std::string& key,
                             const neupan_uav::Control& fallback) {
  const YAML::Node value = node[key];
  if (!value) return fallback;
  if (!value.IsSequence() || value.size() != 4) {
    throw std::runtime_error(key + " must be a length-4 sequence");
  }
  neupan_uav::Control out;
  out << value[0].as<double>(), value[1].as<double>(), value[2].as<double>(),
      value[3].as<double>();
  return out;
}

struct RobotRuntimeConfig {
  neupan_uav::Control max_acce =
      neupan_uav::Control::Constant(std::numeric_limits<double>::infinity());
  Eigen::Vector3d tau_velocity = Eigen::Vector3d(0.35, 0.35, 0.45);
  Eigen::Vector3d gain_velocity = Eigen::Vector3d::Ones();
  double tau_yaw_rate = 0.30;
  double gain_yaw_rate = 1.0;
  double velocity_weight_scale = 0.35;
};

Eigen::MatrixXd continuousA(const RobotRuntimeConfig& robot) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(8, 8);
  A(0, 4) = 1.0;
  A(1, 5) = 1.0;
  A(2, 6) = 1.0;
  A(3, 7) = 1.0;
  for (int i = 0; i < 3; ++i) {
    A(4 + i, 4 + i) = -1.0 / robot.tau_velocity(i);
  }
  A(7, 7) = -1.0 / robot.tau_yaw_rate;
  return A;
}

Eigen::MatrixXd continuousB(const RobotRuntimeConfig& robot) {
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(8, 4);
  for (int i = 0; i < 3; ++i) {
    B(4 + i, i) = robot.gain_velocity(i) / robot.tau_velocity(i);
  }
  B(7, 3) = robot.gain_yaw_rate / robot.tau_yaw_rate;
  return B;
}

void discretizeDynamics(const RobotRuntimeConfig& robot, double dt,
                        Eigen::MatrixXd& A, Eigen::MatrixXd& B,
                        Eigen::VectorXd& C) {
  const Eigen::MatrixXd Ac = continuousA(robot);
  const Eigen::MatrixXd Bc = continuousB(robot);
  Eigen::MatrixXd aug = Eigen::MatrixXd::Zero(12, 12);
  aug.block(0, 0, 8, 8) = Ac;
  aug.block(0, 8, 8, 4) = Bc;
  const Eigen::MatrixXd expm = (aug * dt).exp();
  A = expm.block(0, 0, 8, 8);
  B = expm.block(0, 8, 8, 4);
  C = Eigen::VectorXd::Zero(8);
}

void loadRobot(const YAML::Node& root, neupan_uav::PlannerConfig& config) {
  const YAML::Node robot = root["robot"];
  if (!robot) return;
  const double length = scalar<double>(robot, "length", 2.0 * config.robot.body_half_extent(0));
  const double width = scalar<double>(robot, "width", 2.0 * config.robot.body_half_extent(1));
  const double height = scalar<double>(robot, "height", 2.0 * config.robot.body_half_extent(2));
  config.robot.body_half_extent =
      Eigen::Vector3d(std::max(0.0, length * 0.5), std::max(0.0, width * 0.5),
                      std::max(0.0, height * 0.5));
  config.robot.max_control =
      control4(robot, "max_speed", config.robot.max_control);
}

RobotRuntimeConfig loadRobotRuntime(const YAML::Node& root) {
  RobotRuntimeConfig runtime;
  const YAML::Node robot = root["robot"];
  if (!robot) return runtime;
  runtime.max_acce = control4(robot, "max_acce", runtime.max_acce);
  runtime.tau_velocity =
      vec3(robot, "tau_velocity", runtime.tau_velocity);
  runtime.gain_velocity =
      vec3(robot, "gain_velocity", runtime.gain_velocity);
  runtime.tau_yaw_rate =
      scalar<double>(robot, "tau_yaw_rate", runtime.tau_yaw_rate);
  runtime.gain_yaw_rate =
      scalar<double>(robot, "gain_yaw_rate", runtime.gain_yaw_rate);
  runtime.velocity_weight_scale = scalar<double>(
      robot, "velocity_weight_scale", runtime.velocity_weight_scale);
  if ((runtime.tau_velocity.array() <= 0.0).any() ||
      runtime.tau_yaw_rate <= 0.0) {
    throw std::runtime_error("robot tau_velocity and tau_yaw_rate must be positive");
  }
  return runtime;
}

void loadInitialPath(const YAML::Node& root, neupan_uav::PlannerConfig& config) {
  const YAML::Node ipath = root["ipath"];
  if (!ipath) return;
  config.arrive_threshold =
      scalar<double>(ipath, "arrive_threshold", config.arrive_threshold);
  const YAML::Node waypoints = ipath["waypoints"];
  if (!waypoints) return;
  if (!waypoints.IsSequence()) {
    throw std::runtime_error("ipath.waypoints must be a sequence");
  }
  config.initial_path.waypoints.clear();
  for (const auto& point : waypoints) {
    if (!point.IsSequence() || point.size() != 4) {
      throw std::runtime_error("each ipath waypoint must be [x, y, z, yaw]");
    }
    config.initial_path.waypoints.emplace_back(
        point[0].as<double>(), point[1].as<double>(), point[2].as<double>(),
        point[3].as<double>());
  }
  if (!config.initial_path.waypoints.empty()) {
    config.has_goal = true;
    config.goal_position = config.initial_path.waypoints.back().head<3>();
  }
}

void loadPreselect(const YAML::Node& root, neupan_uav::PlannerConfig& config) {
  const YAML::Node pre = root["preselect"];
  if (!pre) return;
  config.preselect.enabled = scalar<bool>(pre, "enable", config.preselect.enabled);
  config.preselect.corridor_margin =
      vec3(pre, "corridor_margin", config.preselect.corridor_margin);
  config.preselect.exact_margin =
      vec3(pre, "exact_margin", config.preselect.exact_margin);
  config.preselect.hard_distance =
      scalar<double>(pre, "hard_distance", config.preselect.hard_distance);
  config.preselect.per_step =
      scalar<int>(pre, "per_step", config.preselect.per_step);
  config.preselect.nearest_ratio =
      scalar<double>(pre, "nearest_quota_ratio", config.preselect.nearest_ratio);
  config.preselect.diversity_enabled =
      scalar<bool>(pre, "diversity_enable", config.preselect.diversity_enabled);
  config.preselect.diversity_voxel =
      vec3(pre, "diversity_voxel_size", config.preselect.diversity_voxel);
  config.preselect.diversity_ratio =
      scalar<double>(pre, "diversity_quota_ratio", config.preselect.diversity_ratio);
  config.preselect.temporal_enabled =
      scalar<bool>(pre, "temporal_enable", config.preselect.temporal_enabled);
  config.preselect.temporal_radius =
      scalar<double>(pre, "temporal_radius", config.preselect.temporal_radius);
  config.preselect.temporal_ratio =
      scalar<double>(pre, "temporal_quota_ratio", config.preselect.temporal_ratio);
}

void loadFarfieldGuide(const YAML::Node& root,
                       neupan_uav::PlannerConfig& config) {
  const YAML::Node far = root["farfield_guide"];
  if (!far) return;
  if (!far.IsMap()) {
    throw std::runtime_error("farfield_guide must be a mapping");
  }
  auto& out = config.farfield_guide;
  out.enabled = scalar<bool>(far, "enable", out.enabled);
  out.range_backoff =
      scalar<double>(far, "range_backoff", out.range_backoff);
  out.range_scale = scalar<double>(far, "range_scale", out.range_scale);
  out.range_far_limit =
      scalar<double>(far, "range_far_limit", out.range_far_limit);
  out.lateral_width =
      scalar<double>(far, "lateral_width", out.lateral_width);
  out.center_width =
      scalar<double>(far, "center_width", out.center_width);
  out.height_window =
      scalar<double>(far, "height_window", out.height_window);
  out.voxel_size = vec3(far, "voxel_size", out.voxel_size);
  out.trigger_count =
      scalar<int>(far, "trigger_count", out.trigger_count);
  out.offset_min = scalar<double>(far, "offset_min", out.offset_min);
  out.offset_max = scalar<double>(far, "offset_max", out.offset_max);
  out.offset_speed_gain =
      scalar<double>(far, "offset_speed_gain", out.offset_speed_gain);
  out.offset_alpha =
      scalar<double>(far, "offset_alpha", out.offset_alpha);
  out.release_alpha =
      scalar<double>(far, "release_alpha", out.release_alpha);
  out.release_count =
      scalar<int>(far, "release_count", out.release_count);
  out.release_confirm_cycles =
      scalar<int>(far, "release_confirm_cycles",
                  out.release_confirm_cycles);
}

void loadPan(const YAML::Node& root, neupan_uav::PlannerConfig& config) {
  const YAML::Node pan = root["pan"];
  if (!pan) return;
  config.pan.iter_num = scalar<int>(pan, "iter_num", config.pan.iter_num);
  config.pan.iter_threshold =
      scalar<double>(pan, "iter_threshold", config.pan.iter_threshold);
  config.pan.dune_max_num =
      scalar<int>(pan, "dune_max_num", config.pan.dune_max_num);
  config.pan.nrmp_max_num =
      scalar<int>(pan, "nrmp_max_num", config.pan.nrmp_max_num);
  config.pan.dune.select_num =
      static_cast<std::size_t>(std::max(0, config.pan.nrmp_max_num));
  config.pan.dune.dune_max_num =
      static_cast<std::size_t>(std::max(0, config.pan.dune_max_num));
  config.pan.dune.select_nearest_ratio = scalar<double>(
      pan, "dune_select_nearest_ratio", config.pan.dune.select_nearest_ratio);
  config.pan.dune.select_temporal_ratio = scalar<double>(
      pan, "dune_select_temporal_ratio", config.pan.dune.select_temporal_ratio);
  config.pan.dune.select_diversity_ratio = scalar<double>(
      pan, "dune_select_diversity_ratio", config.pan.dune.select_diversity_ratio);
}

void loadAdjust(const YAML::Node& root, neupan_uav::PlannerConfig& config) {
  const YAML::Node adjust = root["adjust"];
  if (!adjust) return;
  config.pan.p_u = scalar<double>(adjust, "p_u", config.pan.p_u);
  config.pan.eta = scalar<double>(adjust, "eta", config.pan.eta);
  config.pan.d_min = scalar<double>(adjust, "d_min", config.pan.d_min);
  config.pan.d_max = scalar<double>(adjust, "d_max", config.pan.d_max);
  config.pan.ro_obs = scalar<double>(adjust, "ro_obs", config.pan.ro_obs);
  config.pan.bk = scalar<double>(adjust, "bk", config.pan.bk);
  config.pan.smooth_du =
      scalar<double>(adjust, "smooth_du", config.pan.smooth_du);
  config.pan.smooth_u0 =
      scalar<double>(adjust, "smooth_u0", config.pan.smooth_u0);

  config.pan.nrmp.enable_control_smoothing = scalar<bool>(
      adjust, "enable_control_smoothing",
      config.pan.nrmp.enable_control_smoothing);

  const YAML::Node solver_args = adjust["solver_args"];
  if (solver_args) {
    auto& opts = config.pan.nrmp.solver_options;
    opts.eps_abs = scalar<double>(solver_args, "eps_abs", opts.eps_abs);
    opts.eps_rel = scalar<double>(solver_args, "eps_rel", opts.eps_rel);
    opts.max_iter = scalar<int>(solver_args, "max_iter", opts.max_iter);
    opts.warm_starting =
        scalar<bool>(solver_args, "warm_starting", opts.warm_starting);
    opts.polishing = scalar<bool>(solver_args, "polishing", opts.polishing);
    opts.verbose = scalar<bool>(solver_args, "verbose", opts.verbose);
  }
}

void finalizeDerivedConfig(neupan_uav::PlannerConfig& config,
                           const RobotRuntimeConfig& runtime,
                           const YAML::Node& root) {
  config.pan.receding = config.receding;
  config.pan.step_time = config.step_time;
  config.pan.point_flow.receding = config.receding;
  config.pan.point_flow.dt = config.step_time;
  config.pan.point_flow.dune_max_num =
      static_cast<std::size_t>(std::max(0, config.pan.dune_max_num));
  config.pan.point_flow.body_half_extent = config.robot.body_half_extent;
  config.preselect.max_points =
      static_cast<std::size_t>(std::max(0, config.pan.dune_max_num));

  config.pan.dune.receding = config.receding;
  config.pan.dune.point_dim = 3;
  config.pan.dune.dune_max_num =
      static_cast<std::size_t>(std::max(0, config.pan.dune_max_num));
  config.pan.dune.select_num =
      static_cast<std::size_t>(std::max(0, config.pan.nrmp_max_num));
  config.pan.dune.edge_dim = 6;
  config.pan.dune.G.resize(6, 3);
  config.pan.dune.G << 1.0F, 0.0F, 0.0F,
      -1.0F, 0.0F, 0.0F,
      0.0F, 1.0F, 0.0F,
      0.0F, -1.0F, 0.0F,
      0.0F, 0.0F, 1.0F,
      0.0F, 0.0F, -1.0F;
  config.pan.dune.h.resize(6);
  config.pan.dune.h << static_cast<float>(config.robot.body_half_extent(0)),
      static_cast<float>(config.robot.body_half_extent(0)),
      static_cast<float>(config.robot.body_half_extent(1)),
      static_cast<float>(config.robot.body_half_extent(1)),
      static_cast<float>(config.robot.body_half_extent(2)),
      static_cast<float>(config.robot.body_half_extent(2));
  config.pan.has_dune_config = config.pan.dune_max_num > 0;

  config.pan.nrmp.receding = config.receding;
  config.pan.nrmp.state_dim = 8;
  config.pan.nrmp.control_dim = 4;
  config.pan.nrmp.geom_dim = 3;
  config.pan.nrmp.point_dim = 3;
  config.pan.nrmp.max_num = config.pan.nrmp_max_num;
  config.pan.nrmp.no_obs = config.pan.nrmp_max_num <= 0;
  discretizeDynamics(runtime, config.step_time, config.pan.nrmp.dynamics_A,
                     config.pan.nrmp.dynamics_B, config.pan.nrmp.dynamics_C);
  config.pan.nrmp.speed_bound = config.robot.max_control;
  config.pan.nrmp.acce_bound = runtime.max_acce * config.step_time;
  config.pan.nrmp.tracking_speed_bound = config.robot.max_control.cwiseAbs();
  config.pan.has_nrmp_config = config.pan.nrmp_max_num > 0;

  const YAML::Node adjust = root["adjust"];
  const double q_s = adjust ? scalar<double>(adjust, "q_s", 1.0) : 1.0;
  config.pan.state_weights = Eigen::VectorXd::Ones(8);
  config.pan.state_weights.tail<4>().setConstant(
      q_s * runtime.velocity_weight_scale);
}

}  // namespace

LoadedPlannerConfig loadPlannerConfig(const std::string& yaml_path) {
  YAML::Node root = YAML::LoadFile(yaml_path);
  if (!root || !root.IsMap()) {
    throw std::runtime_error("planner config root must be a YAML mapping");
  }

  LoadedPlannerConfig loaded;
  auto& config = loaded.planner;
  config.collision_threshold =
      scalar<double>(root, "collision_threshold", config.collision_threshold);
  config.receding = scalar<int>(root, "receding", config.receding);
  config.step_time = scalar<double>(root, "step_time", config.step_time);
  config.ref_speed = scalar<double>(root, "ref_speed", config.ref_speed);
  config.arrive_threshold =
      scalar<double>(root, "arrive_threshold", config.arrive_threshold);
  loaded.farfield_guide_present = static_cast<bool>(root["farfield_guide"]);

  loadRobot(root, config);
  const RobotRuntimeConfig runtime = loadRobotRuntime(root);
  loadInitialPath(root, config);
  loadPreselect(root, config);
  loadFarfieldGuide(root, config);
  loadPan(root, config);
  loadAdjust(root, config);
  finalizeDerivedConfig(config, runtime, root);
  return loaded;
}

}  // namespace neupan_ros
