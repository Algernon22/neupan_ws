#include "neupan_ros/config_loader.hpp"

#include "neupan_uav/rknn_runner.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

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

void loadRobot(const YAML::Node& root, neupan_uav::PlannerConfig& config,
               neupan_uav::UavDynamicsConfig& dynamics) {
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
  dynamics.max_acceleration =
      control4(robot, "max_acce", dynamics.max_acceleration);
  dynamics.velocity_time_constant =
      vec3(robot, "tau_velocity", dynamics.velocity_time_constant);
  dynamics.velocity_gain = vec3(robot, "gain_velocity", dynamics.velocity_gain);
  dynamics.yaw_rate_time_constant =
      scalar<double>(robot, "tau_yaw_rate", dynamics.yaw_rate_time_constant);
  dynamics.yaw_rate_gain =
      scalar<double>(robot, "gain_yaw_rate", dynamics.yaw_rate_gain);
  dynamics.velocity_weight_scale = scalar<double>(
      robot, "velocity_weight_scale", dynamics.velocity_weight_scale);
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
  config.pan.dune.select_nearest_ratio = scalar<double>(
      pan, "dune_select_nearest_ratio", config.pan.dune.select_nearest_ratio);
  config.pan.dune.select_temporal_ratio = scalar<double>(
      pan, "dune_select_temporal_ratio", config.pan.dune.select_temporal_ratio);
  config.pan.dune.select_diversity_ratio = scalar<double>(
      pan, "dune_select_diversity_ratio", config.pan.dune.select_diversity_ratio);
}

void loadAdjust(const YAML::Node& root, neupan_uav::PlannerConfig& config,
                double& state_weight_gain) {
  const YAML::Node adjust = root["adjust"];
  if (!adjust) return;
  state_weight_gain = scalar<double>(adjust, "q_s", state_weight_gain);
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

}  // namespace

LoadedPlannerConfig loadPlannerConfig(const std::string& yaml_path) {
  YAML::Node root = YAML::LoadFile(yaml_path);
  if (!root || !root.IsMap()) {
    throw std::runtime_error("planner config root must be a YAML mapping");
  }

  neupan_uav::UavPlannerConfigSpec spec;
  auto& config = spec.planner;
  config.collision_threshold =
      scalar<double>(root, "collision_threshold", config.collision_threshold);
  config.receding = scalar<int>(root, "receding", config.receding);
  config.step_time = scalar<double>(root, "step_time", config.step_time);
  config.ref_speed = scalar<double>(root, "ref_speed", config.ref_speed);
  config.arrive_threshold =
      scalar<double>(root, "arrive_threshold", config.arrive_threshold);

  loadRobot(root, config, spec.dynamics);
  loadInitialPath(root, config);
  loadPreselect(root, config);
  loadFarfieldGuide(root, config);
  loadPan(root, config);
  loadAdjust(root, config, spec.state_weight_gain);

  LoadedPlannerConfig loaded;
  loaded.planner = neupan_uav::buildUavPlannerConfig(std::move(spec));
  return loaded;
}

}  // namespace neupan_ros
