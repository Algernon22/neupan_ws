#include "neupan_uav/planner.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>

namespace neupan_uav {

namespace {

bool allFinite(const Eigen::MatrixXd& matrix) {
  for (Eigen::Index i = 0; i < matrix.size(); ++i) {
    if (!std::isfinite(matrix.data()[i])) return false;
  }
  return true;
}

bool allFiniteVector(const Eigen::VectorXd& vector) {
  for (Eigen::Index i = 0; i < vector.size(); ++i) {
    if (!std::isfinite(vector(i))) return false;
  }
  return true;
}

ObstaclePreselectorConfig makePreselectorConfig(const PlannerConfig& config) {
  ObstaclePreselectorConfig preselect = config.preselect;
  preselect.dt = config.step_time;
  preselect.body_half_extent = config.robot.body_half_extent;
  return preselect;
}

PanConfig makePanConfig(const PlannerConfig& config) {
  const PanConfig& pan = config.pan;
  if (pan.receding != config.receding) {
    throw std::invalid_argument(
        "PanConfig::receding must match PlannerConfig::receding");
  }
  if (pan.step_time != config.step_time) {
    throw std::invalid_argument(
        "PanConfig::step_time must match PlannerConfig::step_time");
  }
  if (pan.point_flow.receding != config.receding) {
    throw std::invalid_argument(
        "PointFlowConfig::receding must match PlannerConfig::receding");
  }
  if (pan.point_flow.dt != config.step_time) {
    throw std::invalid_argument(
        "PointFlowConfig::dt must match PlannerConfig::step_time");
  }
  if (!pan.point_flow.body_half_extent.isApprox(config.robot.body_half_extent)) {
    throw std::invalid_argument(
        "PointFlowConfig::body_half_extent must match RobotModelConfig");
  }
  return pan;
}

Eigen::VectorXd copyStatePrefix(const Eigen::VectorXd& state, int state_dim) {
  Eigen::VectorXd out = Eigen::VectorXd::Zero(state_dim);
  const Eigen::Index take =
      std::min<Eigen::Index>(state.size(), static_cast<Eigen::Index>(state_dim));
  if (take > 0) out.head(take) = state.head(take);
  return out;
}

Eigen::VectorXd controlToDim(const Control& control, int control_dim) {
  Eigen::VectorXd out = Eigen::VectorXd::Zero(control_dim);
  const int take = std::min(4, control_dim);
  for (int i = 0; i < take; ++i) out(i) = control(i);
  return out;
}

Eigen::VectorXd rolloutState(const Eigen::VectorXd& state,
                             const Eigen::VectorXd& control,
                             const PlannerConfig& config) {
  const PanConfig& pan = config.pan;
  if (pan.has_nrmp_config &&
      pan.nrmp.dynamics_A.rows() == state.size() &&
      pan.nrmp.dynamics_A.cols() == state.size() &&
      pan.nrmp.dynamics_B.rows() == state.size() &&
      pan.nrmp.dynamics_B.cols() == control.size() &&
      pan.nrmp.dynamics_C.size() == state.size()) {
    return pan.nrmp.dynamics_A * state + pan.nrmp.dynamics_B * control +
           pan.nrmp.dynamics_C;
  }

  Eigen::VectorXd next = state;
  if (state.size() >= 3 && control.size() >= 3) {
    next.head<3>() += config.step_time * control.head<3>();
  }
  if (state.size() >= 4 && control.size() >= 4) {
    next(3) += config.step_time * control(3);
  }
  return next;
}

}  // namespace

Planner::Planner(const PlannerConfig& config)
    : config_(config),
      robot_(config.robot),
      preselector_(makePreselectorConfig(config)),
      farfield_guide_(config.farfield_guide),
      pan_(makePanConfig(config)) {
  if (config_.receding <= 0) {
    throw std::invalid_argument("PlannerConfig::receding must be positive");
  }
  if (!(config_.step_time > 0.0) || !std::isfinite(config_.step_time)) {
    throw std::invalid_argument("PlannerConfig::step_time must be finite and positive");
  }
  if (config_.collision_threshold < 0.0 ||
      !std::isfinite(config_.collision_threshold)) {
    throw std::invalid_argument(
        "PlannerConfig::collision_threshold must be finite and non-negative");
  }
  const int control_dim =
      config_.pan.has_nrmp_config ? config_.pan.nrmp.control_dim : 4;
  control_buffer_ = Eigen::MatrixXd::Zero(control_dim, config_.receding);
  initializePathCache();
}

void Planner::setRknnRunner(std::unique_ptr<RknnRunner> runner) {
  pan_.setRknnRunner(std::move(runner));
}

PlannerOutput Planner::forward(const PlannerInput& input) {
  const auto start = std::chrono::steady_clock::now();
  const Control seed = previous_command_;

  if (!input.valid) {
    clearPreviousCommand();
    PlannerOutput out = invalidOutput("invalid_input");
    out.seed_control = seed;
    return out;
  }
  if (input.stale) {
    clearPreviousCommand();
    PlannerOutput out = invalidOutput("stale_input");
    out.seed_control = seed;
    return out;
  }
  if (input.state.size() < 4 || !allFiniteVector(input.state)) {
    clearPreviousCommand();
    PlannerOutput out = invalidOutput("invalid_state");
    out.seed_control = seed;
    return out;
  }
  if (input.obstacle_points.rows() != 3 ||
      !allFinite(input.obstacle_points)) {
    clearPreviousCommand();
    PlannerOutput out = invalidOutput("invalid_obstacle_points");
    out.seed_control = seed;
    return out;
  }
  if (input.obstacle_velocities.size() != 0 &&
      (input.obstacle_velocities.rows() != 3 ||
       input.obstacle_velocities.cols() != input.obstacle_points.cols() ||
       !allFinite(input.obstacle_velocities))) {
    clearPreviousCommand();
    PlannerOutput out = invalidOutput("invalid_obstacle_velocities");
    out.seed_control = seed;
    return out;
  }

  updatePathProgress(input.state);

  PlannerOutput out;
  out.seed_control = seed;
  out.profile.input_obstacle_count =
      static_cast<std::size_t>(input.obstacle_points.cols());

  if (hasArrived(input.state)) {
    out.command = Control::Zero();
    out.ready = true;
    out.reason = "arrived";
    out.arrive = true;
    out.min_distance = minBodyClearance(input.state, input.obstacle_points);
    clearPreviousCommand();
    resetControlBuffer();
    out.profile.forward_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    return out;
  }

  ObstacleSelection raw_selection;
  raw_selection.points = input.obstacle_points;
  raw_selection.velocities = input.obstacle_velocities;
  FarfieldGuideProfile farfield_profile;
  PanInput pan_input = buildPanInput(input, raw_selection, seed,
                                     &farfield_profile);
  out.profile.farfield_sec = farfield_profile.total_sec;
  out.profile.farfield_active = farfield_profile.active;
  out.profile.farfield_near_m = farfield_profile.near_m;
  out.profile.farfield_far_m = farfield_profile.far_m;
  out.profile.farfield_offset_m = farfield_profile.offset_m;
  out.profile.farfield_target_offset_m = farfield_profile.target_offset_m;
  out.profile.farfield_center_count = farfield_profile.center_count;
  out.profile.farfield_left_count = farfield_profile.left_count;
  out.profile.farfield_right_count = farfield_profile.right_count;
  out.profile.farfield_release_streak = farfield_profile.release_streak;
  const ObstacleSelection selected =
      preselector_.selectWithNominalTrajectory(
          pan_input.nominal_states, input.obstacle_points,
          input.obstacle_velocities, pan_input.attitude_horizon);
  pan_input.obstacle_points = selected.points;
  pan_input.obstacle_velocities = selected.velocities;
  pan_input.selection_tags = selected.tags;
  out.profile.input_obstacle_count = selected.profile.input_obstacle_count;
  out.profile.corridor_obstacle_count =
      selected.profile.corridor_obstacle_count;
  out.profile.preselected_obstacle_count =
      selected.profile.preselected_obstacle_count;
  out.profile.preselect_max_count = selected.profile.preselect_max_count;
  out.profile.preselect_sec = selected.profile.preselect_sec;
  out.profile.preselect_corridor_sec =
      selected.profile.preselect_corridor_sec;
  out.profile.preselect_distance_sec =
      selected.profile.preselect_distance_sec;
  out.profile.preselect_select_sec = selected.profile.preselect_select_sec;
  out.profile.hard_count = selected.profile.hard_count;
  out.profile.nearest_quota = selected.profile.nearest_quota;
  out.profile.nearest_selected = selected.profile.nearest_selected;
  out.profile.temporal_quota = selected.profile.temporal_quota;
  out.profile.continuity_hits = selected.profile.continuity_hits;
  out.profile.continuity_selected = selected.profile.continuity_selected;
  out.profile.diversity_quota = selected.profile.diversity_quota;
  out.profile.diversity_candidates = selected.profile.diversity_candidates;
  out.profile.diversity_selected = selected.profile.diversity_selected;
  out.profile.fill_selected = selected.profile.fill_selected;
  out.min_distance = minBodyClearance(input.state, selected.points);

  if (out.min_distance < config_.collision_threshold) {
    out.command = Control::Zero();
    out.ready = true;
    out.reason = "planner_stop";
    out.stop = true;
    clearPreviousCommand();
    resetControlBuffer();
    out.profile.forward_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    return out;
  }

  const PanOutput pan_out = pan_.forward(pan_input);
  out.command = robot_.clampControl(pan_out.command);
  previous_command_ = out.command;
  out.trajectory = pan_out.trajectory;
  out.reference = pan_out.reference;
  out.control_trajectory = pan_out.control_trajectory;
  out.nominal_distance = pan_out.nominal_distance;
  if (pan_out.control_trajectory.rows() == control_buffer_.rows() &&
      pan_out.control_trajectory.cols() == control_buffer_.cols()) {
    control_buffer_ = pan_out.control_trajectory;
  }
  out.profile.osqp_status = pan_out.profile.osqp_status;
  out.profile.osqp_iteration_count = pan_out.profile.osqp_iteration_count;
  out.profile.osqp_solve_sec = pan_out.profile.osqp_solve_sec;
  out.profile.osqp_run_time_sec = pan_out.profile.osqp_run_time_sec;
  out.profile.pan_iterations = pan_out.profile.pan_iterations;
  out.profile.pan_iteration_limit = pan_out.profile.pan_iteration_limit;
  out.profile.dune_sec = pan_out.profile.dune_sec;
  out.profile.dune_inference_sec = pan_out.profile.dune_inference_sec;
  out.profile.dune_select_sec = pan_out.profile.dune_select_sec;
  out.profile.nrmp_sec = pan_out.profile.nrmp_sec;
  out.profile.dune_selected_count = pan_out.profile.dune_selected_count;
  if (std::isfinite(pan_out.min_distance)) {
    out.min_distance = pan_out.min_distance;
  }
  out.profile.forward_sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  return out;
}

void Planner::reset() {
  clearPreviousCommand();
  resetControlBuffer();
  preselector_.reset();
  farfield_guide_.reset();
  path_progress_s_ = 0.0;
  arrive_latched_ = false;
}

PlannerOutput Planner::invalidOutput(const std::string& reason) const {
  PlannerOutput out;
  out.ready = false;
  out.reason = reason;
  out.command = Control::Zero();
  out.seed_control = previous_command_;
  return out;
}

PanInput Planner::buildPanInput(const PlannerInput& input,
                                const ObstacleSelection& selected,
                                const Control& seed,
                                FarfieldGuideProfile* farfield_profile) {
  const bool has_nrmp_config = config_.pan.has_nrmp_config;
  const int state_dim =
      has_nrmp_config ? config_.pan.nrmp.state_dim
                      : std::max<int>(4, static_cast<int>(input.state.size()));
  const int control_dim =
      has_nrmp_config ? config_.pan.nrmp.control_dim : 4;
  const int T = config_.receding;

  const Control desired = desiredControl(input.state);
  const Eigen::VectorXd desired_vec = controlToDim(desired, control_dim);
  const Eigen::VectorXd seed_vec = controlToDim(seed, control_dim);

  PanInput pan_input;
  pan_input.seed_control = seed;
  pan_input.desired_control = desired;
  pan_input.obstacle_points = selected.points;
  pan_input.obstacle_velocities = selected.velocities;
  pan_input.selection_tags = selected.tags;
  pan_input.nominal_states.resize(state_dim, T + 1);
  pan_input.reference_states.resize(state_dim, T + 1);
  pan_input.nominal_controls.resize(control_dim, T);
  pan_input.reference_controls.resize(control_dim, T);

  Eigen::VectorXd nominal = copyStatePrefix(input.state, state_dim);
  if (input.state.size() == 6 && state_dim >= 4) {
    nominal(3) = input.state(5);
  }
  Eigen::VectorXd reference = nominal;
  pan_input.nominal_states.col(0) = nominal;
  for (int t = 0; t < T; ++t) {
    Eigen::VectorXd nominal_control =
        control_buffer_.rows() == control_dim && control_buffer_.cols() == T
            ? control_buffer_.col(t)
            : desired_vec;
    if (t == 0) nominal_control = seed_vec;
    pan_input.nominal_controls.col(t) = nominal_control;
    nominal = rolloutState(nominal, nominal_control, config_);
    pan_input.nominal_states.col(t + 1) = nominal;
  }

  if (hasInitialPath()) {
    Eigen::Matrix<Scalar, 4, Eigen::Dynamic> ref_geom = referenceGeometry();
    const FarfieldGuideResult farfield = farfield_guide_.apply(
        ref_geom, input.obstacle_points, input.state, config_.ref_speed,
        config_.step_time, config_.receding);
    ref_geom = farfield.reference_geometry;
    if (farfield_profile != nullptr) *farfield_profile = farfield.profile;
    std::vector<Control> ref_twist(static_cast<std::size_t>(T), Control::Zero());
    for (int t = 0; t < T; ++t) {
      ref_twist[static_cast<std::size_t>(t)] =
          (ref_geom.col(t + 1) - ref_geom.col(t)) / config_.step_time;
    }
    for (int t = 0; t <= T; ++t) {
      Eigen::VectorXd ref_state = Eigen::VectorXd::Zero(state_dim);
      ref_state.head<3>() = ref_geom.col(t).head<3>();
      if (state_dim >= 4) ref_state(3) = ref_geom(3, t);
      if (state_dim >= 8) {
        const Control& twist =
            ref_twist[static_cast<std::size_t>(std::min(t, T - 1))];
        ref_state.segment<4>(4) = twist;
      }
      pan_input.reference_states.col(t) = ref_state;
      if (t < T) {
        pan_input.reference_controls.col(t) =
            controlToDim(robot_.clampControl(ref_twist[static_cast<std::size_t>(t)]),
                         control_dim);
      }
    }
  } else {
    if (farfield_profile != nullptr) *farfield_profile = FarfieldGuideProfile();
    pan_input.reference_states.col(0) = reference;
    for (int t = 0; t < T; ++t) {
      pan_input.reference_controls.col(t) = desired_vec;
      reference = rolloutState(reference, desired_vec, config_);
      pan_input.reference_states.col(t + 1) = reference;
    }
  }

  pan_input.attitude_horizon.resize(3, T + 1);
  for (int t = 0; t <= T; ++t) {
    if (input.state.size() >= 6) {
      pan_input.attitude_horizon(0, t) = input.state(3);
      pan_input.attitude_horizon(1, t) = input.state(4);
      pan_input.attitude_horizon(2, t) = input.state(5);
    } else {
      pan_input.attitude_horizon(0, t) = 0.0;
      pan_input.attitude_horizon(1, t) = 0.0;
      pan_input.attitude_horizon(2, t) =
          pan_input.nominal_states.rows() >= 4
              ? pan_input.nominal_states(3, t)
              : 0.0;
    }
  }
  return pan_input;
}

Control Planner::desiredControl(const Eigen::VectorXd& state) const {
  if (hasInitialPath()) {
    const Eigen::Vector4d current = samplePath(path_progress_s_);
    const Eigen::Vector4d next = samplePath(
        path_progress_s_ +
        std::max(0.0, config_.ref_speed) * config_.step_time);
    Control desired = Control::Zero();
    desired = (next - current) / config_.step_time;
    return robot_.clampControl(desired);
  }

  Control desired = config_.placeholder_command;
  if (config_.has_goal && state.size() >= 3) {
    const Eigen::Vector3d pos = state.head<3>();
    const Eigen::Vector3d delta = config_.goal_position - pos;
    const double distance = delta.norm();
    desired.setZero();
    if (distance > 1.0e-9) {
      desired.head<3>() =
          delta / distance * std::max(0.0, config_.ref_speed);
    }
    if (state.size() >= 4) {
      desired(3) = config_.placeholder_command(3);
    }
  }
  return robot_.clampControl(desired);
}

bool Planner::hasArrived(const Eigen::VectorXd& state) {
  if (arrive_latched_) return true;
  Eigen::Vector3d target = Eigen::Vector3d::Zero();
  if (hasInitialPath()) {
    target = path_waypoints_.back().head<3>();
  } else if (config_.has_goal) {
    target = config_.goal_position;
  } else {
    return false;
  }

  const Eigen::Vector3d pos = state.head<3>();
  if ((pos - target).norm() <= config_.arrive_threshold) {
    arrive_latched_ = true;
    return true;
  }
  return false;
}

double Planner::minBodyClearance(const Eigen::VectorXd& state,
                                 const PointMatrix& points) const {
  if (points.cols() == 0) {
    return std::numeric_limits<double>::infinity();
  }

  const Eigen::Vector3d pos = state.head<3>();
  const Eigen::Vector3d half = config_.robot.body_half_extent.cwiseMax(0.0);
  double min_clearance = std::numeric_limits<double>::infinity();
  for (Eigen::Index col = 0; col < points.cols(); ++col) {
    const Eigen::Vector3d local = points.col(col) - pos;
    const Eigen::Vector3d delta = (local.cwiseAbs() - half).cwiseMax(0.0);
    min_clearance = std::min(min_clearance, delta.norm());
  }
  return min_clearance;
}

void Planner::initializePathCache() {
  path_waypoints_ = config_.initial_path.waypoints;
  path_s_.clear();
  path_progress_s_ = 0.0;
  arrive_latched_ = false;
  if (path_waypoints_.empty()) return;

  path_s_.reserve(path_waypoints_.size());
  path_s_.push_back(0.0);
  for (std::size_t i = 0; i < path_waypoints_.size(); ++i) {
    if (!path_waypoints_[i].allFinite()) {
      throw std::invalid_argument(
          "InitialPathConfig::waypoints must contain finite [x, y, z, yaw]");
    }
    if (i == 0) continue;
    const double ds =
        (path_waypoints_[i].head<3>() - path_waypoints_[i - 1].head<3>())
            .norm();
    path_s_.push_back(path_s_.back() + ds);
  }
}

bool Planner::hasInitialPath() const {
  return !path_waypoints_.empty();
}

void Planner::updatePathProgress(const Eigen::VectorXd& state) {
  if (!hasInitialPath()) return;
  const double projected = projectPathProgress(state.head<3>());
  path_progress_s_ =
      config_.initial_path.monotonic_progress
          ? std::max(path_progress_s_, projected)
          : projected;
  if (!path_s_.empty()) {
    path_progress_s_ =
        std::clamp(path_progress_s_, 0.0, path_s_.back());
  }
}

double Planner::projectPathProgress(const Eigen::Vector3d& position) const {
  if (path_waypoints_.size() <= 1 || path_s_.empty()) return 0.0;

  double best_s = 0.0;
  double best_dist_sq = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i + 1 < path_waypoints_.size(); ++i) {
    const Eigen::Vector3d start = path_waypoints_[i].head<3>();
    const Eigen::Vector3d segment =
        path_waypoints_[i + 1].head<3>() - start;
    const double len_sq = segment.squaredNorm();
    if (len_sq <= 1.0e-18) continue;
    const double ratio = std::clamp(
        (position - start).dot(segment) / len_sq, 0.0, 1.0);
    const Eigen::Vector3d projected = start + ratio * segment;
    const double dist_sq = (position - projected).squaredNorm();
    if (dist_sq < best_dist_sq) {
      best_dist_sq = dist_sq;
      best_s = path_s_[i] +
               ratio * (path_s_[i + 1] - path_s_[i]);
    }
  }
  return best_s;
}

Eigen::Vector4d Planner::samplePath(double progress_s) const {
  if (path_waypoints_.empty()) return Eigen::Vector4d::Zero();
  if (path_waypoints_.size() == 1 || path_s_.empty() ||
      path_s_.back() <= 1.0e-18) {
    return path_waypoints_.back();
  }

  const double s = std::clamp(progress_s, 0.0, path_s_.back());
  for (std::size_t i = 0; i + 1 < path_waypoints_.size(); ++i) {
    if (s > path_s_[i + 1] && i + 2 < path_waypoints_.size()) continue;
    const double seg_len = path_s_[i + 1] - path_s_[i];
    if (seg_len <= 1.0e-18) continue;
    const double ratio = std::clamp((s - path_s_[i]) / seg_len, 0.0, 1.0);
    return path_waypoints_[i] +
           ratio * (path_waypoints_[i + 1] - path_waypoints_[i]);
  }
  return path_waypoints_.back();
}

Eigen::Matrix<Scalar, 4, Eigen::Dynamic> Planner::referenceGeometry() const {
  Eigen::Matrix<Scalar, 4, Eigen::Dynamic> ref(4, config_.receding + 1);
  const double step_len = std::max(0.0, config_.ref_speed) * config_.step_time;
  for (int t = 0; t <= config_.receding; ++t) {
    ref.col(t) = samplePath(path_progress_s_ + step_len * t);
  }
  return ref;
}

void Planner::clearPreviousCommand() {
  previous_command_.setZero();
}

void Planner::resetControlBuffer() {
  control_buffer_.setZero();
}

}  // namespace neupan_uav
