#include "neupan_uav/planner.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
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

bool allFiniteState(const UavState& state) {
  return state.position_world.allFinite() && state.velocity_world.allFinite() &&
         std::isfinite(state.attitude_world_body.w()) &&
         std::isfinite(state.attitude_world_body.x()) &&
         std::isfinite(state.attitude_world_body.y()) &&
         std::isfinite(state.attitude_world_body.z()) &&
         std::isfinite(state.yaw_rate);
}

Eigen::VectorXd controlToDim(const Control& control, int control_dim) {
  Eigen::VectorXd out = Eigen::VectorXd::Zero(control_dim);
  const int take = std::min(4, control_dim);
  for (int i = 0; i < take; ++i) out(i) = control(i);
  return out;
}

Eigen::VectorXd rolloutState(const Eigen::VectorXd& state,
                             const Eigen::VectorXd& control,
                             const CompiledPlannerConfig& config) {
  const PanConfig& pan = config.pan();
  if (pan.nrmp.dynamics_A.rows() == state.size() &&
      pan.nrmp.dynamics_A.cols() == state.size() &&
      pan.nrmp.dynamics_B.rows() == state.size() &&
      pan.nrmp.dynamics_B.cols() == control.size() &&
      pan.nrmp.dynamics_C.size() == state.size()) {
    return pan.nrmp.dynamics_A * state + pan.nrmp.dynamics_B * control +
           pan.nrmp.dynamics_C;
  }

  Eigen::VectorXd next = state;
  if (state.size() >= 3 && control.size() >= 3) {
    next.head<3>() += config.stepTime() * control.head<3>();
  }
  if (state.size() >= 4 && control.size() >= 4) {
    next(3) += config.stepTime() * control(3);
  }
  return next;
}

}  // namespace

Planner::Planner(const CompiledPlannerConfig& config)
    : config_(config),
      robot_(config.robot()),
      preselector_(config.preselect()),
      farfield_guide_(config.farfieldGuide()),
      pan_(config.pan()) {
  const int control_dim = config_.pan().nrmp.control_dim;
  control_buffer_ = Eigen::MatrixXd::Zero(control_dim, config_.receding());
  initializePathCache();
}

void Planner::setRknnRunner(std::unique_ptr<RknnRunner> runner) {
  pan_.setRknnRunner(std::move(runner));
}

PlannerResult Planner::forward(const PlannerInput& input) {
  const auto start = std::chrono::steady_clock::now();
  const Control seed = previous_command_;
  PlannerDiagnostics diagnostics;
  diagnostics.warm_start_seed = seed;

  if (!input.valid) {
    clearPreviousCommand();
    return PlannerResult::faultStop(PlannerFault::kInvalidInput,
                                    "planner input was marked invalid",
                                    std::move(diagnostics));
  }
  if (input.stale) {
    clearPreviousCommand();
    return PlannerResult::faultStop(PlannerFault::kStaleInput,
                                    "planner input is stale",
                                    std::move(diagnostics));
  }
  if (!allFiniteState(input.state)) {
    clearPreviousCommand();
    return PlannerResult::faultStop(PlannerFault::kInvalidState,
                                    "state contains invalid values",
                                    std::move(diagnostics));
  }
  if (input.obstacle_points.rows() != 3 ||
      !allFinite(input.obstacle_points)) {
    clearPreviousCommand();
    return PlannerResult::faultStop(
        PlannerFault::kInvalidObstaclePoints,
        "obstacle points must have shape 3xN and finite values",
        std::move(diagnostics));
  }
  if (input.obstacle_velocities.size() != 0 &&
      (input.obstacle_velocities.rows() != 3 ||
       input.obstacle_velocities.cols() != input.obstacle_points.cols() ||
       !allFinite(input.obstacle_velocities))) {
    clearPreviousCommand();
    return PlannerResult::faultStop(
        PlannerFault::kInvalidObstacleVelocities,
        "obstacle velocities must be empty or finite with shape 3xN matching "
        "obstacle points",
        std::move(diagnostics));
  }

  const DynamicsState state = toDynamicsState(input.state, last_yaw_);
  updatePathProgress(state);
  last_yaw_ = state(3);

  diagnostics.profile.input_obstacle_count =
      static_cast<std::size_t>(input.obstacle_points.cols());

  const double raw_clearance = minBodyClearance(state, input.obstacle_points);
  diagnostics.min_clearance = raw_clearance;

  if (raw_clearance < config_.collisionThreshold()) {
    clearPreviousCommand();
    resetControlBuffer();
    diagnostics.profile.forward_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    return PlannerResult::safetyStop(SafetyStopCause::kClearanceViolation,
                                     raw_clearance,
                                     config_.collisionThreshold(),
                                     std::move(diagnostics));
  }

  if (hasArrived(state)) {
    clearPreviousCommand();
    resetControlBuffer();
    diagnostics.profile.forward_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    return PlannerResult::goalReached(std::move(diagnostics));
  }

  ObstacleSelection raw_selection;
  raw_selection.points = input.obstacle_points;
  raw_selection.velocities = input.obstacle_velocities;
  FarfieldGuideProfile farfield_profile;

  try {
    PanInput pan_input = buildPanInput(input, raw_selection, seed,
                                       &farfield_profile);
    diagnostics.profile.farfield_sec = farfield_profile.total_sec;
    diagnostics.profile.farfield_active = farfield_profile.active;
    diagnostics.profile.farfield_near_m = farfield_profile.near_m;
    diagnostics.profile.farfield_far_m = farfield_profile.far_m;
    diagnostics.profile.farfield_offset_m = farfield_profile.offset_m;
    diagnostics.profile.farfield_target_offset_m =
        farfield_profile.target_offset_m;
    diagnostics.profile.farfield_center_count = farfield_profile.center_count;
    diagnostics.profile.farfield_left_count = farfield_profile.left_count;
    diagnostics.profile.farfield_right_count = farfield_profile.right_count;
    diagnostics.profile.farfield_release_streak =
        farfield_profile.release_streak;
    const ObstacleSelection selected =
        preselector_.selectWithNominalTrajectory(
            pan_input.nominal_states, input.obstacle_points,
            input.obstacle_velocities, pan_input.attitude_horizon);
    pan_input.obstacle_points = selected.points;
    pan_input.obstacle_velocities = selected.velocities;
    pan_input.selection_tags = selected.tags;
    diagnostics.profile.input_obstacle_count =
        selected.profile.input_obstacle_count;
    diagnostics.profile.corridor_obstacle_count =
        selected.profile.corridor_obstacle_count;
    diagnostics.profile.preselected_obstacle_count =
        selected.profile.preselected_obstacle_count;
    diagnostics.profile.preselect_max_count =
        selected.profile.preselect_max_count;
    diagnostics.profile.preselect_sec = selected.profile.preselect_sec;
    diagnostics.profile.preselect_corridor_sec =
        selected.profile.preselect_corridor_sec;
    diagnostics.profile.preselect_distance_sec =
        selected.profile.preselect_distance_sec;
    diagnostics.profile.preselect_select_sec =
        selected.profile.preselect_select_sec;
    diagnostics.profile.hard_count = selected.profile.hard_count;
    diagnostics.profile.nearest_quota = selected.profile.nearest_quota;
    diagnostics.profile.nearest_selected = selected.profile.nearest_selected;
    diagnostics.profile.temporal_quota = selected.profile.temporal_quota;
    diagnostics.profile.continuity_hits = selected.profile.continuity_hits;
    diagnostics.profile.continuity_selected =
        selected.profile.continuity_selected;
    diagnostics.profile.diversity_quota = selected.profile.diversity_quota;
    diagnostics.profile.diversity_candidates =
        selected.profile.diversity_candidates;
    diagnostics.profile.diversity_selected =
        selected.profile.diversity_selected;
    diagnostics.profile.fill_selected = selected.profile.fill_selected;

    const double selected_clearance = minBodyClearance(state, selected.points);
    diagnostics.min_clearance = selected_clearance;
    if (selected_clearance < config_.collisionThreshold()) {
      clearPreviousCommand();
      resetControlBuffer();
      diagnostics.profile.forward_sec =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
              .count();
      return PlannerResult::safetyStop(SafetyStopCause::kClearanceViolation,
                                       selected_clearance,
                                       config_.collisionThreshold(),
                                       std::move(diagnostics));
    }

    const PanOutput pan_out = pan_.forward(pan_input);
    Control command = robot_.clampControl(pan_out.command);
    previous_command_ = command;

    PlanBundle plan;
    plan.trajectory = pan_out.trajectory;
    plan.reference = pan_out.reference;
    plan.control_trajectory = pan_out.control_trajectory;
    plan.nominal_distance = pan_out.nominal_distance;
    if (pan_out.control_trajectory.rows() == control_buffer_.rows() &&
        pan_out.control_trajectory.cols() == control_buffer_.cols()) {
      control_buffer_ = pan_out.control_trajectory;
    }
    diagnostics.profile.osqp_status = pan_out.profile.osqp_status;
    diagnostics.profile.osqp_iteration_count =
        pan_out.profile.osqp_iteration_count;
    diagnostics.profile.osqp_solve_sec = pan_out.profile.osqp_solve_sec;
    diagnostics.profile.osqp_run_time_sec = pan_out.profile.osqp_run_time_sec;
    diagnostics.profile.pan_iterations = pan_out.profile.pan_iterations;
    diagnostics.profile.pan_iteration_limit =
        pan_out.profile.pan_iteration_limit;
    diagnostics.profile.dune_sec = pan_out.profile.dune_sec;
    diagnostics.profile.dune_inference_sec =
        pan_out.profile.dune_inference_sec;
    diagnostics.profile.dune_select_sec = pan_out.profile.dune_select_sec;
    diagnostics.profile.nrmp_sec = pan_out.profile.nrmp_sec;
    diagnostics.profile.dune_selected_count =
        pan_out.profile.dune_selected_count;
    if (std::isfinite(pan_out.min_distance)) {
      diagnostics.min_clearance = pan_out.min_distance;
    }
    diagnostics.profile.forward_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    return PlannerResult::tracking(command, std::move(plan),
                                   std::move(diagnostics));
  } catch (const std::exception& error) {
    clearPreviousCommand();
    resetControlBuffer();
    diagnostics.profile.forward_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    return PlannerResult::faultStop(classifyPlanningException(error),
                                    error.what(), std::move(diagnostics));
  }
}

void Planner::reset() {
  clearPreviousCommand();
  resetControlBuffer();
  preselector_.reset();
  farfield_guide_.reset();
  path_progress_s_ = 0.0;
  last_yaw_ = std::numeric_limits<double>::quiet_NaN();
  arrive_latched_ = false;
}

PlannerFault Planner::classifyPlanningException(
    const std::exception& error) const {
  const std::string detail = error.what();
  if (detail.find("RKNN") != std::string::npos ||
      detail.find("rknn") != std::string::npos ||
      detail.find("DUNE") != std::string::npos ||
      detail.find("dune") != std::string::npos ||
      detail.find("inference") != std::string::npos) {
    return PlannerFault::kInferenceFailure;
  }
  if (detail.find("unavailable") != std::string::npos) {
    return PlannerFault::kSolverUnavailable;
  }
  return PlannerFault::kSolverFailure;
}

PanInput Planner::buildPanInput(const PlannerInput& input,
                                const ObstacleSelection& selected,
                                const Control& seed,
                                FarfieldGuideProfile* farfield_profile) {
  const int state_dim = config_.pan().nrmp.state_dim;
  const int control_dim = config_.pan().nrmp.control_dim;
  const int T = config_.receding();
  const DynamicsState dynamics_state = toDynamicsState(input.state, last_yaw_);
  const Eigen::Vector3d attitude_rpy = attitudeRpy(input.state, dynamics_state(3));

  const Control desired = desiredControl(dynamics_state);
  const Eigen::VectorXd desired_vec = controlToDim(desired, control_dim);
  const Eigen::VectorXd seed_vec = controlToDim(seed, control_dim);

  PanInput pan_input;
  pan_input.seed_control = seed;
  pan_input.obstacle_points = selected.points;
  pan_input.obstacle_velocities = selected.velocities;
  pan_input.selection_tags = selected.tags;
  pan_input.nominal_states.resize(state_dim, T + 1);
  pan_input.reference_states.resize(state_dim, T + 1);
  pan_input.nominal_controls.resize(control_dim, T);
  pan_input.reference_controls.resize(control_dim, T);

  Eigen::VectorXd nominal = dynamics_state;
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
        ref_geom, input.obstacle_points, dynamics_state, config_.refSpeed(),
        config_.stepTime(), config_.receding());
    ref_geom = farfield.reference_geometry;
    if (farfield_profile != nullptr) *farfield_profile = farfield.profile;
    std::vector<Control> ref_twist(static_cast<std::size_t>(T), Control::Zero());
    for (int t = 0; t < T; ++t) {
      ref_twist[static_cast<std::size_t>(t)] =
          (ref_geom.col(t + 1) - ref_geom.col(t)) / config_.stepTime();
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
    pan_input.attitude_horizon.col(t) = attitude_rpy;
    pan_input.attitude_horizon(2, t) =
        pan_input.nominal_states.rows() >= 4 ? pan_input.nominal_states(3, t)
                                             : attitude_rpy(2);
  }
  return pan_input;
}

Control Planner::desiredControl(const DynamicsState& state) const {
  if (hasInitialPath()) {
    const Eigen::Vector4d current = samplePath(path_progress_s_);
    const Eigen::Vector4d next = samplePath(
        path_progress_s_ +
        std::max(0.0, config_.refSpeed()) * config_.stepTime());
    Control desired = Control::Zero();
    desired = (next - current) / config_.stepTime();
    return robot_.clampControl(desired);
  }

  Control desired = config_.placeholderCommand();
  if (config_.hasGoal()) {
    const Eigen::Vector3d pos = state.head<3>();
    const Eigen::Vector3d delta = config_.goalPosition() - pos;
    const double distance = delta.norm();
    desired.setZero();
    if (distance > 1.0e-9) {
      desired.head<3>() =
          delta / distance * std::max(0.0, config_.refSpeed());
    }
    desired(3) = config_.placeholderCommand()(3);
  }
  return robot_.clampControl(desired);
}

bool Planner::hasArrived(const DynamicsState& state) {
  if (arrive_latched_) return true;
  Eigen::Vector3d target = Eigen::Vector3d::Zero();
  if (hasInitialPath()) {
    target = path_waypoints_.back().head<3>();
  } else if (config_.hasGoal()) {
    target = config_.goalPosition();
  } else {
    return false;
  }

  const Eigen::Vector3d pos = state.head<3>();
  if ((pos - target).norm() <= config_.arriveThreshold()) {
    arrive_latched_ = true;
    return true;
  }
  return false;
}

double Planner::minBodyClearance(const DynamicsState& state,
                                 const PointMatrix& points) const {
  if (points.cols() == 0) {
    return std::numeric_limits<double>::infinity();
  }

  const Eigen::Vector3d pos = state.head<3>();
  const Eigen::Vector3d half = config_.robot().body_half_extent.cwiseMax(0.0);
  double min_clearance = std::numeric_limits<double>::infinity();
  for (Eigen::Index col = 0; col < points.cols(); ++col) {
    const Eigen::Vector3d local = points.col(col) - pos;
    const Eigen::Vector3d delta = (local.cwiseAbs() - half).cwiseMax(0.0);
    min_clearance = std::min(min_clearance, delta.norm());
  }
  return min_clearance;
}

void Planner::initializePathCache() {
  path_waypoints_ = config_.initialPath().waypoints;
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

void Planner::updatePathProgress(const DynamicsState& state) {
  if (!hasInitialPath()) return;
  const double projected = projectPathProgress(state.head<3>());
  path_progress_s_ =
      config_.initialPath().monotonic_progress
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
  Eigen::Matrix<Scalar, 4, Eigen::Dynamic> ref(4, config_.receding() + 1);
  const double step_len = std::max(0.0, config_.refSpeed()) * config_.stepTime();
  for (int t = 0; t <= config_.receding(); ++t) {
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
