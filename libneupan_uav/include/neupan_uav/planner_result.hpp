#pragma once

#include "neupan_uav/types.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace neupan_uav {

enum class SafetyStopCause {
  kClearanceViolation,
  kConstraintViolation,
  kGeofenceViolation,
};

enum class PlannerFault {
  kInvalidInput,
  kInvalidState,
  kInvalidObstaclePoints,
  kInvalidObstacleVelocities,
  kStaleInput,
  kSolverUnavailable,
  kSolverFailure,
  kInferenceFailure,
};

struct PlanBundle {
  Trajectory trajectory;
  Trajectory reference;
  Eigen::MatrixXd control_trajectory;
  Eigen::RowVectorXd nominal_distance;
};

struct Tracking {
  Control command = Control::Zero();
  PlanBundle plan;
};

struct GoalReached {};

struct SafetyStop {
  SafetyStopCause cause = SafetyStopCause::kClearanceViolation;
  double observed_clearance = 0.0;
  double required_clearance = 0.0;
};

struct FaultStop {
  PlannerFault fault = PlannerFault::kInvalidInput;
  std::string detail;
};

using PlannerDecision =
    std::variant<Tracking, GoalReached, SafetyStop, FaultStop>;

struct PlannerDiagnostics {
  PlannerProfile profile;
  Control warm_start_seed = Control::Zero();
  std::optional<double> min_clearance;
};

class PlannerResult {
 public:
  static PlannerResult tracking(Control command, PlanBundle plan,
                                PlannerDiagnostics diagnostics) {
    return PlannerResult(
        Tracking{std::move(command), std::move(plan)}, std::move(diagnostics));
  }

  static PlannerResult goalReached(PlannerDiagnostics diagnostics) {
    return PlannerResult(GoalReached{}, std::move(diagnostics));
  }

  static PlannerResult safetyStop(SafetyStopCause cause,
                                  double observed_clearance,
                                  double required_clearance,
                                  PlannerDiagnostics diagnostics) {
    return PlannerResult(
        SafetyStop{cause, observed_clearance, required_clearance},
        std::move(diagnostics));
  }

  static PlannerResult faultStop(PlannerFault fault, std::string detail,
                                 PlannerDiagnostics diagnostics) {
    return PlannerResult(FaultStop{fault, std::move(detail)},
                         std::move(diagnostics));
  }

  const PlannerDecision& decision() const noexcept { return decision_; }
  const PlannerDiagnostics& diagnostics() const noexcept {
    return diagnostics_;
  }

  bool isTracking() const noexcept {
    return std::holds_alternative<Tracking>(decision_);
  }

  Control commandToPublish() const {
    if (const auto* tracking = std::get_if<Tracking>(&decision_)) {
      return tracking->command;
    }
    return Control::Zero();
  }

 private:
  PlannerResult(PlannerDecision decision, PlannerDiagnostics diagnostics)
      : decision_(std::move(decision)), diagnostics_(std::move(diagnostics)) {}

  PlannerDecision decision_;
  PlannerDiagnostics diagnostics_;
};

inline std::string_view toString(SafetyStopCause cause) {
  switch (cause) {
    case SafetyStopCause::kClearanceViolation:
      return "clearance_violation";
    case SafetyStopCause::kConstraintViolation:
      return "constraint_violation";
    case SafetyStopCause::kGeofenceViolation:
      return "geofence_violation";
  }
  return "unknown_safety_stop";
}

inline std::string_view toString(PlannerFault fault) {
  switch (fault) {
    case PlannerFault::kInvalidInput:
      return "invalid_input";
    case PlannerFault::kInvalidState:
      return "invalid_state";
    case PlannerFault::kInvalidObstaclePoints:
      return "invalid_obstacle_points";
    case PlannerFault::kInvalidObstacleVelocities:
      return "invalid_obstacle_velocities";
    case PlannerFault::kStaleInput:
      return "stale_input";
    case PlannerFault::kSolverUnavailable:
      return "solver_unavailable";
    case PlannerFault::kSolverFailure:
      return "solver_failure";
    case PlannerFault::kInferenceFailure:
      return "inference_failure";
  }
  return "unknown_planner_fault";
}

}  // namespace neupan_uav
