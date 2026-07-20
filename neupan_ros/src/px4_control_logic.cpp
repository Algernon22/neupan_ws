#include "neupan_ros/px4_control_logic.hpp"

#include <algorithm>

namespace neupan_ros {

namespace {

bool takeoffReleased(const ControlInputs& inputs, double target_height_m) {
  return inputs.has_altitude &&
         inputs.altitude_m >= std::max(0.0, target_height_m);
}

}  // namespace

ControlPhase advanceControlPhase(ControlPhase previous_phase,
                                 const ControlInputs& inputs,
                                 const ControlConfig& config) {
  if (previous_phase == ControlPhase::kInit) {
    if (!inputs.vehicle_status_seen || !inputs.odom_fresh) {
      return ControlPhase::kInit;
    }
    return config.enable_takeoff_phase ? ControlPhase::kTakeoff
                                       : ControlPhase::kCarry;
  }

  if (previous_phase == ControlPhase::kTakeoff) {
    if (inputs.planner_arrived) return ControlPhase::kFinish;
    if (takeoffReleased(inputs, config.takeoff_target_height_m)) {
      return ControlPhase::kCarry;
    }
    return ControlPhase::kTakeoff;
  }

  if (previous_phase == ControlPhase::kCarry) {
    if (inputs.planner_arrived) return ControlPhase::kFinish;
    return ControlPhase::kCarry;
  }

  return ControlPhase::kFinish;
}

ControlDecision evaluateControlDecision(ControlPhase previous_phase,
                                        const ControlInputs& inputs,
                                        const ControlConfig& config) {
  const ControlPhase phase = advanceControlPhase(previous_phase, inputs, config);

  if (!inputs.vehicle_status_seen) {
    return ControlDecision{phase, ControlPolicy::kZeroHold,
                           "waiting_for_vehicle_state"};
  }
  if (!inputs.odom_fresh) {
    return ControlDecision{phase, ControlPolicy::kZeroHold,
                           "waiting_for_pose"};
  }
  if (phase == ControlPhase::kFinish) {
    return ControlDecision{phase, ControlPolicy::kZeroHold,
                           "mission_finished"};
  }
  if (!inputs.offboard_active) {
    return ControlDecision{phase, ControlPolicy::kZeroHold,
                           "waiting_for_offboard"};
  }
  if (phase == ControlPhase::kTakeoff) {
    return ControlDecision{phase, ControlPolicy::kTakeoffClimb,
                           "takeoff_phase"};
  }
  if (inputs.planner_cmd_fresh) {
    return ControlDecision{phase, ControlPolicy::kFollowPlanner,
                           "following_planner"};
  }
  return ControlDecision{phase, ControlPolicy::kZeroHold,
                         "planner_cmd_stale"};
}

const char* toString(ControlPhase phase) {
  switch (phase) {
    case ControlPhase::kInit:
      return "INIT";
    case ControlPhase::kTakeoff:
      return "TAKEOFF";
    case ControlPhase::kCarry:
      return "CARRY";
    case ControlPhase::kFinish:
      return "FINISH";
  }
  return "UNKNOWN";
}

const char* toString(ControlPolicy policy) {
  switch (policy) {
    case ControlPolicy::kTakeoffClimb:
      return "TAKEOFF_CLIMB";
    case ControlPolicy::kFollowPlanner:
      return "FOLLOW_PLANNER";
    case ControlPolicy::kZeroHold:
      return "ZERO_HOLD";
  }
  return "UNKNOWN";
}

}  // namespace neupan_ros
