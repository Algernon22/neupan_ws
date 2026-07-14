#pragma once

#include <string>

namespace neupan_ros {

enum class ControlPhase {
  kInit,
  kTakeoff,
  kCarry,
  kFinish,
};

enum class ControlPolicy {
  kTakeoffClimb,
  kFollowPlanner,
  kZeroHold,
};

struct ControlInputs {
  bool vehicle_status_seen = false;
  bool odom_fresh = false;
  bool offboard_active = false;
  bool has_altitude = false;
  double altitude_m = 0.0;
  bool planner_cmd_fresh = false;
  bool planner_arrived = false;
};

struct ControlConfig {
  bool enable_takeoff_phase = true;
  double takeoff_release_height_m = 1.9;
};

struct ControlDecision {
  ControlPhase phase = ControlPhase::kInit;
  ControlPolicy policy = ControlPolicy::kZeroHold;
  std::string reason = "boot";

  bool operator==(const ControlDecision& other) const {
    return phase == other.phase && policy == other.policy &&
           reason == other.reason;
  }
  bool operator!=(const ControlDecision& other) const {
    return !(*this == other);
  }
};

ControlPhase advanceControlPhase(ControlPhase previous_phase,
                                 const ControlInputs& inputs,
                                 const ControlConfig& config);
ControlDecision evaluateControlDecision(ControlPhase previous_phase,
                                        const ControlInputs& inputs,
                                        const ControlConfig& config);
const char* toString(ControlPhase phase);
const char* toString(ControlPolicy policy);

}  // namespace neupan_ros
