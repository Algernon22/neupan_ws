#include "neupan_ros/px4_control_logic.hpp"

#include <gtest/gtest.h>

namespace {

neupan_ros::ControlInputs makeInputs() {
  neupan_ros::ControlInputs inputs;
  inputs.vehicle_status_seen = true;
  inputs.odom_fresh = true;
  inputs.offboard_active = true;
  inputs.has_altitude = true;
  inputs.altitude_m = 2.5;
  inputs.planner_cmd_fresh = true;
  inputs.planner_cmd_gap_sec = 0.0;
  inputs.planner_arrived = false;
  return inputs;
}

neupan_ros::ControlConfig makeConfig() {
  neupan_ros::ControlConfig config;
  config.enable_takeoff_phase = true;
  config.takeoff_release_height_m = 1.9;
  return config;
}

}  // namespace

TEST(Px4ControlLogic, TakeoffPhaseHoldsClimbPolicyBelowReleaseHeight) {
  auto inputs = makeInputs();
  inputs.altitude_m = 1.2;
  const auto decision = neupan_ros::evaluateControlDecision(
      neupan_ros::ControlPhase::kTakeoff, inputs, makeConfig());

  EXPECT_EQ(decision.phase, neupan_ros::ControlPhase::kTakeoff);
  EXPECT_EQ(decision.policy, neupan_ros::ControlPolicy::kTakeoffClimb);
  EXPECT_EQ(decision.reason, "takeoff_phase");
}

TEST(Px4ControlLogic, CarryFollowsPlannerWhenFresh) {
  const auto decision = neupan_ros::evaluateControlDecision(
      neupan_ros::ControlPhase::kCarry, makeInputs(), makeConfig());

  EXPECT_EQ(decision.phase, neupan_ros::ControlPhase::kCarry);
  EXPECT_EQ(decision.policy, neupan_ros::ControlPolicy::kFollowPlanner);
  EXPECT_EQ(decision.reason, "following_planner");
}

TEST(Px4ControlLogic, WaitsForVehicleStateUntilFirstStateSeen) {
  auto inputs = makeInputs();
  inputs.vehicle_status_seen = false;
  const auto decision = neupan_ros::evaluateControlDecision(
      neupan_ros::ControlPhase::kInit, inputs, makeConfig());

  EXPECT_EQ(decision.phase, neupan_ros::ControlPhase::kInit);
  EXPECT_EQ(decision.policy, neupan_ros::ControlPolicy::kZeroHold);
  EXPECT_EQ(decision.reason, "waiting_for_vehicle_state");
}

TEST(Px4ControlLogic, InitWaitsForPoseUntilOdomFresh) {
  auto inputs = makeInputs();
  inputs.odom_fresh = false;
  const auto decision = neupan_ros::evaluateControlDecision(
      neupan_ros::ControlPhase::kInit, inputs, makeConfig());

  EXPECT_EQ(decision.phase, neupan_ros::ControlPhase::kInit);
  EXPECT_EQ(decision.policy, neupan_ros::ControlPolicy::kZeroHold);
  EXPECT_EQ(decision.reason, "waiting_for_pose");
}

TEST(Px4ControlLogic, PreOffboardHoldsZeroUntilModeSwitch) {
  auto inputs = makeInputs();
  inputs.offboard_active = false;
  const auto decision = neupan_ros::evaluateControlDecision(
      neupan_ros::ControlPhase::kInit, inputs, makeConfig());

  EXPECT_EQ(decision.phase, neupan_ros::ControlPhase::kTakeoff);
  EXPECT_EQ(decision.policy, neupan_ros::ControlPolicy::kZeroHold);
  EXPECT_EQ(decision.reason, "waiting_for_offboard");
}

TEST(Px4ControlLogic, PlannerStallZeroHolds) {
  auto inputs = makeInputs();
  inputs.planner_cmd_fresh = false;
  inputs.planner_cmd_gap_sec = 1.0;
  const auto decision = neupan_ros::evaluateControlDecision(
      neupan_ros::ControlPhase::kCarry, inputs, makeConfig());

  EXPECT_EQ(decision.policy, neupan_ros::ControlPolicy::kZeroHold);
  EXPECT_EQ(decision.reason, "planner_cmd_stale");
}

TEST(Px4ControlLogic, ArrivalSwitchesToFinish) {
  auto inputs = makeInputs();
  inputs.planner_arrived = true;
  const auto decision = neupan_ros::evaluateControlDecision(
      neupan_ros::ControlPhase::kCarry, inputs, makeConfig());

  EXPECT_EQ(decision.phase, neupan_ros::ControlPhase::kFinish);
  EXPECT_EQ(decision.policy, neupan_ros::ControlPolicy::kZeroHold);
  EXPECT_EQ(decision.reason, "mission_finished");
}

TEST(Px4ControlLogic, FinishPhaseStaysInFinishWithoutFcStateReset) {
  auto inputs = makeInputs();
  inputs.planner_arrived = false;
  const auto decision = neupan_ros::evaluateControlDecision(
      neupan_ros::ControlPhase::kFinish, inputs, makeConfig());

  EXPECT_EQ(decision.phase, neupan_ros::ControlPhase::kFinish);
  EXPECT_EQ(decision.policy, neupan_ros::ControlPolicy::kZeroHold);
  EXPECT_EQ(decision.reason, "mission_finished");
}
