#include "neupan_uav/planner.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <utility>
#include <variant>

namespace {

neupan_uav::UavState stateAt(double x, double y, double z, double yaw) {
  neupan_uav::UavState state;
  state.position_world << x, y, z;
  state.attitude_world_body =
      Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
  return state;
}

neupan_uav::PlannerInput basicInput() {
  neupan_uav::PlannerInput input;
  input.state = stateAt(1.0, 2.0, 3.0, 0.4);
  input.obstacle_points = neupan_uav::emptyPointMatrix();
  return input;
}

neupan_uav::CompiledPlannerConfig buildConfig(
    neupan_uav::PlannerOptions options = {}) {
  return neupan_uav::compilePlannerConfig(options);
}

neupan_uav::CompiledPlannerConfig configWithCommand(
    const neupan_uav::Control& command) {
  neupan_uav::PlannerOptions options;
  options.default_command = command;
  options.robot.max_control =
      neupan_uav::Control::Constant(std::numeric_limits<double>::infinity());
  return buildConfig(std::move(options));
}

const neupan_uav::Tracking& tracking(
    const neupan_uav::PlannerResult& result) {
  return std::get<neupan_uav::Tracking>(result.decision());
}

}  // namespace

TEST(PlannerWarmStart, InitialPreviousCommandIsZero) {
  neupan_uav::Planner planner(buildConfig());

  EXPECT_TRUE(planner.previousCommand().isZero());

  const neupan_uav::PlannerResult out = planner.forward(basicInput());
  EXPECT_TRUE(out.isTracking());
  EXPECT_TRUE(out.diagnostics().warm_start_seed.isZero());
  EXPECT_TRUE(planner.previousCommand().isZero());
}

TEST(PlannerWarmStart, OnlyExecutedCommandSeedsNextCycle) {
  neupan_uav::PlannerOptions options;
  options.robot.max_control =
      neupan_uav::Control::Constant(std::numeric_limits<double>::infinity());
  options.initial_path.waypoints = {
      Eigen::Vector4d(1.0, 2.0, 3.0, 0.4),
      Eigen::Vector4d(2.0, 2.0, 3.0, 0.4),
  };
  neupan_uav::Planner planner(buildConfig(std::move(options)));

  const neupan_uav::PlannerResult first = planner.forward(basicInput());
  ASSERT_TRUE(first.isTracking());
  EXPECT_TRUE(first.diagnostics().warm_start_seed.isZero());
  EXPECT_FALSE(tracking(first).command.isZero());
  EXPECT_TRUE(planner.previousCommand().isZero());

  const neupan_uav::PlannerResult second = planner.forward(basicInput());
  ASSERT_TRUE(second.isTracking());
  EXPECT_TRUE(second.diagnostics().warm_start_seed.isZero());

  const neupan_uav::Control executed =
      (neupan_uav::Control() << 0.1, -0.2, 0.3, -0.4).finished();
  planner.setExecutedCommand(executed);
  const neupan_uav::PlannerResult third = planner.forward(basicInput());
  ASSERT_TRUE(third.isTracking());
  EXPECT_TRUE(third.diagnostics().warm_start_seed.isApprox(executed));
  EXPECT_TRUE(planner.previousCommand().isApprox(executed));
}

TEST(PlannerWarmStart, StoresClampedExecutedCommandForNextSeed) {
  neupan_uav::PlannerOptions options;
  options.default_command << 2.0, -2.0, 0.5, 3.0;
  options.robot.max_control << 0.4, 0.3, 1.0, 0.2;
  neupan_uav::Planner planner(buildConfig(std::move(options)));

  const neupan_uav::Control executed =
      (neupan_uav::Control() << 2.0, -2.0, 0.5, 3.0).finished();
  planner.setExecutedCommand(executed);
  EXPECT_NEAR(planner.previousCommand()(0), 0.4, 1.0e-12);
  EXPECT_NEAR(planner.previousCommand()(1), -0.3, 1.0e-12);
  EXPECT_NEAR(planner.previousCommand()(2), 0.5, 1.0e-12);
  EXPECT_NEAR(planner.previousCommand()(3), 0.2, 1.0e-12);

  const neupan_uav::PlannerResult second = planner.forward(basicInput());
  ASSERT_TRUE(second.isTracking());
  EXPECT_TRUE(second.diagnostics().warm_start_seed.isApprox(
      planner.previousCommand()));
}

TEST(PlannerWarmStart, ResetClearsPreviousCommand) {
  neupan_uav::Planner planner(configWithCommand(
      (neupan_uav::Control() << 1.0, 2.0, 3.0, 4.0).finished()));
  planner.setExecutedCommand(
      (neupan_uav::Control() << 0.1, 0.2, 0.3, 0.4).finished());
  ASSERT_FALSE(planner.previousCommand().isZero());

  planner.reset();

  EXPECT_TRUE(planner.previousCommand().isZero());
}

TEST(PlannerWarmStart, ArriveReturnsZeroAndResetsNextSeed) {
  neupan_uav::PlannerOptions options;
  options.default_command << 0.7, 0.0, 0.0, 0.0;
  options.has_goal = true;
  options.goal_position = Eigen::Vector3d(9.0, 2.0, 3.0);
  options.arrive_threshold = 0.5;
  neupan_uav::Planner planner(buildConfig(std::move(options)));
  planner.setExecutedCommand(
      (neupan_uav::Control() << 0.7, 0.0, 0.0, 0.0).finished());
  ASSERT_FALSE(planner.previousCommand().isZero());

  neupan_uav::PlannerInput arrived_input = basicInput();
  arrived_input.state = stateAt(9.0, 2.0, 3.0, 0.4);
  const neupan_uav::PlannerResult out = planner.forward(arrived_input);

  EXPECT_TRUE(std::holds_alternative<neupan_uav::GoalReached>(out.decision()));
  EXPECT_TRUE(out.commandToPublish().isZero());
  EXPECT_FALSE(out.diagnostics().warm_start_seed.isZero());
  EXPECT_FALSE(planner.previousCommand().isZero());
}

TEST(PlannerWarmStart, StopReturnsZeroAndResetsNextSeed) {
  neupan_uav::PlannerOptions options;
  options.default_command << 0.7, 0.0, 0.0, 0.0;
  options.collision_threshold = 0.2;
  options.robot.body_half_extent = Eigen::Vector3d::Zero();
  neupan_uav::Planner planner(buildConfig(std::move(options)));
  planner.setExecutedCommand(
      (neupan_uav::Control() << 0.7, 0.0, 0.0, 0.0).finished());
  ASSERT_FALSE(planner.previousCommand().isZero());

  neupan_uav::PlannerInput input = basicInput();
  input.obstacle_points.resize(3, 1);
  input.obstacle_points.col(0) = input.state.position_world;

  const neupan_uav::PlannerResult out = planner.forward(input);

  ASSERT_TRUE(std::holds_alternative<neupan_uav::SafetyStop>(out.decision()));
  const auto& stop = std::get<neupan_uav::SafetyStop>(out.decision());
  EXPECT_EQ(stop.cause, neupan_uav::SafetyStopCause::kClearanceViolation);
  EXPECT_TRUE(out.commandToPublish().isZero());
  EXPECT_FALSE(out.diagnostics().warm_start_seed.isZero());
  EXPECT_FALSE(planner.previousCommand().isZero());
}

TEST(PlannerWarmStart, InvalidInputsClearPreviousCommand) {
  neupan_uav::Planner planner(configWithCommand(
      (neupan_uav::Control() << 0.7, 0.6, 0.5, 0.4).finished()));
  planner.setExecutedCommand(
      (neupan_uav::Control() << 0.7, 0.6, 0.5, 0.4).finished());
  ASSERT_FALSE(planner.previousCommand().isZero());

  neupan_uav::PlannerInput stale = basicInput();
  stale.stale = true;
  const neupan_uav::PlannerResult stale_out = planner.forward(stale);
  ASSERT_TRUE(std::holds_alternative<neupan_uav::FaultStop>(
      stale_out.decision()));
  EXPECT_EQ(std::get<neupan_uav::FaultStop>(stale_out.decision()).fault,
            neupan_uav::PlannerFault::kStaleInput);
  EXPECT_TRUE(stale_out.commandToPublish().isZero());
  EXPECT_FALSE(stale_out.diagnostics().warm_start_seed.isZero());
  EXPECT_FALSE(planner.previousCommand().isZero());

  neupan_uav::PlannerInput invalid = basicInput();
  invalid.valid = false;
  const neupan_uav::PlannerResult invalid_out = planner.forward(invalid);
  ASSERT_TRUE(std::holds_alternative<neupan_uav::FaultStop>(
      invalid_out.decision()));
  EXPECT_EQ(std::get<neupan_uav::FaultStop>(invalid_out.decision()).fault,
            neupan_uav::PlannerFault::kInvalidInput);
  EXPECT_TRUE(invalid_out.commandToPublish().isZero());
  EXPECT_FALSE(invalid_out.diagnostics().warm_start_seed.isZero());
  EXPECT_FALSE(planner.previousCommand().isZero());
}
