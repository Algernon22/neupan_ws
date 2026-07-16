#include "neupan_uav/planner.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <utility>

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
  options.placeholder_command = command;
  options.robot.max_control =
      neupan_uav::Control::Constant(std::numeric_limits<double>::infinity());
  return buildConfig(std::move(options));
}

}  // namespace

TEST(PlannerWarmStart, InitialPreviousCommandIsZero) {
  neupan_uav::Planner planner(buildConfig());

  EXPECT_TRUE(planner.previousCommand().isZero());

  const neupan_uav::PlannerOutput out = planner.forward(basicInput());
  EXPECT_TRUE(out.ready);
  EXPECT_EQ(out.reason, "planner_ok");
  EXPECT_TRUE(out.seed_control.isZero());
  EXPECT_TRUE(planner.previousCommand().allFinite());
}

TEST(PlannerWarmStart, SuccessfulForwardSeedsNextCycleWithPublishedCommand) {
  neupan_uav::PlannerOptions options;
  options.robot.max_control =
      neupan_uav::Control::Constant(std::numeric_limits<double>::infinity());
  options.initial_path.waypoints = {
      Eigen::Vector4d(1.0, 2.0, 3.0, 0.4),
      Eigen::Vector4d(2.0, 2.0, 3.0, 0.4),
  };
  neupan_uav::Planner planner(buildConfig(std::move(options)));

  const neupan_uav::PlannerOutput first = planner.forward(basicInput());
  ASSERT_TRUE(first.ready);
  EXPECT_TRUE(first.seed_control.isZero());
  EXPECT_FALSE(first.command.isZero());
  EXPECT_TRUE(planner.previousCommand().isApprox(first.command));

  const neupan_uav::PlannerOutput second = planner.forward(basicInput());
  ASSERT_TRUE(second.ready);
  EXPECT_TRUE(second.seed_control.isApprox(first.command));
  EXPECT_TRUE(planner.previousCommand().isApprox(second.command));
}

TEST(PlannerWarmStart, StoresClampedCommandForNextSeed) {
  neupan_uav::PlannerOptions options;
  options.placeholder_command << 2.0, -2.0, 0.5, 3.0;
  options.robot.max_control << 0.4, 0.3, 1.0, 0.2;
  neupan_uav::Planner planner(buildConfig(std::move(options)));

  const neupan_uav::PlannerOutput first = planner.forward(basicInput());
  ASSERT_TRUE(first.ready);
  EXPECT_LE(first.command.cwiseAbs()(0), 0.4 + 1.0e-12);
  EXPECT_LE(first.command.cwiseAbs()(1), 0.3 + 1.0e-12);
  EXPECT_LE(first.command.cwiseAbs()(2), 1.0 + 1.0e-12);
  EXPECT_LE(first.command.cwiseAbs()(3), 0.2 + 1.0e-12);
  EXPECT_TRUE(planner.previousCommand().isApprox(first.command));

  const neupan_uav::PlannerOutput second = planner.forward(basicInput());
  ASSERT_TRUE(second.ready);
  EXPECT_TRUE(second.seed_control.isApprox(first.command));
}

TEST(PlannerWarmStart, ResetClearsPreviousCommand) {
  neupan_uav::Planner planner(configWithCommand(
      (neupan_uav::Control() << 1.0, 2.0, 3.0, 4.0).finished()));
  ASSERT_TRUE(planner.forward(basicInput()).ready);
  ASSERT_FALSE(planner.previousCommand().isZero());

  planner.reset();

  EXPECT_TRUE(planner.previousCommand().isZero());
}

TEST(PlannerWarmStart, ArriveReturnsZeroAndResetsNextSeed) {
  neupan_uav::PlannerOptions options;
  options.placeholder_command << 0.7, 0.0, 0.0, 0.0;
  options.has_goal = true;
  options.goal_position = Eigen::Vector3d(9.0, 2.0, 3.0);
  options.arrive_threshold = 0.5;
  neupan_uav::Planner planner(buildConfig(std::move(options)));
  ASSERT_TRUE(planner.forward(basicInput()).ready);
  ASSERT_FALSE(planner.previousCommand().isZero());

  neupan_uav::PlannerInput arrived_input = basicInput();
  arrived_input.state = stateAt(9.0, 2.0, 3.0, 0.4);
  const neupan_uav::PlannerOutput out = planner.forward(arrived_input);

  EXPECT_TRUE(out.ready);
  EXPECT_TRUE(out.arrive);
  EXPECT_EQ(out.reason, "arrived");
  EXPECT_TRUE(out.command.isZero());
  EXPECT_FALSE(out.seed_control.isZero());
  EXPECT_TRUE(planner.previousCommand().isZero());
}

TEST(PlannerWarmStart, StopReturnsZeroAndResetsNextSeed) {
  neupan_uav::PlannerOptions options;
  options.placeholder_command << 0.7, 0.0, 0.0, 0.0;
  options.collision_threshold = 0.2;
  options.robot.body_half_extent = Eigen::Vector3d::Zero();
  neupan_uav::Planner planner(buildConfig(std::move(options)));
  ASSERT_TRUE(planner.forward(basicInput()).ready);
  ASSERT_FALSE(planner.previousCommand().isZero());

  neupan_uav::PlannerInput input = basicInput();
  input.obstacle_points.resize(3, 1);
  input.obstacle_points.col(0) = input.state.position_world;

  const neupan_uav::PlannerOutput out = planner.forward(input);

  EXPECT_TRUE(out.ready);
  EXPECT_TRUE(out.stop);
  EXPECT_EQ(out.reason, "planner_stop");
  EXPECT_TRUE(out.command.isZero());
  EXPECT_FALSE(out.seed_control.isZero());
  EXPECT_TRUE(planner.previousCommand().isZero());
}

TEST(PlannerWarmStart, InvalidInputsClearPreviousCommand) {
  neupan_uav::Planner planner(configWithCommand(
      (neupan_uav::Control() << 0.7, 0.6, 0.5, 0.4).finished()));
  ASSERT_TRUE(planner.forward(basicInput()).ready);
  ASSERT_FALSE(planner.previousCommand().isZero());

  neupan_uav::PlannerInput stale = basicInput();
  stale.stale = true;
  const neupan_uav::PlannerOutput stale_out = planner.forward(stale);
  EXPECT_FALSE(stale_out.ready);
  EXPECT_EQ(stale_out.reason, "stale_input");
  EXPECT_FALSE(stale_out.seed_control.isZero());
  EXPECT_TRUE(planner.previousCommand().isZero());

  ASSERT_TRUE(planner.forward(basicInput()).ready);
  neupan_uav::PlannerInput invalid = basicInput();
  invalid.valid = false;
  const neupan_uav::PlannerOutput invalid_out = planner.forward(invalid);
  EXPECT_FALSE(invalid_out.ready);
  EXPECT_EQ(invalid_out.reason, "invalid_input");
  EXPECT_FALSE(invalid_out.seed_control.isZero());
  EXPECT_TRUE(planner.previousCommand().isZero());
}
