#include "neupan_uav/planner.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <utility>

namespace {

neupan_uav::PlannerInput basicInput() {
  neupan_uav::PlannerInput input;
  input.state = Eigen::Vector4d(1.0, 2.0, 3.0, 0.4);
  input.obstacle_points = neupan_uav::emptyPointMatrix();
  return input;
}

neupan_uav::PlannerConfig buildConfig(neupan_uav::PlannerConfig config = {}) {
  neupan_uav::UavPlannerConfigSpec spec;
  spec.planner = std::move(config);
  return neupan_uav::buildUavPlannerConfig(std::move(spec));
}

neupan_uav::PlannerConfig configWithCommand(
    const neupan_uav::Control& command) {
  neupan_uav::PlannerConfig config;
  config.placeholder_command = command;
  config.robot.max_control =
      neupan_uav::Control::Constant(std::numeric_limits<double>::infinity());
  return buildConfig(std::move(config));
}

}  // namespace

TEST(PlannerWarmStart, InitialPreviousCommandIsZero) {
  neupan_uav::Planner planner(buildConfig());

  EXPECT_TRUE(planner.previousCommand().isZero());

  const neupan_uav::PlannerOutput out = planner.forward(basicInput());
  EXPECT_TRUE(out.ready);
  EXPECT_EQ(out.reason, "planner_ok");
  EXPECT_TRUE(out.seed_control.isZero());
  EXPECT_TRUE(planner.previousCommand().isZero());
}

TEST(PlannerWarmStart, SuccessfulForwardSeedsNextCycleWithPublishedCommand) {
  const neupan_uav::Control desired =
      (neupan_uav::Control() << 0.1, 0.2, 0.3, 0.4).finished();
  neupan_uav::Planner planner(configWithCommand(desired));

  const neupan_uav::PlannerOutput first = planner.forward(basicInput());
  ASSERT_TRUE(first.ready);
  EXPECT_TRUE(first.seed_control.isZero());
  EXPECT_TRUE(first.command.isApprox(desired));
  EXPECT_TRUE(planner.previousCommand().isApprox(first.command));

  const neupan_uav::PlannerOutput second = planner.forward(basicInput());
  ASSERT_TRUE(second.ready);
  EXPECT_TRUE(second.seed_control.isApprox(first.command));
  EXPECT_TRUE(planner.previousCommand().isApprox(second.command));
}

TEST(PlannerWarmStart, StoresClampedCommandForNextSeed) {
  neupan_uav::PlannerConfig config;
  config.placeholder_command << 2.0, -2.0, 0.5, 3.0;
  config.robot.max_control << 0.4, 0.3, 1.0, 0.2;
  neupan_uav::Planner planner(buildConfig(std::move(config)));

  const neupan_uav::PlannerOutput first = planner.forward(basicInput());
  ASSERT_TRUE(first.ready);
  EXPECT_TRUE(first.command.isApprox(
      (neupan_uav::Control() << 0.4, -0.3, 0.5, 0.2).finished()));
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
  neupan_uav::PlannerConfig config;
  config.placeholder_command << 0.7, 0.0, 0.0, 0.0;
  config.has_goal = true;
  config.goal_position = Eigen::Vector3d(9.0, 2.0, 3.0);
  config.arrive_threshold = 0.5;
  neupan_uav::Planner planner(buildConfig(std::move(config)));
  ASSERT_TRUE(planner.forward(basicInput()).ready);
  ASSERT_FALSE(planner.previousCommand().isZero());

  neupan_uav::PlannerInput arrived_input = basicInput();
  arrived_input.state = Eigen::Vector4d(9.0, 2.0, 3.0, 0.4);
  const neupan_uav::PlannerOutput out = planner.forward(arrived_input);

  EXPECT_TRUE(out.ready);
  EXPECT_TRUE(out.arrive);
  EXPECT_EQ(out.reason, "arrived");
  EXPECT_TRUE(out.command.isZero());
  EXPECT_FALSE(out.seed_control.isZero());
  EXPECT_TRUE(planner.previousCommand().isZero());
}

TEST(PlannerWarmStart, StopReturnsZeroAndResetsNextSeed) {
  neupan_uav::PlannerConfig config;
  config.placeholder_command << 0.7, 0.0, 0.0, 0.0;
  config.collision_threshold = 0.2;
  config.robot.body_half_extent = Eigen::Vector3d::Zero();
  neupan_uav::Planner planner(buildConfig(std::move(config)));
  ASSERT_TRUE(planner.forward(basicInput()).ready);
  ASSERT_FALSE(planner.previousCommand().isZero());

  neupan_uav::PlannerInput input = basicInput();
  input.obstacle_points.resize(3, 1);
  input.obstacle_points.col(0) = input.state.head<3>();

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
