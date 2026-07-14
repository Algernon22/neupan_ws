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

TEST(PlannerWarmStart, InitialPreviousAppliedControlIsZero) {
  neupan_uav::Planner planner(buildConfig());

  EXPECT_TRUE(planner.previousAppliedControl().isZero());

  const neupan_uav::PlannerOutput out = planner.forward(basicInput());
  EXPECT_TRUE(out.ready);
  EXPECT_EQ(out.reason, "planner_ok");
  EXPECT_TRUE(out.seed_control.isZero());
  EXPECT_TRUE(planner.previousAppliedControl().isZero());
}

TEST(PlannerWarmStart, NotifyAppliedControlSeedsNextForward) {
  neupan_uav::Planner planner(buildConfig());
  const neupan_uav::Control applied =
      (neupan_uav::Control() << 1.0, -2.0, 0.5, 0.25).finished();

  planner.notifyAppliedControl(applied);

  EXPECT_TRUE(planner.previousAppliedControl().isApprox(applied));
  const neupan_uav::PlannerOutput out = planner.forward(basicInput());
  EXPECT_TRUE(out.seed_control.isApprox(applied));
  EXPECT_TRUE(planner.previousAppliedControl().isApprox(applied));
}

TEST(PlannerWarmStart, ForwardDoesNotAutoApplyPlannerOutput) {
  const neupan_uav::Control desired =
      (neupan_uav::Control() << 0.1, 0.2, 0.3, 0.4).finished();
  neupan_uav::Planner planner(configWithCommand(desired));
  const neupan_uav::Control applied =
      (neupan_uav::Control() << -1.0, 0.0, 0.2, -0.3).finished();
  planner.notifyAppliedControl(applied);

  const neupan_uav::PlannerOutput out = planner.forward(basicInput());

  EXPECT_TRUE(out.command.isApprox(desired));
  EXPECT_TRUE(out.seed_control.isApprox(applied));
  EXPECT_TRUE(planner.previousAppliedControl().isApprox(applied));
}

TEST(PlannerWarmStart, AcceptsFlatAndRowAppliedControlShapes) {
  neupan_uav::Planner planner(buildConfig());
  Eigen::RowVector4d row;
  row << 0.1, 0.2, 0.3, 0.4;

  planner.notifyAppliedControl(row);
  EXPECT_TRUE(planner.previousAppliedControl().isApprox(row.transpose()));

  Eigen::VectorXd flat(4);
  flat << 4.0, 3.0, 2.0, 1.0;
  planner.notifyAppliedControl(flat);
  EXPECT_TRUE(planner.previousAppliedControl().isApprox(flat));
}

TEST(PlannerWarmStart, RejectsBadAppliedControl) {
  neupan_uav::Planner planner(buildConfig());

  EXPECT_THROW(planner.notifyAppliedControl(Eigen::Vector3d::Zero()),
               std::invalid_argument);

  Eigen::Vector4d bad = Eigen::Vector4d::Zero();
  bad(2) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(planner.notifyAppliedControl(bad), std::invalid_argument);
}

TEST(PlannerWarmStart, ResetClearsPreviousAppliedControl) {
  neupan_uav::Planner planner(buildConfig());
  planner.notifyAppliedControl(
      (neupan_uav::Control() << 1.0, 2.0, 3.0, 4.0).finished());

  planner.reset();

  EXPECT_TRUE(planner.previousAppliedControl().isZero());
}

TEST(PlannerWarmStart, ArriveReturnsZeroAndResetsNextSeed) {
  neupan_uav::PlannerConfig config;
  config.has_goal = true;
  config.goal_position = Eigen::Vector3d(1.0, 2.0, 3.0);
  config.arrive_threshold = 0.5;
  neupan_uav::Planner planner(buildConfig(std::move(config)));
  const neupan_uav::Control applied =
      (neupan_uav::Control() << 1.0, 2.0, 3.0, 4.0).finished();
  planner.notifyAppliedControl(applied);

  const neupan_uav::PlannerOutput out = planner.forward(basicInput());

  EXPECT_TRUE(out.ready);
  EXPECT_TRUE(out.arrive);
  EXPECT_EQ(out.reason, "arrived");
  EXPECT_TRUE(out.command.isZero());
  EXPECT_TRUE(out.seed_control.isApprox(applied));
  EXPECT_TRUE(planner.previousAppliedControl().isZero());
}

TEST(PlannerWarmStart, StopReturnsZeroAndResetsNextSeed) {
  neupan_uav::PlannerConfig config;
  config.collision_threshold = 0.2;
  config.robot.body_half_extent = Eigen::Vector3d::Zero();
  neupan_uav::Planner planner(buildConfig(std::move(config)));
  const neupan_uav::Control applied =
      (neupan_uav::Control() << -1.0, 0.1, 0.2, 0.3).finished();
  planner.notifyAppliedControl(applied);

  neupan_uav::PlannerInput input = basicInput();
  input.obstacle_points.resize(3, 1);
  input.obstacle_points.col(0) = input.state.head<3>();

  const neupan_uav::PlannerOutput out = planner.forward(input);

  EXPECT_TRUE(out.ready);
  EXPECT_TRUE(out.stop);
  EXPECT_EQ(out.reason, "planner_stop");
  EXPECT_TRUE(out.command.isZero());
  EXPECT_TRUE(out.seed_control.isApprox(applied));
  EXPECT_TRUE(planner.previousAppliedControl().isZero());
}

TEST(PlannerWarmStart, StaleAndInvalidInputsDoNotResetSeed) {
  neupan_uav::Planner planner(buildConfig());
  const neupan_uav::Control applied =
      (neupan_uav::Control() << 0.7, 0.6, 0.5, 0.4).finished();
  planner.notifyAppliedControl(applied);

  neupan_uav::PlannerInput stale = basicInput();
  stale.stale = true;
  const neupan_uav::PlannerOutput stale_out = planner.forward(stale);
  EXPECT_FALSE(stale_out.ready);
  EXPECT_EQ(stale_out.reason, "stale_input");
  EXPECT_TRUE(stale_out.seed_control.isApprox(applied));
  EXPECT_TRUE(planner.previousAppliedControl().isApprox(applied));

  neupan_uav::PlannerInput invalid = basicInput();
  invalid.valid = false;
  const neupan_uav::PlannerOutput invalid_out = planner.forward(invalid);
  EXPECT_FALSE(invalid_out.ready);
  EXPECT_EQ(invalid_out.reason, "invalid_input");
  EXPECT_TRUE(invalid_out.seed_control.isApprox(applied));
  EXPECT_TRUE(planner.previousAppliedControl().isApprox(applied));
}
