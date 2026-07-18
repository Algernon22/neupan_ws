#include "neupan_uav/robot_model.hpp"

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <cmath>

namespace {

neupan_uav::UavState stateWithAttitude(const Eigen::Quaterniond& attitude) {
  neupan_uav::UavState state;
  state.position_world.setZero();
  state.attitude_world_body = attitude;
  return state;
}

neupan_uav::RobotModel modelWithHalfExtent(const Eigen::Vector3d& half_extent) {
  neupan_uav::RobotModelConfig config;
  config.body_half_extent = half_extent;
  return neupan_uav::RobotModel(config);
}

neupan_uav::PointMatrix point(double x, double y, double z) {
  neupan_uav::PointMatrix points(3, 1);
  points << x, y, z;
  return points;
}

}  // namespace

TEST(RobotModel, IdentityAttitudeMatchesAxisAlignedBox) {
  const neupan_uav::RobotModel robot =
      modelWithHalfExtent(Eigen::Vector3d(1.0, 0.5, 0.25));
  const neupan_uav::UavState state =
      stateWithAttitude(Eigen::Quaterniond::Identity());

  EXPECT_NEAR(robot.minClearance(state, point(1.4, 0.8, 0.25)), 0.5, 1e-12);
}

TEST(RobotModel, YawRotationAvoidsFalseNegative) {
  const neupan_uav::RobotModel robot =
      modelWithHalfExtent(Eigen::Vector3d(2.0, 0.1, 0.1));
  const neupan_uav::UavState state = stateWithAttitude(
      Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 2.0,
                                           Eigen::Vector3d::UnitZ())));

  EXPECT_NEAR(robot.minClearance(state, point(0.0, 1.5, 0.0)), 0.0, 1e-12);
}

TEST(RobotModel, YawRotationAvoidsFalsePositive) {
  const neupan_uav::RobotModel robot =
      modelWithHalfExtent(Eigen::Vector3d(2.0, 0.1, 0.1));
  const neupan_uav::UavState state = stateWithAttitude(
      Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 2.0,
                                           Eigen::Vector3d::UnitZ())));

  EXPECT_NEAR(robot.minClearance(state, point(1.5, 0.0, 0.0)), 1.4, 1e-12);
}

TEST(RobotModel, RollAndPitchAffectClearance) {
  const neupan_uav::RobotModel robot =
      modelWithHalfExtent(Eigen::Vector3d(0.1, 2.0, 0.1));
  const Eigen::Quaterniond attitude =
      Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitY());
  const neupan_uav::UavState state = stateWithAttitude(attitude);

  EXPECT_NEAR(robot.minClearance(state, point(0.0, 0.0, 1.5)), 0.0, 1e-12);
  EXPECT_NEAR(robot.minClearance(state, point(0.0, 1.5, 0.0)), 1.4, 1e-12);
}
