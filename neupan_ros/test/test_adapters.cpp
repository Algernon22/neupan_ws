#include "neupan_ros/adapters.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>

#include "sensor_msgs/msg/point_field.hpp"

namespace {

geometry_msgs::msg::Quaternion yawQuaternion(double yaw) {
  geometry_msgs::msg::Quaternion q;
  q.w = std::cos(yaw * 0.5);
  q.z = std::sin(yaw * 0.5);
  return q;
}

void writeFloat(std::vector<std::uint8_t>& data, std::size_t offset,
                float value) {
  std::memcpy(data.data() + offset, &value, sizeof(float));
}

sensor_msgs::msg::PointCloud2 makeCloud() {
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.height = 1;
  cloud.width = 3;
  cloud.point_step = 16;
  cloud.row_step = cloud.point_step * cloud.width;
  cloud.is_dense = false;
  cloud.fields.resize(3);
  cloud.fields[0].name = "x";
  cloud.fields[0].offset = 0;
  cloud.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud.fields[0].count = 1;
  cloud.fields[1].name = "y";
  cloud.fields[1].offset = 4;
  cloud.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud.fields[1].count = 1;
  cloud.fields[2].name = "z";
  cloud.fields[2].offset = 8;
  cloud.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud.fields[2].count = 1;
  cloud.data.resize(cloud.row_step);
  writeFloat(cloud.data, 0, 1.0F);
  writeFloat(cloud.data, 4, 2.0F);
  writeFloat(cloud.data, 8, 3.0F);
  writeFloat(cloud.data, 16, std::numeric_limits<float>::quiet_NaN());
  writeFloat(cloud.data, 20, 4.0F);
  writeFloat(cloud.data, 24, 5.0F);
  writeFloat(cloud.data, 32, -1.0F);
  writeFloat(cloud.data, 36, -2.0F);
  writeFloat(cloud.data, 40, -3.0F);
  return cloud;
}

}  // namespace

TEST(Adapters, OdometryToStatesMatchesPythonConvention) {
  nav_msgs::msg::Odometry odom;
  odom.pose.pose.position.x = 10.0;
  odom.pose.pose.position.y = 20.0;
  odom.pose.pose.position.z = 30.0;
  odom.pose.pose.orientation = yawQuaternion(M_PI / 2.0);
  odom.twist.twist.linear.x = 1.0;
  odom.twist.twist.linear.y = 0.0;
  odom.twist.twist.linear.z = 0.5;
  odom.twist.twist.angular.z = 0.4;

  const neupan_uav::UavState state = neupan_ros::odometryToState(odom);
  const neupan_uav::DynamicsState dynamics = neupan_uav::toDynamicsState(state);

  EXPECT_NEAR(state.position_world(0), 10.0, 1e-12);
  EXPECT_NEAR(state.position_world(1), 20.0, 1e-12);
  EXPECT_NEAR(state.position_world(2), 30.0, 1e-12);
  EXPECT_NEAR(dynamics(3), M_PI / 2.0, 1e-12);
  EXPECT_NEAR(state.velocity_world(0), 0.0, 1e-12);
  EXPECT_NEAR(state.velocity_world(1), 1.0, 1e-12);
  EXPECT_NEAR(state.velocity_world(2), 0.5, 1e-12);
  EXPECT_NEAR(state.yaw_rate, 0.4, 1e-12);
}

TEST(Adapters, OdometryCanAcceptWorldFrameTwist) {
  nav_msgs::msg::Odometry odom;
  odom.pose.pose.orientation = yawQuaternion(M_PI / 2.0);
  odom.twist.twist.linear.x = 1.0;
  odom.twist.twist.linear.y = 2.0;
  odom.twist.twist.linear.z = 3.0;

  const neupan_uav::UavState state =
      neupan_ros::odometryToState(odom, neupan_ros::TwistLinearFrame::kWorld);

  EXPECT_NEAR(state.velocity_world(0), 1.0, 1e-12);
  EXPECT_NEAR(state.velocity_world(1), 2.0, 1e-12);
  EXPECT_NEAR(state.velocity_world(2), 3.0, 1e-12);
}

TEST(Adapters, UnwrapNearKeepsYawContinuous) {
  const double reference = M_PI - 0.05;
  const double wrapped = -M_PI + 0.04;

  const double unwrapped = neupan_uav::unwrapNear(wrapped, reference);

  EXPECT_NEAR(unwrapped, M_PI + 0.04, 1e-12);
}

TEST(Adapters, ReadsFiniteXyzPoints) {
  const auto parsed = neupan_ros::readXyzPoints(makeCloud());

  ASSERT_TRUE(parsed.has_value());
  const neupan_uav::PointMatrix& points = *parsed;
  ASSERT_EQ(points.rows(), 3);
  ASSERT_EQ(points.cols(), 2);
  EXPECT_NEAR(points(0, 0), 1.0, 1e-6);
  EXPECT_NEAR(points(1, 0), 2.0, 1e-6);
  EXPECT_NEAR(points(2, 0), 3.0, 1e-6);
  EXPECT_NEAR(points(0, 1), -1.0, 1e-6);
  EXPECT_NEAR(points(1, 1), -2.0, 1e-6);
  EXPECT_NEAR(points(2, 1), -3.0, 1e-6);
}

TEST(Adapters, ReadsValidEmptyPointCloud) {
  sensor_msgs::msg::PointCloud2 cloud = makeCloud();
  cloud.width = 0;
  cloud.height = 1;
  cloud.row_step = 0;
  cloud.data.clear();

  const auto parsed = neupan_ros::readXyzPoints(cloud);

  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->rows(), 3);
  EXPECT_EQ(parsed->cols(), 0);
}

TEST(Adapters, RejectsPointCloudMissingXyzField) {
  sensor_msgs::msg::PointCloud2 cloud = makeCloud();
  cloud.fields.pop_back();

  EXPECT_FALSE(neupan_ros::readXyzPoints(cloud).has_value());
}

TEST(Adapters, RejectsPointCloudWithWrongDatatype) {
  sensor_msgs::msg::PointCloud2 cloud = makeCloud();
  cloud.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT64;

  EXPECT_FALSE(neupan_ros::readXyzPoints(cloud).has_value());
}

TEST(Adapters, RejectsPointCloudWithOffsetBeyondPointStep) {
  sensor_msgs::msg::PointCloud2 cloud = makeCloud();
  cloud.fields[2].offset = 14;

  EXPECT_FALSE(neupan_ros::readXyzPoints(cloud).has_value());
}

TEST(Adapters, RejectsPointCloudWithShortData) {
  sensor_msgs::msg::PointCloud2 cloud = makeCloud();
  cloud.data.resize(cloud.data.size() - 1);

  EXPECT_FALSE(neupan_ros::readXyzPoints(cloud).has_value());
}

TEST(Adapters, ReadsOrganizedPointCloudWithRowPadding) {
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.height = 2;
  cloud.width = 2;
  cloud.point_step = 12;
  cloud.row_step = 32;
  cloud.is_bigendian = false;
  cloud.fields.resize(3);
  cloud.fields[0].name = "x";
  cloud.fields[0].offset = 0;
  cloud.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud.fields[0].count = 1;
  cloud.fields[1].name = "y";
  cloud.fields[1].offset = 4;
  cloud.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud.fields[1].count = 1;
  cloud.fields[2].name = "z";
  cloud.fields[2].offset = 8;
  cloud.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud.fields[2].count = 1;
  cloud.data.resize(static_cast<std::size_t>(cloud.row_step) * cloud.height);
  writeFloat(cloud.data, 0, 1.0F);
  writeFloat(cloud.data, 4, 2.0F);
  writeFloat(cloud.data, 8, 3.0F);
  writeFloat(cloud.data, 12, 4.0F);
  writeFloat(cloud.data, 16, 5.0F);
  writeFloat(cloud.data, 20, 6.0F);
  writeFloat(cloud.data, 32, 7.0F);
  writeFloat(cloud.data, 36, 8.0F);
  writeFloat(cloud.data, 40, 9.0F);
  writeFloat(cloud.data, 44, 10.0F);
  writeFloat(cloud.data, 48, 11.0F);
  writeFloat(cloud.data, 52, 12.0F);

  const auto parsed = neupan_ros::readXyzPoints(cloud);

  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->cols(), 4);
  EXPECT_NEAR((*parsed)(0, 0), 1.0, 1e-6);
  EXPECT_NEAR((*parsed)(0, 2), 7.0, 1e-6);
  EXPECT_NEAR((*parsed)(2, 3), 12.0, 1e-6);
}

TEST(Adapters, RejectsAllNanPointCloudWithRawPoints) {
  sensor_msgs::msg::PointCloud2 cloud = makeCloud();
  for (std::size_t offset = 0; offset < cloud.data.size(); offset += 4) {
    writeFloat(cloud.data, offset, std::numeric_limits<float>::quiet_NaN());
  }

  EXPECT_FALSE(neupan_ros::readXyzPoints(cloud).has_value());
}

TEST(Adapters, RejectsBigEndianPointCloud) {
  sensor_msgs::msg::PointCloud2 cloud = makeCloud();
  cloud.is_bigendian = true;

  EXPECT_FALSE(neupan_ros::readXyzPoints(cloud).has_value());
}

TEST(Adapters, TransformsBodyPointsToWorld) {
  neupan_uav::UavState state;
  state.position_world << 10.0, 20.0, 30.0;
  state.attitude_world_body =
      Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ()));
  neupan_uav::PointMatrix body(3, 2);
  body << 1.0, 0.0, 0.0, 2.0, 0.0, 0.0;

  const neupan_uav::PointMatrix world =
      neupan_ros::pointsBodyToWorld(body, state);
  ASSERT_EQ(world.cols(), 2);
  EXPECT_NEAR(world(0, 0), 10.0, 1e-12);
  EXPECT_NEAR(world(1, 0), 21.0, 1e-12);
  EXPECT_NEAR(world(0, 1), 8.0, 1e-12);
  EXPECT_NEAR(world(1, 1), 20.0, 1e-12);
}
