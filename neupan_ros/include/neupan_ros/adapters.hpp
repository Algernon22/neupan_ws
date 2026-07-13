#pragma once

#include "neupan_uav/types.hpp"

#include <Eigen/Dense>

#include <cstdint>
#include <optional>
#include <vector>

#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

namespace neupan_ros {

struct OdomStates {
  Eigen::Matrix<double, 6, 1> state6 = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 4, 1> state4 = Eigen::Matrix<double, 4, 1>::Zero();
  Eigen::Matrix<double, 4, 1> twist4 = Eigen::Matrix<double, 4, 1>::Zero();
};

struct Rpy {
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
};

Rpy quaternionToRpy(const geometry_msgs::msg::Quaternion& q);
Eigen::Matrix3d quaternionToRotationMatrix(
    const geometry_msgs::msg::Quaternion& q);
OdomStates odometryToStates(const nav_msgs::msg::Odometry& msg);
neupan_uav::PointMatrix readXyzPoints(const sensor_msgs::msg::PointCloud2& msg);
neupan_uav::PointMatrix pointsBodyToWorld(
    const neupan_uav::PointMatrix& points_body,
    const Eigen::Matrix<double, 6, 1>& state6);
neupan_uav::PointMatrix pointsWorldToBody(
    const neupan_uav::PointMatrix& points_world,
    const Eigen::Matrix<double, 6, 1>& state6);
double minBodyClearance(const neupan_uav::PointMatrix& points_body,
                        const Eigen::Vector3d& body_half_extent);

std::uint64_t stampToNanoseconds(const builtin_interfaces::msg::Time& stamp);
double stampNanosecondsToSeconds(std::uint64_t stamp_ns);

geometry_msgs::msg::TwistStamped controlToTwistStamped(
    const neupan_uav::Control& control,
    const builtin_interfaces::msg::Time& stamp,
    const std::string& frame_id);
neupan_uav::Control twistStampedToControl(
    const geometry_msgs::msg::TwistStamped& msg);

}  // namespace neupan_ros
