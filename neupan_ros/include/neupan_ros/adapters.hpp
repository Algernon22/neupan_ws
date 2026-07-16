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

enum class TwistLinearFrame {
  kBody,
  kWorld,
};

Eigen::Quaterniond quaternionFromMsg(const geometry_msgs::msg::Quaternion& q);
Eigen::Matrix3d quaternionToRotationMatrix(
    const geometry_msgs::msg::Quaternion& q);
neupan_uav::UavState odometryToState(
    const nav_msgs::msg::Odometry& msg,
    TwistLinearFrame twist_linear_frame = TwistLinearFrame::kBody);
std::optional<neupan_uav::PointMatrix> readXyzPoints(
    const sensor_msgs::msg::PointCloud2& msg);
neupan_uav::PointMatrix pointsBodyToWorld(
    const neupan_uav::PointMatrix& points_body,
    const neupan_uav::UavState& state);
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
