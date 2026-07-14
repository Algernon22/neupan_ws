#include "neupan_ros/adapters.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace neupan_ros {

namespace {

double finiteOrZero(double value) {
  return std::isfinite(value) ? value : 0.0;
}

int fieldOffset(const sensor_msgs::msg::PointCloud2& msg,
                const std::string& name) {
  for (const auto& field : msg.fields) {
    if (field.name == name) return static_cast<int>(field.offset);
  }
  return -1;
}

float readFloat32(const std::vector<std::uint8_t>& data, std::size_t offset) {
  float value = 0.0F;
  std::memcpy(&value, data.data() + offset, sizeof(float));
  return value;
}

Eigen::Matrix3d rpyToRotation(double roll, double pitch, double yaw) {
  const double cr = std::cos(roll);
  const double sr = std::sin(roll);
  const double cp = std::cos(pitch);
  const double sp = std::sin(pitch);
  const double cy = std::cos(yaw);
  const double sy = std::sin(yaw);

  Eigen::Matrix3d rot;
  rot << cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
      sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, -sp,
      cp * sr, cp * cr;
  return rot;
}

}  // namespace

Rpy quaternionToRpy(const geometry_msgs::msg::Quaternion& q) {
  const double x = q.x;
  const double y = q.y;
  const double z = q.z;
  const double w = q.w;

  const double sinr_cosp = 2.0 * (w * x + y * z);
  const double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
  const double roll = std::atan2(sinr_cosp, cosr_cosp);

  const double sinp = 2.0 * (w * y - z * x);
  double pitch = 0.0;
  if (std::abs(sinp) >= 1.0) {
    pitch = std::copysign(M_PI / 2.0, sinp);
  } else {
    pitch = std::asin(sinp);
  }

  const double siny_cosp = 2.0 * (w * z + x * y);
  const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
  const double yaw = std::atan2(siny_cosp, cosy_cosp);

  return Rpy{roll, pitch, yaw};
}

Eigen::Matrix3d quaternionToRotationMatrix(
    const geometry_msgs::msg::Quaternion& q) {
  const double x = q.x;
  const double y = q.y;
  const double z = q.z;
  const double w = q.w;
  Eigen::Matrix3d rot;
  rot << 1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),
      2.0 * (x * z + w * y), 2.0 * (x * y + w * z),
      1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
      2.0 * (x * z - w * y), 2.0 * (y * z + w * x),
      1.0 - 2.0 * (x * x + y * y);
  return rot;
}

OdomStates odometryToStates(const nav_msgs::msg::Odometry& msg) {
  const auto& pos = msg.pose.pose.position;
  const auto& ori = msg.pose.pose.orientation;
  const Rpy rpy = quaternionToRpy(ori);

  OdomStates out;
  out.state6 << finiteOrZero(pos.x), finiteOrZero(pos.y), finiteOrZero(pos.z),
      rpy.roll, rpy.pitch, rpy.yaw;
  return out;
}

neupan_uav::PointMatrix readXyzPoints(
    const sensor_msgs::msg::PointCloud2& msg) {
  const int x_offset = fieldOffset(msg, "x");
  const int y_offset = fieldOffset(msg, "y");
  const int z_offset = fieldOffset(msg, "z");
  if (x_offset < 0 || y_offset < 0 || z_offset < 0 || msg.point_step == 0) {
    return neupan_uav::emptyPointMatrix();
  }

  const std::size_t width =
      static_cast<std::size_t>(msg.width) * static_cast<std::size_t>(msg.height);
  const std::size_t point_step = static_cast<std::size_t>(msg.point_step);
  const std::size_t required = width * point_step;
  if (msg.data.size() < required) return neupan_uav::emptyPointMatrix();

  std::vector<Eigen::Vector3d> finite_points;
  finite_points.reserve(width);
  for (std::size_t i = 0; i < width; ++i) {
    const std::size_t base = i * point_step;
    const float x = readFloat32(msg.data, base + static_cast<std::size_t>(x_offset));
    const float y = readFloat32(msg.data, base + static_cast<std::size_t>(y_offset));
    const float z = readFloat32(msg.data, base + static_cast<std::size_t>(z_offset));
    if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
      finite_points.emplace_back(static_cast<double>(x), static_cast<double>(y),
                                 static_cast<double>(z));
    }
  }

  neupan_uav::PointMatrix out(3, static_cast<Eigen::Index>(finite_points.size()));
  for (Eigen::Index i = 0; i < out.cols(); ++i) {
    out.col(i) = finite_points[static_cast<std::size_t>(i)];
  }
  return out;
}

neupan_uav::PointMatrix pointsBodyToWorld(
    const neupan_uav::PointMatrix& points_body,
    const Eigen::Matrix<double, 6, 1>& state6) {
  if (points_body.rows() != 3 || points_body.cols() == 0) {
    return neupan_uav::emptyPointMatrix();
  }
  const Eigen::Vector3d position = state6.head<3>();
  const Eigen::Matrix3d rot =
      rpyToRotation(state6(3), state6(4), state6(5));
  return (rot * points_body).colwise() + position;
}

double minBodyClearance(const neupan_uav::PointMatrix& points_body,
                        const Eigen::Vector3d& body_half_extent) {
  if (points_body.rows() != 3 || points_body.cols() == 0) {
    return std::numeric_limits<double>::infinity();
  }
  double min_clearance = std::numeric_limits<double>::infinity();
  for (Eigen::Index col = 0; col < points_body.cols(); ++col) {
    const Eigen::Vector3d delta =
        (points_body.col(col).cwiseAbs() - body_half_extent).cwiseMax(0.0);
    min_clearance = std::min(min_clearance, delta.norm());
  }
  return min_clearance;
}

std::uint64_t stampToNanoseconds(const builtin_interfaces::msg::Time& stamp) {
  return static_cast<std::uint64_t>(stamp.sec) * 1000000000ULL +
         static_cast<std::uint64_t>(stamp.nanosec);
}

double stampNanosecondsToSeconds(std::uint64_t stamp_ns) {
  return static_cast<double>(stamp_ns) * 1.0e-9;
}

geometry_msgs::msg::TwistStamped controlToTwistStamped(
    const neupan_uav::Control& control,
    const builtin_interfaces::msg::Time& stamp,
    const std::string& frame_id) {
  geometry_msgs::msg::TwistStamped msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  msg.twist.linear.x = control(0);
  msg.twist.linear.y = control(1);
  msg.twist.linear.z = control(2);
  msg.twist.angular.z = control(3);
  return msg;
}

neupan_uav::Control twistStampedToControl(
    const geometry_msgs::msg::TwistStamped& msg) {
  neupan_uav::Control control;
  control << msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
      msg.twist.angular.z;
  return control;
}

}  // namespace neupan_ros
