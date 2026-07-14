#include "neupan_ros/adapters.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace neupan_ros {

namespace {

double finiteOrZero(double value) {
  return std::isfinite(value) ? value : 0.0;
}

const sensor_msgs::msg::PointField* findField(
    const sensor_msgs::msg::PointCloud2& msg, const std::string& name) {
  for (const auto& field : msg.fields) {
    if (field.name == name) return &field;
  }
  return nullptr;
}

bool checkedMul(std::size_t lhs, std::size_t rhs, std::size_t& out) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    return false;
  }
  out = lhs * rhs;
  return true;
}

bool validFloat32Field(const sensor_msgs::msg::PointField* field,
                       std::size_t point_step) {
  if (field == nullptr) return false;
  if (field->datatype != sensor_msgs::msg::PointField::FLOAT32) return false;
  if (field->count < 1) return false;
  const std::size_t offset = static_cast<std::size_t>(field->offset);
  return offset <= point_step && sizeof(float) <= point_step - offset;
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

std::optional<neupan_uav::PointMatrix> readXyzPoints(
    const sensor_msgs::msg::PointCloud2& msg) {
  if (msg.is_bigendian) return std::nullopt;
  if (msg.point_step == 0) return std::nullopt;

  const std::size_t point_step = static_cast<std::size_t>(msg.point_step);
  const auto* x_field = findField(msg, "x");
  const auto* y_field = findField(msg, "y");
  const auto* z_field = findField(msg, "z");
  if (!validFloat32Field(x_field, point_step) ||
      !validFloat32Field(y_field, point_step) ||
      !validFloat32Field(z_field, point_step)) {
    return std::nullopt;
  }

  const std::size_t width =
      static_cast<std::size_t>(msg.width);
  const std::size_t height =
      static_cast<std::size_t>(msg.height);
  if (width == 0 || height == 0) return neupan_uav::emptyPointMatrix();

  std::size_t min_row_step = 0;
  if (!checkedMul(width, point_step, min_row_step)) return std::nullopt;
  const std::size_t row_step = static_cast<std::size_t>(msg.row_step);
  if (row_step < min_row_step) return std::nullopt;

  std::size_t required = 0;
  if (!checkedMul(row_step, height, required)) return std::nullopt;
  if (msg.data.size() < required) return std::nullopt;

  std::size_t point_count = 0;
  if (!checkedMul(width, height, point_count)) return std::nullopt;

  std::vector<Eigen::Vector3d> finite_points;
  finite_points.reserve(point_count);
  const std::size_t x_offset = static_cast<std::size_t>(x_field->offset);
  const std::size_t y_offset = static_cast<std::size_t>(y_field->offset);
  const std::size_t z_offset = static_cast<std::size_t>(z_field->offset);
  for (std::size_t row = 0; row < height; ++row) {
    const std::size_t row_base = row * row_step;
    for (std::size_t col = 0; col < width; ++col) {
      const std::size_t base = row_base + col * point_step;
      const float x = readFloat32(msg.data, base + x_offset);
      const float y = readFloat32(msg.data, base + y_offset);
      const float z = readFloat32(msg.data, base + z_offset);
      if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
        finite_points.emplace_back(static_cast<double>(x),
                                   static_cast<double>(y),
                                   static_cast<double>(z));
      }
    }
  }
  if (point_count > 0 && finite_points.empty()) return std::nullopt;

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
