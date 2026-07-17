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

}  // namespace

Eigen::Quaterniond quaternionFromMsg(const geometry_msgs::msg::Quaternion& q) {
  return neupan_uav::normalizedOrIdentity(Eigen::Quaterniond(q.w, q.x, q.y, q.z));
}

Eigen::Matrix3d quaternionToRotationMatrix(
    const geometry_msgs::msg::Quaternion& q) {
  return quaternionFromMsg(q).toRotationMatrix();
}

neupan_uav::UavState odometryToState(
    const nav_msgs::msg::Odometry& msg,
    TwistLinearFrame twist_linear_frame) {
  const auto& pos = msg.pose.pose.position;
  const auto& ori = msg.pose.pose.orientation;
  const auto& twist = msg.twist.twist;

  neupan_uav::UavState out;
  out.position_world << finiteOrZero(pos.x), finiteOrZero(pos.y),
      finiteOrZero(pos.z);
  out.attitude_world_body = quaternionFromMsg(ori);
  out.velocity_world << finiteOrZero(twist.linear.x),
      finiteOrZero(twist.linear.y), finiteOrZero(twist.linear.z);
  if (twist_linear_frame == TwistLinearFrame::kBody) {
    out.velocity_world = out.attitude_world_body.toRotationMatrix() *
                         out.velocity_world;
  }
  out.yaw_rate = finiteOrZero(twist.angular.z);
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
    const neupan_uav::UavState& state) {
  if (points_body.rows() != 3 || points_body.cols() == 0) {
    return neupan_uav::emptyPointMatrix();
  }
  const Eigen::Matrix3d rot =
      neupan_uav::normalizedOrIdentity(state.attitude_world_body)
          .toRotationMatrix();
  return (rot * points_body).colwise() + state.position_world;
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
