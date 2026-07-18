#include "neupan_uav/robot_model.hpp"

#include "detail/box_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace neupan_uav {

RobotModel::RobotModel(RobotModelConfig config) : config_(std::move(config)) {}

Control RobotModel::clampControl(const Control& control) const {
  Control out = control;
  for (Eigen::Index i = 0; i < out.size(); ++i) {
    const double limit = config_.max_control(i);
    if (std::isfinite(limit)) {
      out(i) = std::max(-limit, std::min(limit, out(i)));
    }
  }
  return out;
}

double RobotModel::minClearance(
    const UavState& state,
    const PointMatrix& obstacle_points_world) const {
  if (obstacle_points_world.cols() == 0) {
    return std::numeric_limits<double>::infinity();
  }

  const Eigen::Quaterniond q_world_body =
      normalizedOrIdentity(state.attitude_world_body);
  const Eigen::Matrix3d rotation_body_world =
      q_world_body.toRotationMatrix().transpose();
  const Eigen::Vector3d half_extent =
      config_.body_half_extent.cwiseMax(0.0);

  double min_clearance_squared = std::numeric_limits<double>::infinity();
  for (Eigen::Index col = 0; col < obstacle_points_world.cols(); ++col) {
    const Eigen::Vector3d point_body =
        rotation_body_world *
        (obstacle_points_world.col(col) - state.position_world);
    min_clearance_squared = std::min(
        min_clearance_squared,
        detail::pointToBoxClearanceSquared(point_body, half_extent));
  }
  return std::sqrt(min_clearance_squared);
}

}  // namespace neupan_uav
