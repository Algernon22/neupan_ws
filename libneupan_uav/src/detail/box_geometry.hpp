#pragma once

#include <Eigen/Dense>

namespace neupan_uav::detail {

inline double pointToBoxClearanceSquared(
    const Eigen::Vector3d& point_local,
    const Eigen::Vector3d& half_extent) noexcept {
  const Eigen::Vector3d delta =
      (point_local.cwiseAbs() - half_extent).cwiseMax(0.0);
  return delta.squaredNorm();
}

}  // namespace neupan_uav::detail
