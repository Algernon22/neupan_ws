#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cmath>
#include <cstddef>
#include <limits>

namespace neupan_uav {

using Scalar = double;
using Control = Eigen::Matrix<Scalar, 4, 1>;
using DynamicsState = Eigen::Matrix<Scalar, 8, 1>;
using PointMatrix = Eigen::Matrix<Scalar, 3, Eigen::Dynamic>;
using Trajectory = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

inline PointMatrix emptyPointMatrix() { return PointMatrix(3, 0); }

struct UavState {
  Eigen::Vector3d position_world = Eigen::Vector3d::Zero();
  Eigen::Quaterniond attitude_world_body = Eigen::Quaterniond::Identity();
  Eigen::Vector3d velocity_world = Eigen::Vector3d::Zero();
  double yaw_rate = 0.0;
};

struct Rpy {
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
};

inline double finiteOrZero(double value) {
  return std::isfinite(value) ? value : 0.0;
}

inline double unwrapNear(double yaw, double reference) {
  return reference + std::remainder(yaw - reference, 2.0 * M_PI);
}

inline Eigen::Quaterniond normalizedOrIdentity(Eigen::Quaterniond q) {
  if (!std::isfinite(q.w()) || !std::isfinite(q.x()) ||
      !std::isfinite(q.y()) || !std::isfinite(q.z()) ||
      q.squaredNorm() <= 1.0e-18) {
    return Eigen::Quaterniond::Identity();
  }
  q.normalize();
  return q;
}

inline Rpy quaternionToRpy(const Eigen::Quaterniond& q_in) {
  const Eigen::Quaterniond q = normalizedOrIdentity(q_in);
  const double w = q.w();
  const double x = q.x();
  const double y = q.y();
  const double z = q.z();

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

inline DynamicsState toDynamicsState(const UavState& state,
                                     double yaw_reference =
                                         std::numeric_limits<double>::quiet_NaN()) {
  DynamicsState x = DynamicsState::Zero();
  Rpy rpy = quaternionToRpy(state.attitude_world_body);
  if (std::isfinite(yaw_reference)) {
    rpy.yaw = unwrapNear(rpy.yaw, yaw_reference);
  }

  x.segment<3>(0) = state.position_world;
  x(3) = rpy.yaw;
  x.segment<3>(4) = state.velocity_world;
  x(7) = state.yaw_rate;
  for (Eigen::Index i = 0; i < x.size(); ++i) x(i) = finiteOrZero(x(i));
  return x;
}

inline Eigen::Vector3d attitudeRpy(const UavState& state,
                                   double yaw_reference =
                                       std::numeric_limits<double>::quiet_NaN()) {
  Rpy rpy = quaternionToRpy(state.attitude_world_body);
  if (std::isfinite(yaw_reference)) {
    rpy.yaw = unwrapNear(rpy.yaw, yaw_reference);
  }
  return Eigen::Vector3d(finiteOrZero(rpy.roll), finiteOrZero(rpy.pitch),
                         finiteOrZero(rpy.yaw));
}

struct PlannerProfile {
  double forward_sec = 0.0;
  double preselect_sec = 0.0;
  double preselect_corridor_sec = 0.0;
  double preselect_distance_sec = 0.0;
  double preselect_select_sec = 0.0;
  double dune_sec = 0.0;
  double dune_inference_sec = 0.0;
  double dune_select_sec = 0.0;
  double nrmp_sec = 0.0;
  std::size_t input_obstacle_count = 0;
  std::size_t corridor_obstacle_count = 0;
  std::size_t preselected_obstacle_count = 0;
  std::size_t preselect_max_count = 0;
  std::size_t hard_count = 0;
  std::size_t nearest_quota = 0;
  std::size_t nearest_selected = 0;
  std::size_t temporal_quota = 0;
  std::size_t continuity_hits = 0;
  std::size_t continuity_selected = 0;
  std::size_t diversity_quota = 0;
  std::size_t diversity_candidates = 0;
  std::size_t diversity_selected = 0;
  std::size_t fill_selected = 0;
  std::size_t dune_selected_count = 0;
  int pan_iterations = 0;
  int pan_iteration_limit = 0;
  int osqp_status = 0;
  int osqp_iteration_count = 0;
  double osqp_solve_sec = 0.0;
  double osqp_run_time_sec = 0.0;
  double farfield_sec = 0.0;
  bool farfield_active = false;
  double farfield_near_m = 0.0;
  double farfield_far_m = 0.0;
  double farfield_offset_m = 0.0;
  double farfield_target_offset_m = 0.0;
  int farfield_center_count = 0;
  int farfield_left_count = 0;
  int farfield_right_count = 0;
  int farfield_release_streak = 0;
};

struct PlannerInput {
  UavState state;
  PointMatrix obstacle_points = emptyPointMatrix();
  PointMatrix obstacle_velocities = emptyPointMatrix();
  bool valid = true;
  bool stale = false;
};

}  // namespace neupan_uav
