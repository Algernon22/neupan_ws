#pragma once

#include "neupan_uav/types.hpp"

#include <Eigen/Dense>

#include <array>
#include <cstddef>
#include <vector>

namespace neupan_uav {

enum SelectionTag : unsigned char {
  kTemporalSelectionTag = 1u << 0,
  kDiversitySelectionTag = 1u << 1,
};

inline bool hasTemporalTag(unsigned char tag) {
  return (tag & kTemporalSelectionTag) != 0u;
}

inline bool hasDiversityTag(unsigned char tag) {
  return (tag & kDiversitySelectionTag) != 0u;
}

struct ObstaclePreselectorConfig {
  bool enabled = true;
  std::size_t max_points = 0;
  int per_step = 4;
  double dt = 0.1;
  Eigen::Vector3d body_half_extent = Eigen::Vector3d::Zero();
  Eigen::Vector3d corridor_margin = Eigen::Vector3d(1.0, 1.0, 0.6);
  Eigen::Vector3d exact_margin = Eigen::Vector3d(0.20, 0.20, 0.12);
  Eigen::Vector3d diversity_voxel = Eigen::Vector3d(0.8, 0.8, 0.4);
  double hard_distance = 0.45;
  double nearest_ratio = 0.60;
  bool temporal_enabled = true;
  double temporal_radius = 0.45;
  double temporal_ratio = 0.20;
  bool diversity_enabled = true;
  double diversity_ratio = 0.20;
};

struct ObstacleSelection {
  PointMatrix points = emptyPointMatrix();
  PointMatrix velocities = emptyPointMatrix();
  // One byte per selected point. Bit 0 marks temporal-continuity selections;
  // bit 1 marks diversity-quota selections.
  std::vector<unsigned char> tags;
  PlannerProfile profile;
};

class ObstaclePreselector {
 public:
  explicit ObstaclePreselector(std::size_t max_points = 0);
  explicit ObstaclePreselector(ObstaclePreselectorConfig config);

  ObstacleSelection select(const PointMatrix& points);
  ObstacleSelection select(const PointMatrix& points,
                           const PointMatrix& velocities);
  ObstacleSelection selectWithNominalTrajectory(
      const Trajectory& nominal_states, const PointMatrix& points,
      const PointMatrix& velocities = emptyPointMatrix(),
      const Eigen::MatrixXd& attitude_horizon = Eigen::MatrixXd());

  void reset();
  const PointMatrix& previousPoints() const { return previous_points_; }
  void setPreviousPoints(const PointMatrix& points);
  const ObstaclePreselectorConfig& config() const { return config_; }

 private:
  ObstacleSelection selectByColumnLimit(const PointMatrix& points,
                                        const PointMatrix& velocities) const;

  ObstaclePreselectorConfig config_;
  PointMatrix previous_points_ = emptyPointMatrix();
};

struct PointFlowConfig {
  int receding = 0;
  double dt = 0.1;
  std::size_t dune_max_num = 0;
  Eigen::Vector3d body_half_extent = Eigen::Vector3d::Zero();
};

struct PointFlowResult {
  std::vector<PointMatrix> point_flow;
  std::vector<Eigen::Matrix3d> rotations;
  std::vector<PointMatrix> obstacle_points_by_step;
  PointMatrix selected_points = emptyPointMatrix();
  PointMatrix selected_velocities = emptyPointMatrix();
  bool fallback_applied = false;
};

class PointFlowBuilder {
 public:
  explicit PointFlowBuilder(PointFlowConfig config);

  PointFlowResult generate(
      const Trajectory& nominal_states, const PointMatrix& obstacle_points,
      const PointMatrix& velocities = emptyPointMatrix(),
      const Eigen::MatrixXd& attitude_horizon = Eigen::MatrixXd()) const;

 private:
  PointFlowConfig config_;
};

}  // namespace neupan_uav
