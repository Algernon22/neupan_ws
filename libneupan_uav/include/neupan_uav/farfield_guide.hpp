#pragma once

#include "neupan_uav/types.hpp"

#include <Eigen/Dense>

#include <limits>
#include <utility>

namespace neupan_uav {

struct FarfieldGuideConfig {
  bool enabled = false;
  double range_backoff = 1.0;
  double range_scale = 1.5;
  double range_far_limit = std::numeric_limits<double>::infinity();
  double lateral_width = 5.0;
  double center_width = 2.0;
  double height_window = 1.5;
  Eigen::Vector3d voxel_size = Eigen::Vector3d(1.0, 1.0, 0.8);
  int trigger_count = 8;
  double offset_min = 1.5;
  double offset_max = 4.0;
  double offset_speed_gain = 0.6;
  double offset_alpha = 0.20;
  double release_alpha = 0.08;
  int release_count = -1;
  int release_confirm_cycles = 3;
};

struct FarfieldGuideProfile {
  bool active = false;
  double total_sec = 0.0;
  double near_m = 0.0;
  double far_m = 0.0;
  double target_offset_m = 0.0;
  double offset_m = 0.0;
  int center_count = 0;
  int left_count = 0;
  int right_count = 0;
  int release_streak = 0;
};

struct FarfieldGuideResult {
  Eigen::Matrix<Scalar, 4, Eigen::Dynamic> reference_geometry;
  FarfieldGuideProfile profile;
};

class FarfieldGuide {
 public:
  explicit FarfieldGuide(FarfieldGuideConfig config = FarfieldGuideConfig());

  FarfieldGuideResult apply(
      const Eigen::Matrix<Scalar, 4, Eigen::Dynamic>& reference_geometry,
      const PointMatrix& obstacle_points,
      const Eigen::VectorXd& state,
      double ref_speed,
      double step_time,
      int receding);

  void reset();

  const FarfieldGuideConfig& config() const { return config_; }
  double offset() const { return offset_; }

 private:
  struct OccupancyStats {
    int center_count = 0;
    int left_count = 0;
    int right_count = 0;
  };

  FarfieldGuideProfile emptyProfile() const;
  std::pair<double, double> rangeFromReference(
      const Eigen::Matrix<Scalar, 4, Eigen::Dynamic>& reference_geometry,
      double ref_speed,
      double step_time,
      int receding) const;
  std::pair<double, double> limitRangeToVisibleWindow(
      double near_m,
      double far_m) const;
  double targetAbsOffset(double ref_speed) const;
  double updateTargetOffset(const OccupancyStats& stats,
                            double target_abs_offset);
  int lockedSide() const;
  static int chooseClearSide(const OccupancyStats& stats);
  OccupancyStats occupancyStats(
      const PointMatrix& points,
      const Eigen::Matrix<Scalar, 4, Eigen::Dynamic>& reference_geometry,
      const Eigen::VectorXd& state,
      double near_m,
      double far_m) const;
  Eigen::Matrix<Scalar, 4, Eigen::Dynamic> shiftReferenceLaterally(
      const Eigen::Matrix<Scalar, 4, Eigen::Dynamic>& reference_geometry,
      const Eigen::VectorXd& state,
      double offset) const;

  FarfieldGuideConfig config_;
  double offset_ = 0.0;
  int side_ = 0;
  int release_streak_ = 0;
  bool releasing_ = false;
};

}  // namespace neupan_uav
