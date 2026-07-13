#include "neupan_uav/farfield_guide.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <string>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace neupan_uav {

namespace {

using Clock = std::chrono::steady_clock;

double elapsedSeconds(const Clock::time_point& start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

double finiteNonnegative(double value, const char* name) {
  if (!std::isfinite(value) || value < 0.0) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and non-negative");
  }
  return value;
}

double finitePositive(double value, const char* name) {
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and positive");
  }
  return value;
}

double ratio(double value, const char* name) {
  if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and within [0, 1]");
  }
  return value;
}

Eigen::Vector3d positiveVector(const Eigen::Vector3d& value,
                               const char* name) {
  if (!value.allFinite() || (value.array() <= 0.0).any()) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and positive");
  }
  return value;
}

struct VoxelKey {
  int x = 0;
  int y = 0;
  int z = 0;

  bool operator==(const VoxelKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelHash {
  std::size_t operator()(const VoxelKey& key) const {
    std::size_t h = static_cast<std::size_t>(key.x) * 73856093u;
    h ^= static_cast<std::size_t>(key.y) * 19349663u;
    h ^= static_cast<std::size_t>(key.z) * 83492791u;
    return h;
  }
};

VoxelKey voxelKey(double x, double y, double z,
                  const Eigen::Vector3d& voxel) {
  return {
      static_cast<int>(std::floor(x / voxel(0))),
      static_cast<int>(std::floor(y / voxel(1))),
      static_cast<int>(std::floor(z / voxel(2))),
  };
}

template <typename Derived>
bool allFinite(const Eigen::MatrixBase<Derived>& matrix) {
  for (Eigen::Index i = 0; i < matrix.size(); ++i) {
    if (!std::isfinite(matrix.derived().data()[i])) return false;
  }
  return true;
}

double yawFromPlannerState(const Eigen::VectorXd& state) {
  return state.size() == 6 ? state(5) : state(3);
}

FarfieldGuideConfig normalizeConfig(FarfieldGuideConfig config) {
  config.range_backoff =
      finiteNonnegative(config.range_backoff,
                        "FarfieldGuideConfig::range_backoff");
  if (!std::isfinite(config.range_scale) || config.range_scale <= 1.0) {
    throw std::invalid_argument(
        "FarfieldGuideConfig::range_scale must be finite and > 1.0");
  }
  if (std::isfinite(config.range_far_limit)) {
    config.range_far_limit =
        finitePositive(config.range_far_limit,
                       "FarfieldGuideConfig::range_far_limit");
  }
  config.lateral_width =
      finitePositive(config.lateral_width,
                     "FarfieldGuideConfig::lateral_width");
  config.center_width =
      finitePositive(config.center_width,
                     "FarfieldGuideConfig::center_width");
  if (config.center_width > config.lateral_width) {
    throw std::invalid_argument(
        "FarfieldGuideConfig::center_width must be <= lateral_width");
  }
  config.height_window =
      finitePositive(config.height_window,
                     "FarfieldGuideConfig::height_window");
  config.voxel_size =
      positiveVector(config.voxel_size,
                     "FarfieldGuideConfig::voxel_size");
  config.trigger_count = std::max(1, config.trigger_count);
  if (config.release_count < 0) {
    config.release_count = std::max(0, config.trigger_count / 2);
  }
  if (config.release_count >= config.trigger_count) {
    throw std::invalid_argument(
        "FarfieldGuideConfig::release_count must be < trigger_count");
  }
  config.release_confirm_cycles =
      std::max(1, config.release_confirm_cycles);
  config.offset_min =
      finiteNonnegative(config.offset_min,
                        "FarfieldGuideConfig::offset_min");
  config.offset_max =
      finiteNonnegative(config.offset_max,
                        "FarfieldGuideConfig::offset_max");
  if (config.offset_min > config.offset_max) {
    throw std::invalid_argument(
        "FarfieldGuideConfig::offset_min must be <= offset_max");
  }
  config.offset_speed_gain =
      finiteNonnegative(config.offset_speed_gain,
                        "FarfieldGuideConfig::offset_speed_gain");
  config.offset_alpha =
      ratio(config.offset_alpha, "FarfieldGuideConfig::offset_alpha");
  config.release_alpha =
      ratio(config.release_alpha, "FarfieldGuideConfig::release_alpha");
  return config;
}

}  // namespace

FarfieldGuide::FarfieldGuide(FarfieldGuideConfig config)
    : config_(normalizeConfig(std::move(config))) {
  reset();
}

void FarfieldGuide::reset() {
  offset_ = 0.0;
  side_ = 0;
  release_streak_ = 0;
  releasing_ = false;
}

FarfieldGuideProfile FarfieldGuide::emptyProfile() const {
  FarfieldGuideProfile profile;
  profile.offset_m = offset_;
  profile.release_streak = release_streak_;
  return profile;
}

FarfieldGuideResult FarfieldGuide::apply(
    const Eigen::Matrix<Scalar, 4, Eigen::Dynamic>& reference_geometry,
    const PointMatrix& obstacle_points,
    const Eigen::VectorXd& state,
    double ref_speed,
    double step_time,
    int receding) {
  const auto start = Clock::now();
  FarfieldGuideResult result;
  result.reference_geometry = reference_geometry;
  result.profile = emptyProfile();

  if (!config_.enabled) {
    result.profile.total_sec = elapsedSeconds(start);
    return result;
  }
  if (reference_geometry.rows() < 4 ||
      reference_geometry.cols() <= 0 ||
      !allFinite(reference_geometry)) {
    throw std::invalid_argument(
        "farfield reference_geometry must have at least 4 rows and finite columns");
  }
  if (obstacle_points.rows() != 3 || !allFinite(obstacle_points)) {
    throw std::invalid_argument(
        "farfield obstacle_points must have shape 3xN and be finite");
  }
  if (state.size() < 4 || !state.allFinite()) {
    throw std::invalid_argument(
        "farfield state must have at least 4 finite elements");
  }

  const auto range =
      rangeFromReference(reference_geometry, ref_speed, step_time, receding);
  const double near_m = range.first;
  const double far_m = range.second;
  result.profile.near_m = near_m;
  result.profile.far_m = far_m;

  const double target_abs_offset = targetAbsOffset(ref_speed);
  OccupancyStats stats;
  if (obstacle_points.cols() > 0 && far_m > near_m &&
      target_abs_offset > 0.0) {
    stats = occupancyStats(obstacle_points, reference_geometry, state, near_m,
                           far_m);
  }
  result.profile.center_count = stats.center_count;
  result.profile.left_count = stats.left_count;
  result.profile.right_count = stats.right_count;

  double target_offset = 0.0;
  if (target_abs_offset > 0.0) {
    target_offset = updateTargetOffset(stats, target_abs_offset);
  }

  const double alpha =
      std::abs(target_offset) > 1.0e-6 ? config_.offset_alpha
                                       : config_.release_alpha;
  offset_ = (1.0 - alpha) * offset_ + alpha * target_offset;
  offset_ = std::clamp(offset_, -config_.offset_max, config_.offset_max);
  if (std::abs(target_offset) <= 1.0e-6 && std::abs(offset_) < 1.0e-3) {
    offset_ = 0.0;
    side_ = 0;
    release_streak_ = 0;
    releasing_ = false;
  }

  result.profile.active = std::abs(offset_) > 1.0e-3;
  result.profile.target_offset_m = target_offset;
  result.profile.offset_m = offset_;
  result.profile.release_streak = release_streak_;

  if (std::abs(offset_) > 1.0e-6) {
    result.reference_geometry =
        shiftReferenceLaterally(reference_geometry, state, offset_);
  }
  result.profile.total_sec = elapsedSeconds(start);
  return result;
}

std::pair<double, double> FarfieldGuide::rangeFromReference(
    const Eigen::Matrix<Scalar, 4, Eigen::Dynamic>& reference_geometry,
    double ref_speed,
    double step_time,
    int receding) const {
  double horizon_m = 0.0;
  if (reference_geometry.cols() <= 1) {
    horizon_m = std::max(0.0, ref_speed) * std::max(0.0, step_time) *
                static_cast<double>(std::max(0, receding));
  } else {
    for (Eigen::Index col = 1; col < reference_geometry.cols(); ++col) {
      horizon_m +=
          (reference_geometry.col(col).head<3>() -
           reference_geometry.col(col - 1).head<3>()).norm();
    }
  }
  const double near_m = std::max(0.0, horizon_m - config_.range_backoff);
  const double far_m = near_m * config_.range_scale;
  return limitRangeToVisibleWindow(near_m, far_m);
}

std::pair<double, double> FarfieldGuide::limitRangeToVisibleWindow(
    double near_m,
    double far_m) const {
  if (!std::isfinite(config_.range_far_limit) ||
      far_m <= config_.range_far_limit) {
    return {near_m, far_m};
  }

  far_m = config_.range_far_limit;
  if (near_m >= far_m) {
    const double tail_window_m =
        std::max(2.0, 2.0 * config_.voxel_size(0));
    near_m = std::max(0.0, far_m - tail_window_m);
  }
  return {near_m, far_m};
}

double FarfieldGuide::targetAbsOffset(double ref_speed) const {
  if (config_.offset_max <= 0.0) return 0.0;
  const double raw = config_.offset_speed_gain * std::max(0.0, ref_speed);
  return std::clamp(raw, config_.offset_min, config_.offset_max);
}

double FarfieldGuide::updateTargetOffset(const OccupancyStats& stats,
                                         double target_abs_offset) {
  if (stats.center_count >= config_.trigger_count) {
    int side = lockedSide();
    if (side == 0) side = chooseClearSide(stats);
    side_ = side;
    release_streak_ = 0;
    releasing_ = false;
    return static_cast<double>(side) * target_abs_offset;
  }

  if (side_ == 0) {
    release_streak_ = 0;
    releasing_ = false;
    return 0.0;
  }
  if (releasing_) return 0.0;

  if (stats.center_count <= config_.release_count) {
    ++release_streak_;
    if (release_streak_ >= config_.release_confirm_cycles) {
      releasing_ = true;
      return 0.0;
    }
  } else {
    release_streak_ = 0;
  }
  return static_cast<double>(side_) * target_abs_offset;
}

int FarfieldGuide::lockedSide() const {
  if (side_ != 0) return side_;
  if (std::abs(offset_) > 0.2) return offset_ > 0.0 ? 1 : -1;
  return 0;
}

int FarfieldGuide::chooseClearSide(const OccupancyStats& stats) {
  return stats.left_count <= stats.right_count ? 1 : -1;
}

FarfieldGuide::OccupancyStats FarfieldGuide::occupancyStats(
    const PointMatrix& points,
    const Eigen::Matrix<Scalar, 4, Eigen::Dynamic>& reference_geometry,
    const Eigen::VectorXd& state,
    double near_m,
    double far_m) const {
  OccupancyStats stats;
  Eigen::Vector2d forward = Eigen::Vector2d::Zero();
  if (reference_geometry.cols() > 1) {
    forward =
        reference_geometry.col(reference_geometry.cols() - 1).head<2>() -
        reference_geometry.col(0).head<2>();
    const double norm = forward.norm();
    if (norm > 1.0e-6) {
      forward /= norm;
    } else {
      forward.setZero();
    }
  }
  if (forward.norm() <= 1.0e-6) {
    const double yaw = yawFromPlannerState(state);
    forward << std::cos(yaw), std::sin(yaw);
  }
  const Eigen::Vector2d left(-forward(1), forward(0));
  const Eigen::Vector3d origin = state.head<3>();
  const double ref_z =
      reference_geometry.cols() > 0 ? reference_geometry(2, 0) : origin(2);

  std::unordered_set<VoxelKey, VoxelHash> center_voxels;
  std::unordered_set<VoxelKey, VoxelHash> left_voxels;
  std::unordered_set<VoxelKey, VoxelHash> right_voxels;
  bool need_sides = false;

  for (Eigen::Index col = 0; col < points.cols(); ++col) {
    const Eigen::Vector3d delta = points.col(col) - origin;
    if (!delta.allFinite()) continue;
    const double forward_dist = forward.dot(delta.head<2>());
    const double lateral = left.dot(delta.head<2>());
    const double vertical = points(2, col) - ref_z;
    if (forward_dist < near_m || forward_dist > far_m ||
        lateral < -config_.lateral_width ||
        lateral > config_.lateral_width ||
        vertical < -config_.height_window ||
        vertical > config_.height_window) {
      continue;
    }
    if (lateral >= -config_.center_width &&
        lateral <= config_.center_width) {
      center_voxels.insert(
          voxelKey(forward_dist, lateral, vertical, config_.voxel_size));
      if (static_cast<int>(center_voxels.size()) >= config_.trigger_count) {
        need_sides = true;
      }
    }
    if (need_sides) {
      if (lateral > config_.center_width) {
        left_voxels.insert(
            voxelKey(forward_dist, lateral, vertical, config_.voxel_size));
      } else if (lateral < -config_.center_width) {
        right_voxels.insert(
            voxelKey(forward_dist, lateral, vertical, config_.voxel_size));
      }
    }
  }

  stats.center_count = static_cast<int>(center_voxels.size());
  if (stats.center_count >= config_.trigger_count) {
    stats.left_count = static_cast<int>(left_voxels.size());
    stats.right_count = static_cast<int>(right_voxels.size());
  }
  return stats;
}

Eigen::Matrix<Scalar, 4, Eigen::Dynamic>
FarfieldGuide::shiftReferenceLaterally(
    const Eigen::Matrix<Scalar, 4, Eigen::Dynamic>& reference_geometry,
    const Eigen::VectorXd& state,
    double offset) const {
  Eigen::Matrix<Scalar, 4, Eigen::Dynamic> shifted = reference_geometry;
  if (shifted.cols() <= 0) return shifted;

  Eigen::Vector2d forward = Eigen::Vector2d::Zero();
  if (shifted.cols() > 1) {
    forward = shifted.col(shifted.cols() - 1).head<2>() -
              shifted.col(0).head<2>();
    const double norm = forward.norm();
    if (norm > 1.0e-6) {
      forward /= norm;
    } else {
      forward.setZero();
    }
  }
  if (forward.norm() <= 1.0e-6) {
    const double yaw = state.size() >= 4 ? yawFromPlannerState(state) : 0.0;
    forward << std::cos(yaw), std::sin(yaw);
  }
  const Eigen::Vector2d left(-forward(1), forward(0));
  const double denom =
      shifted.cols() > 1 ? static_cast<double>(shifted.cols() - 1) : 1.0;
  for (Eigen::Index col = 0; col < shifted.cols(); ++col) {
    const double ramp = denom > 0.0 ? static_cast<double>(col) / denom : 0.0;
    shifted(0, col) += left(0) * offset * ramp;
    shifted(1, col) += left(1) * offset * ramp;
  }
  return shifted;
}

}  // namespace neupan_uav
