#pragma once

#include "neupan_uav/types.hpp"

#include <Eigen/Dense>

#include <limits>

namespace neupan_uav {

struct RobotModelConfig {
  Eigen::Vector3d body_half_extent =
      Eigen::Vector3d(0.23, 0.23, 0.06);
  Control max_control = Control::Constant(
      std::numeric_limits<Scalar>::infinity());
};

class RobotModel {
 public:
  explicit RobotModel(RobotModelConfig config = RobotModelConfig());

  const RobotModelConfig& config() const { return config_; }
  Control clampControl(const Control& control) const;

 private:
  RobotModelConfig config_;
};

}  // namespace neupan_uav
