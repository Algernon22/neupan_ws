#include "neupan_uav/robot_model.hpp"

#include <algorithm>
#include <cmath>

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

}  // namespace neupan_uav
