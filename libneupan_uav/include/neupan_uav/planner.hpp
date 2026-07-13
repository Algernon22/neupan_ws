#pragma once

#include "neupan_uav/config.hpp"
#include "neupan_uav/obstacle_preselector.hpp"
#include "neupan_uav/pan.hpp"
#include "neupan_uav/robot_model.hpp"
#include "neupan_uav/types.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace neupan_uav {

class Planner {
 public:
  explicit Planner(const PlannerConfig& config);

  PlannerOutput forward(const PlannerInput& input);
  void setRknnRunner(std::unique_ptr<RknnRunner> runner);

  // The applied control is [vx, vy, vz, yaw_rate]. Linear velocities are m/s;
  // yaw rate is rad/s. This value must be the command actually published or
  // applied downstream, not a planner output that has not been published.
  template <typename Derived>
  void notifyAppliedControl(const Eigen::MatrixBase<Derived>& control) {
    previous_applied_control_ = normalizeControl(control);
  }

  void reset();

  const PlannerConfig& config() const { return config_; }
  Control previousAppliedControl() const { return previous_applied_control_; }

 private:
  template <typename Derived>
  static Control normalizeControl(const Eigen::MatrixBase<Derived>& control) {
    const auto tmp = control.derived().eval();
    if (tmp.size() != 4) {
      throw std::invalid_argument(
          "applied control must have 4 elements: [vx, vy, vz, yaw_rate]");
    }

    Control normalized;
    if (tmp.rows() == 4 && tmp.cols() == 1) {
      for (Eigen::Index i = 0; i < 4; ++i) normalized(i) = tmp(i, 0);
    } else if (tmp.rows() == 1 && tmp.cols() == 4) {
      for (Eigen::Index i = 0; i < 4; ++i) normalized(i) = tmp(0, i);
    } else {
      for (Eigen::Index i = 0; i < 4; ++i) normalized(i) = tmp.data()[i];
    }

    for (Eigen::Index i = 0; i < normalized.size(); ++i) {
      if (!std::isfinite(normalized(i))) {
        throw std::invalid_argument("applied control must be finite");
      }
    }
    return normalized;
  }

  PlannerOutput invalidOutput(const std::string& reason) const;
  PanInput buildPanInput(const PlannerInput& input,
                         const ObstacleSelection& selected,
                         const Control& seed,
                         FarfieldGuideProfile* farfield_profile);
  Control desiredControl(const Eigen::VectorXd& state) const;
  bool hasArrived(const Eigen::VectorXd& state);
  double minBodyClearance(const Eigen::VectorXd& state,
                          const PointMatrix& points) const;
  void initializePathCache();
  bool hasInitialPath() const;
  void updatePathProgress(const Eigen::VectorXd& state);
  double projectPathProgress(const Eigen::Vector3d& position) const;
  Eigen::Vector4d samplePath(double progress_s) const;
  Eigen::Matrix<Scalar, 4, Eigen::Dynamic> referenceGeometry() const;
  void resetAppliedControl();
  void resetControlBuffer();

  PlannerConfig config_;
  RobotModel robot_;
  ObstaclePreselector preselector_;
  FarfieldGuide farfield_guide_;
  PAN pan_;
  Control previous_applied_control_ = Control::Zero();
  Eigen::MatrixXd control_buffer_;
  std::vector<Eigen::Vector4d> path_waypoints_;
  std::vector<double> path_s_;
  double path_progress_s_ = 0.0;
  bool arrive_latched_ = false;
};

}  // namespace neupan_uav
