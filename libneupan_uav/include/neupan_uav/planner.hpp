#pragma once

#include "neupan_uav/config.hpp"
#include "neupan_uav/obstacle_preselector.hpp"
#include "neupan_uav/pan.hpp"
#include "neupan_uav/robot_model.hpp"
#include "neupan_uav/types.hpp"

#include <Eigen/Dense>

#include <memory>
#include <string>
#include <vector>

namespace neupan_uav {

class Planner {
 public:
  explicit Planner(const PlannerConfig& config);

  PlannerOutput forward(const PlannerInput& input);
  void setRknnRunner(std::unique_ptr<RknnRunner> runner);

  void reset();

  const PlannerConfig& config() const { return config_; }
  Control previousCommand() const { return previous_command_; }

 private:
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
  void clearPreviousCommand();
  void resetControlBuffer();

  PlannerConfig config_;
  RobotModel robot_;
  ObstaclePreselector preselector_;
  FarfieldGuide farfield_guide_;
  PAN pan_;
  Control previous_command_ = Control::Zero();
  Eigen::MatrixXd control_buffer_;
  std::vector<Eigen::Vector4d> path_waypoints_;
  std::vector<double> path_s_;
  double path_progress_s_ = 0.0;
  bool arrive_latched_ = false;
};

}  // namespace neupan_uav
