#pragma once

#include "neupan_uav/config.hpp"
#include "neupan_uav/obstacle_preselector.hpp"
#include "neupan_uav/pan.hpp"
#include "neupan_uav/planner_result.hpp"
#include "neupan_uav/robot_model.hpp"
#include "neupan_uav/types.hpp"

#include <Eigen/Dense>

#include <memory>
#include <string>
#include <vector>

namespace neupan_uav {

class Planner {
 public:
  explicit Planner(const CompiledPlannerConfig& config);

  PlannerResult forward(const PlannerInput& input);
  void setRknnRunner(std::unique_ptr<RknnRunner> runner);
  void setExecutedCommand(const Control& command);

  void reset();

  const CompiledPlannerConfig& config() const { return config_; }
  Control previousCommand() const { return executed_command_; }
  Control executedCommand() const { return executed_command_; }

 private:
  PlannerFault classifyPlanningException(const std::exception& error) const;
  PanInput buildPanInput(const PlannerInput& input,
                         const ObstacleSelection& selected,
                         const Control& seed,
                         FarfieldGuideProfile* farfield_profile);
  Control desiredControl(const DynamicsState& state) const;
  bool hasArrived(const DynamicsState& state);
  void initializePathCache();
  bool hasInitialPath() const;
  void updatePathProgress(const DynamicsState& state);
  double projectPathProgress(const Eigen::Vector3d& position) const;
  Eigen::Vector4d samplePath(double progress_s) const;
  Eigen::Matrix<Scalar, 4, Eigen::Dynamic> referenceGeometry() const;
  void clearExecutedCommand();

  CompiledPlannerConfig config_;
  RobotModel robot_;
  ObstaclePreselector preselector_;
  FarfieldGuide farfield_guide_;
  PAN pan_;
  Control executed_command_ = Control::Zero();
  std::vector<Eigen::Vector4d> path_waypoints_;
  std::vector<double> path_s_;
  double path_progress_s_ = 0.0;
  double last_yaw_ = std::numeric_limits<double>::quiet_NaN();
  bool arrive_latched_ = false;
};

}  // namespace neupan_uav
