#pragma once

#include "neupan_uav/config.hpp"

#include <string>

namespace neupan_ros {

struct LoadedPlannerConfig {
  neupan_uav::PlannerOptions options;
  neupan_uav::UavDynamicsConfig dynamics;
  double state_weight_gain = 1.0;
};

LoadedPlannerConfig loadPlannerConfig(const std::string& yaml_path);
neupan_uav::CompiledPlannerConfig compileLoadedPlannerConfig(
    const LoadedPlannerConfig& loaded);

}  // namespace neupan_ros
