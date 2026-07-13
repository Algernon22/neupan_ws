#pragma once

#include "neupan_uav/config.hpp"

#include <string>

namespace neupan_ros {

struct LoadedPlannerConfig {
  neupan_uav::PlannerConfig planner;
  bool farfield_guide_present = false;
};

LoadedPlannerConfig loadPlannerConfig(const std::string& yaml_path);

}  // namespace neupan_ros
