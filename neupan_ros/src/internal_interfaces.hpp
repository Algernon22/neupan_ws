#pragma once

namespace neupan_ros::internal {

inline constexpr char kPlannerCommandTopic[] = "neupan/planner/cmd_vel";
inline constexpr char kPlannerArrivedTopic[] = "neupan/planner/arrived";
inline constexpr char kAppliedCommandTopic[] = "neupan/control/applied_cmd_vel";
inline constexpr char kDefaultCommandFrame[] = "camera_init";

}  // namespace neupan_ros::internal
