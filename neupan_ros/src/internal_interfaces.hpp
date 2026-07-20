#pragma once

namespace neupan_ros::internal {

inline constexpr char kPlannerCommandTopic[] = "neupan/planner/cmd_vel";
inline constexpr char kPlannerArrivedTopic[] = "neupan/planner/arrived";
inline constexpr char kPlannerLocalPathTopic[] = "neupan/planner/local_path";
inline constexpr char kPlannerReferencePathTopic[] = "neupan/planner/reference_path";
inline constexpr char kExecutedCommandTopic[] = "neupan/control/executed_cmd_vel";
inline constexpr char kDefaultCommandFrame[] = "camera_init";

}  // namespace neupan_ros::internal
