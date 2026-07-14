#pragma once

#include "neupan_ros/px4_control_logic.hpp"
#include "neupan_uav/types.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

#include "geometry_msgs/msg/twist_stamped.hpp"
#include "mavros_msgs/msg/state.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/string.hpp"

namespace neupan_ros {

class Px4ControlNode final : public rclcpp::Node {
 public:
  explicit Px4ControlNode(
      const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

 private:
  double nowSec();
  double sourceAgeSec(std::uint64_t stamp_ns, double now_s) const;
  bool sourceStampAcceptable(const std::string& key_prefix,
                             const std::string& source_name,
                             std::uint64_t stamp_ns,
                             const std::optional<std::uint64_t>& last_stamp_ns,
                             double max_age_s, double now_s);
  bool topicFresh(const std::optional<std::uint64_t>& stamp_ns,
                  double timeout_s, double now_s) const;
  bool plannerCommandFrameValid(const std::string& frame_id);
  void logThrottledWarn(const std::string& key, double period_s,
                        const std::string& message);
  ControlInputs buildInputs();
  void publishDebugTopic(const ControlDecision& decision);
  void publishSetpoint(const neupan_uav::Control& control);
  neupan_uav::Control takeoffSetpoint() const;
  void timerCallback();
  void cmdCallback(const geometry_msgs::msg::TwistStamped::SharedPtr msg);
  void plannerArrivedCallback(const std_msgs::msg::Bool::SharedPtr msg);
  void statusCallback(const mavros_msgs::msg::State::SharedPtr msg);
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);

  std::string mavros_state_topic_;
  std::string state_odom_topic_;
  std::string setpoint_topic_;
  std::string command_frame_;
  std::string control_debug_topic_;

  double command_timeout_s_ = 0.30;
  double state_odom_timeout_s_ = 0.30;
  double stamp_future_tolerance_s_ = 0.020;
  double takeoff_climb_speed_ = 1.0;
  double takeoff_max_climb_speed_ = 2.0;
  ControlConfig config_;

  std::optional<mavros_msgs::msg::State> latest_status_;
  std::optional<nav_msgs::msg::Odometry> latest_odom_;
  std::optional<std::uint64_t> latest_odom_stamp_ns_;
  std::optional<neupan_uav::Control> latest_cmd_;
  std::optional<std::uint64_t> latest_cmd_stamp_ns_;
  bool planner_arrived_ = false;
  std::string last_warned_cmd_frame_;
  std::unordered_map<std::string, double> last_log_time_;
  ControlPhase phase_ = ControlPhase::kInit;
  ControlDecision decision_;

  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr setpoint_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr debug_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr arrived_sub_;
  rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr status_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
};

}  // namespace neupan_ros
