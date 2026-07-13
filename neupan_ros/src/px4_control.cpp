#include "neupan_ros/px4_control.hpp"

#include "neupan_ros/adapters.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace neupan_ros {

namespace {

std::string upper(std::string value) {
  for (char& c : value) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return value;
}

}  // namespace

Px4ControlNode::Px4ControlNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("px4_control", options) {
  declare_parameter<std::string>("planner_cmd_topic", "/neupan/planner/cmd_vel");
  declare_parameter<std::string>("mavros_state_topic", "/mavros/state");
  declare_parameter<std::string>("state_odom_topic",
                                 "/mavros/local_position/odom");
  declare_parameter<std::string>("mavros_setpoint_velocity_topic",
                                 "/mavros/setpoint_velocity/cmd_vel");
  declare_parameter<std::string>("applied_cmd_topic",
                                 "/neupan/control/applied_cmd_vel");
  declare_parameter<std::string>("planner_arrived_topic",
                                 "/neupan/planner/arrived");
  declare_parameter<std::string>("setpoint_frame", "camera_init");
  declare_parameter<double>("heartbeat_rate", 20.0);
  declare_parameter<double>("command_timeout", 0.30);
  declare_parameter<double>("state_odom_timeout", 0.30);
  declare_parameter<double>("stamp_future_tolerance_ms", 20.0);
  declare_parameter<bool>("enable_takeoff_phase", true);
  declare_parameter<double>("takeoff_phase_height", 1.8);
  declare_parameter<double>("takeoff_phase_hysteresis", 0.1);
  declare_parameter<double>("takeoff_phase_climb_speed", 1.0);
  declare_parameter<double>("takeoff_phase_max_climb_speed", 2.0);
  declare_parameter<std::string>("control_debug_topic", "/neupan/control_debug");

  planner_cmd_topic_ = get_parameter("planner_cmd_topic").as_string();
  mavros_state_topic_ = get_parameter("mavros_state_topic").as_string();
  state_odom_topic_ = get_parameter("state_odom_topic").as_string();
  setpoint_topic_ =
      get_parameter("mavros_setpoint_velocity_topic").as_string();
  applied_cmd_topic_ = get_parameter("applied_cmd_topic").as_string();
  planner_arrived_topic_ =
      get_parameter("planner_arrived_topic").as_string();
  setpoint_frame_ = get_parameter("setpoint_frame").as_string();
  control_debug_topic_ = get_parameter("control_debug_topic").as_string();
  command_timeout_s_ = std::max(0.01, get_parameter("command_timeout").as_double());
  state_odom_timeout_s_ =
      std::max(0.05, get_parameter("state_odom_timeout").as_double());
  stamp_future_tolerance_s_ =
      std::max(0.0, 1.0e-3 * get_parameter("stamp_future_tolerance_ms").as_double());
  takeoff_climb_speed_ =
      std::max(0.0, get_parameter("takeoff_phase_climb_speed").as_double());
  takeoff_max_climb_speed_ =
      std::max(0.0, get_parameter("takeoff_phase_max_climb_speed").as_double());
  const double takeoff_release_height =
      std::max(0.0, get_parameter("takeoff_phase_height").as_double()) +
      std::max(0.0, get_parameter("takeoff_phase_hysteresis").as_double());
  config_.enable_takeoff_phase =
      get_parameter("enable_takeoff_phase").as_bool();
  config_.takeoff_release_height_m = takeoff_release_height;

  const double heartbeat_rate =
      std::max(2.0, get_parameter("heartbeat_rate").as_double());

  setpoint_pub_ =
      create_publisher<geometry_msgs::msg::TwistStamped>(setpoint_topic_, 10);
  applied_cmd_pub_ =
      create_publisher<geometry_msgs::msg::TwistStamped>(applied_cmd_topic_, 10);
  debug_pub_ = create_publisher<std_msgs::msg::String>(control_debug_topic_, 10);

  cmd_sub_ = create_subscription<geometry_msgs::msg::TwistStamped>(
      planner_cmd_topic_, 10,
      [this](geometry_msgs::msg::TwistStamped::SharedPtr msg) {
        cmdCallback(msg);
      });
  arrived_sub_ = create_subscription<std_msgs::msg::Bool>(
      planner_arrived_topic_, 10,
      [this](std_msgs::msg::Bool::SharedPtr msg) {
        plannerArrivedCallback(msg);
      });
  status_sub_ = create_subscription<mavros_msgs::msg::State>(
      mavros_state_topic_, rclcpp::SensorDataQoS(),
      [this](mavros_msgs::msg::State::SharedPtr msg) { statusCallback(msg); });
  odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      state_odom_topic_, rclcpp::SensorDataQoS(),
      [this](nav_msgs::msg::Odometry::SharedPtr msg) { odomCallback(msg); });
  timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / heartbeat_rate),
      [this]() { timerCallback(); });
}

double Px4ControlNode::nowSec() {
  return static_cast<double>(get_clock()->now().nanoseconds()) * 1.0e-9;
}

double Px4ControlNode::sourceAgeSec(std::uint64_t stamp_ns,
                                    double now_s) const {
  return now_s - stampNanosecondsToSeconds(stamp_ns);
}

void Px4ControlNode::logThrottledWarn(const std::string& key, double period_s,
                                      const std::string& message) {
  const double now_s = nowSec();
  const auto it = last_log_time_.find(key);
  if (it != last_log_time_.end() && now_s - it->second < period_s) return;
  last_log_time_[key] = now_s;
  RCLCPP_WARN(get_logger(), "%s", message.c_str());
}

bool Px4ControlNode::sourceStampAcceptable(
    const std::string& key_prefix, const std::string& source_name,
    std::uint64_t stamp_ns, const std::optional<std::uint64_t>& last_stamp_ns,
    double max_age_s, double now_s) {
  const double age_s = sourceAgeSec(stamp_ns, now_s);
  if (age_s < -stamp_future_tolerance_s_) {
    logThrottledWarn(key_prefix + "_future", 1.0,
                     source_name + " stamp is in the future");
    return false;
  }
  if (last_stamp_ns.has_value() && stamp_ns < *last_stamp_ns) {
    logThrottledWarn(key_prefix + "_regressed", 1.0,
                     source_name + " stamp regressed");
    return false;
  }
  if (max_age_s > 0.0 && age_s > max_age_s) {
    logThrottledWarn(key_prefix + "_stale", 1.0,
                     source_name + " source is stale");
    return false;
  }
  return true;
}

bool Px4ControlNode::topicFresh(
    const std::optional<std::uint64_t>& stamp_ns, double timeout_s,
    double now_s) const {
  if (!stamp_ns.has_value()) return false;
  const double age_s = sourceAgeSec(*stamp_ns, now_s);
  if (age_s < -stamp_future_tolerance_s_) return false;
  return age_s <= timeout_s;
}

bool Px4ControlNode::plannerCommandFrameValid(const std::string& frame_id) {
  if (frame_id.empty() || frame_id == setpoint_frame_) {
    last_warned_cmd_frame_.clear();
    return true;
  }
  if (frame_id != last_warned_cmd_frame_) {
    RCLCPP_WARN(get_logger(), "Ignoring planner command in frame '%s'; expected '%s'",
                frame_id.c_str(), setpoint_frame_.c_str());
    last_warned_cmd_frame_ = frame_id;
  }
  return false;
}

void Px4ControlNode::cmdCallback(
    const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
  if (!plannerCommandFrameValid(msg->header.frame_id)) return;
  const double now_s = nowSec();
  const std::uint64_t stamp_ns = stampToNanoseconds(msg->header.stamp);
  if (!sourceStampAcceptable("planner_cmd", "planner_cmd", stamp_ns,
                             latest_cmd_stamp_ns_, command_timeout_s_, now_s)) {
    return;
  }
  latest_cmd_ = twistStampedToControl(*msg);
  latest_cmd_stamp_ns_ = stamp_ns;
}

void Px4ControlNode::plannerArrivedCallback(
    const std_msgs::msg::Bool::SharedPtr msg) {
  planner_arrived_ = msg->data;
}

void Px4ControlNode::statusCallback(
    const mavros_msgs::msg::State::SharedPtr msg) {
  latest_status_ = *msg;
}

void Px4ControlNode::odomCallback(
    const nav_msgs::msg::Odometry::SharedPtr msg) {
  const double now_s = nowSec();
  const std::uint64_t stamp_ns = stampToNanoseconds(msg->header.stamp);
  if (!sourceStampAcceptable("odom", "odom", stamp_ns, latest_odom_stamp_ns_,
                             state_odom_timeout_s_, now_s)) {
    return;
  }
  latest_odom_ = *msg;
  latest_odom_stamp_ns_ = stamp_ns;
}

ControlInputs Px4ControlNode::buildInputs() {
  const double now_s = nowSec();
  ControlInputs inputs;
  inputs.vehicle_status_seen = latest_status_.has_value();
  inputs.odom_fresh =
      topicFresh(latest_odom_stamp_ns_, state_odom_timeout_s_, now_s);
  inputs.offboard_active =
      latest_status_.has_value() &&
      upper(latest_status_->mode) == "OFFBOARD";
  if (latest_odom_.has_value()) {
    inputs.has_altitude = true;
    inputs.altitude_m = latest_odom_->pose.pose.position.z;
  }
  inputs.planner_cmd_gap_sec = std::numeric_limits<double>::infinity();
  if (latest_cmd_stamp_ns_.has_value()) {
    inputs.planner_cmd_gap_sec =
        std::max(0.0, sourceAgeSec(*latest_cmd_stamp_ns_, now_s));
    inputs.planner_cmd_fresh =
        topicFresh(latest_cmd_stamp_ns_, command_timeout_s_, now_s);
  }
  inputs.planner_arrived = planner_arrived_;
  return inputs;
}

void Px4ControlNode::publishDebugTopic(const ControlDecision& decision) {
  std_msgs::msg::String msg;
  msg.data = std::string("phase=") + toString(decision.phase) +
             " policy=" + toString(decision.policy) +
             " reason=" + decision.reason;
  debug_pub_->publish(msg);
}

void Px4ControlNode::publishSetpoint(const neupan_uav::Control& control) {
  const auto msg =
      controlToTwistStamped(control, get_clock()->now(), setpoint_frame_);
  setpoint_pub_->publish(msg);
  applied_cmd_pub_->publish(msg);
}

neupan_uav::Control Px4ControlNode::takeoffSetpoint() const {
  neupan_uav::Control out = neupan_uav::Control::Zero();
  out(2) = std::min(takeoff_climb_speed_, takeoff_max_climb_speed_);
  return out;
}

void Px4ControlNode::timerCallback() {
  const ControlInputs inputs = buildInputs();
  const ControlDecision next =
      evaluateControlDecision(phase_, inputs, config_);
  const ControlDecision previous = decision_;
  phase_ = next.phase;
  decision_ = next;
  publishDebugTopic(next);

  if (previous != next) {
    RCLCPP_INFO(get_logger(), "Control phase=%s policy=%s reason=%s",
                toString(next.phase), toString(next.policy),
                next.reason.c_str());
  }

  if (next.policy == ControlPolicy::kZeroHold) {
    publishSetpoint(neupan_uav::Control::Zero());
    return;
  }
  if (next.policy == ControlPolicy::kTakeoffClimb) {
    publishSetpoint(takeoffSetpoint());
    return;
  }
  if (latest_cmd_.has_value()) {
    publishSetpoint(*latest_cmd_);
    return;
  }
  publishSetpoint(neupan_uav::Control::Zero());
}

}  // namespace neupan_ros

#ifndef NEUPAN_ROS_DISABLE_NODE_MAIN
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<neupan_ros::Px4ControlNode>());
  rclcpp::shutdown();
  return 0;
}
#endif
