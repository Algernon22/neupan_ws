#include "neupan_ros/px4_control.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <string>
#include <thread>
#include <vector>

#include "geometry_msgs/msg/twist_stamped.hpp"
#include "mavros_msgs/msg/state.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/string.hpp"

namespace {

using namespace std::chrono_literals;

class RosFixture : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!rclcpp::ok()) {
      int argc = 0;
      char** argv = nullptr;
      rclcpp::init(argc, argv);
    }
  }

  static void TearDownTestSuite() {
    if (rclcpp::ok()) rclcpp::shutdown();
  }
};

std::string topicPrefix(const std::string& suffix) {
  return "/neupan_ros_test_px4_" + suffix;
}

template <typename Predicate, typename Pump>
bool spinUntil(rclcpp::Executor& executor, std::chrono::milliseconds timeout,
               Predicate predicate, Pump pump) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    pump();
    executor.spin_some(20ms);
    if (predicate()) return true;
    std::this_thread::sleep_for(10ms);
  }
  executor.spin_some(100ms);
  return predicate();
}

geometry_msgs::msg::TwistStamped plannerCmd(
    rclcpp::Node& node, double vx, double vy, double vz,
    double yaw_rate, const std::string& frame_id = "camera_init") {
  geometry_msgs::msg::TwistStamped msg;
  msg.header.stamp = node.get_clock()->now();
  msg.header.frame_id = frame_id;
  msg.twist.linear.x = vx;
  msg.twist.linear.y = vy;
  msg.twist.linear.z = vz;
  msg.twist.angular.z = yaw_rate;
  return msg;
}

nav_msgs::msg::Odometry odomAt(rclcpp::Node& node, double altitude_m) {
  nav_msgs::msg::Odometry msg;
  msg.header.stamp = node.get_clock()->now();
  msg.pose.pose.position.z = altitude_m;
  msg.pose.pose.orientation.w = 1.0;
  return msg;
}

bool twistNear(const geometry_msgs::msg::TwistStamped& msg, double vx,
               double vy, double vz, double yaw_rate) {
  constexpr double kTol = 1.0e-9;
  return std::abs(msg.twist.linear.x - vx) <= kTol &&
         std::abs(msg.twist.linear.y - vy) <= kTol &&
         std::abs(msg.twist.linear.z - vz) <= kTol &&
         std::abs(msg.twist.angular.z - yaw_rate) <= kTol;
}

struct Px4Harness {
  explicit Px4Harness(const std::string& suffix, bool enable_takeoff) {
    const std::string prefix = topicPrefix(suffix);
    command_topic = prefix + "/planner_cmd";
    state_topic = prefix + "/state";
    odom_topic = prefix + "/odom";
    setpoint_topic = prefix + "/setpoint";
    applied_topic = prefix + "/applied";
    arrived_topic = prefix + "/arrived";
    debug_topic = prefix + "/debug";

    rclcpp::NodeOptions options;
    options.arguments({
        "--ros-args",
        "-r",
        "neupan/planner/cmd_vel:=" + command_topic,
        "-r",
        "neupan/planner/arrived:=" + arrived_topic,
        "-r",
        "neupan/control/applied_cmd_vel:=" + applied_topic,
    });
    options.parameter_overrides({
        rclcpp::Parameter("mavros_state_topic", state_topic),
        rclcpp::Parameter("state_odom_topic", odom_topic),
        rclcpp::Parameter("mavros_setpoint_velocity_topic", setpoint_topic),
        rclcpp::Parameter("command_frame", "camera_init"),
        rclcpp::Parameter("control_debug_topic", debug_topic),
        rclcpp::Parameter("heartbeat_rate", 40.0),
        rclcpp::Parameter("command_timeout", 1.0),
        rclcpp::Parameter("state_odom_timeout", 1.0),
        rclcpp::Parameter("enable_takeoff_phase", enable_takeoff),
        rclcpp::Parameter("takeoff_phase_height", 1.8),
        rclcpp::Parameter("takeoff_phase_hysteresis", 0.1),
        rclcpp::Parameter("takeoff_phase_climb_speed", 1.0),
        rclcpp::Parameter("takeoff_phase_max_climb_speed", 2.0),
    });

    px4_node = std::make_shared<neupan_ros::Px4ControlNode>(options);
    io_node = std::make_shared<rclcpp::Node>("px4_topic_test_" + suffix);

    state_pub = io_node->create_publisher<mavros_msgs::msg::State>(
        state_topic, rclcpp::SensorDataQoS());
    odom_pub = io_node->create_publisher<nav_msgs::msg::Odometry>(
        odom_topic, rclcpp::SensorDataQoS());
    cmd_pub =
        io_node->create_publisher<geometry_msgs::msg::TwistStamped>(
            command_topic, 10);
    arrived_pub =
        io_node->create_publisher<std_msgs::msg::Bool>(arrived_topic, 10);

    setpoint_sub =
        io_node->create_subscription<geometry_msgs::msg::TwistStamped>(
            setpoint_topic, 10,
            [this](geometry_msgs::msg::TwistStamped::SharedPtr msg) {
              setpoints.push_back(*msg);
            });
    applied_sub =
        io_node->create_subscription<geometry_msgs::msg::TwistStamped>(
            applied_topic, 10,
            [this](geometry_msgs::msg::TwistStamped::SharedPtr msg) {
              applied.push_back(*msg);
            });
    debug_sub = io_node->create_subscription<std_msgs::msg::String>(
        debug_topic, 10, [this](std_msgs::msg::String::SharedPtr msg) {
          debug.push_back(msg->data);
        });

    executor.add_node(px4_node);
    executor.add_node(io_node);
  }

  void publishReady(double altitude_m, const std::string& mode = "OFFBOARD") {
    mavros_msgs::msg::State state;
    state.mode = mode;
    state_pub->publish(state);
    odom_pub->publish(odomAt(*io_node, altitude_m));
  }

  std::string command_topic;
  std::string state_topic;
  std::string odom_topic;
  std::string setpoint_topic;
  std::string applied_topic;
  std::string arrived_topic;
  std::string debug_topic;
  std::shared_ptr<neupan_ros::Px4ControlNode> px4_node;
  std::shared_ptr<rclcpp::Node> io_node;
  rclcpp::executors::SingleThreadedExecutor executor;
  rclcpp::Publisher<mavros_msgs::msg::State>::SharedPtr state_pub;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_pub;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr arrived_pub;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr setpoint_sub;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr applied_sub;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr debug_sub;
  std::vector<geometry_msgs::msg::TwistStamped> setpoints;
  std::vector<geometry_msgs::msg::TwistStamped> applied;
  std::vector<std::string> debug;
};

bool latestDebugContains(const Px4Harness& harness,
                         const std::string& needle) {
  return !harness.debug.empty() &&
         harness.debug.back().find(needle) != std::string::npos;
}

bool latestSetpointAndAppliedNear(const Px4Harness& harness, double vx,
                                  double vy, double vz, double yaw_rate) {
  if (harness.setpoints.empty() || harness.applied.empty()) return false;
  return twistNear(harness.setpoints.back(), vx, vy, vz, yaw_rate) &&
         twistNear(harness.applied.back(), vx, vy, vz, yaw_rate) &&
         harness.setpoints.back().header.frame_id == "camera_init" &&
         harness.applied.back().header.frame_id == "camera_init";
}

}  // namespace

TEST_F(RosFixture, HoldPublishesZeroSetpointAndAppliedFeedback) {
  Px4Harness harness("hold", false);

  const bool ok = spinUntil(
      harness.executor, 2s,
      [&]() {
        return latestDebugContains(harness, "reason=planner_cmd_stale") &&
               latestSetpointAndAppliedNear(harness, 0.0, 0.0, 0.0, 0.0);
      },
      [&]() { harness.publishReady(2.2); });

  ASSERT_TRUE(ok);
}

TEST_F(RosFixture, TakeoffPublishesClimbSetpointAndAppliedFeedback) {
  Px4Harness harness("takeoff", true);

  const bool ok = spinUntil(
      harness.executor, 2s,
      [&]() {
        return latestDebugContains(harness, "reason=takeoff_phase") &&
               latestSetpointAndAppliedNear(harness, 0.0, 0.0, 1.0, 0.0);
      },
      [&]() { harness.publishReady(1.0); });

  ASSERT_TRUE(ok);
}

TEST_F(RosFixture, PlannerCommandIsForwardedToSetpointAndAppliedFeedback) {
  Px4Harness harness("planner_cmd", false);

  const bool ok = spinUntil(
      harness.executor, 2s,
      [&]() {
        return latestDebugContains(harness, "reason=following_planner") &&
               latestSetpointAndAppliedNear(harness, 0.2, -0.1, 0.3, 0.4);
      },
      [&]() {
        harness.publishReady(2.2);
        harness.cmd_pub->publish(
            plannerCmd(*harness.io_node, 0.2, -0.1, 0.3, 0.4));
      });

  ASSERT_TRUE(ok);
}

TEST_F(RosFixture, PlannerArrivedSwitchesToZeroAppliedFeedback) {
  Px4Harness harness("arrived", false);

  const bool followed = spinUntil(
      harness.executor, 2s,
      [&]() {
        return latestDebugContains(harness, "reason=following_planner") &&
               latestSetpointAndAppliedNear(harness, 0.2, -0.1, 0.3, 0.4);
      },
      [&]() {
        harness.publishReady(2.2);
        harness.cmd_pub->publish(
            plannerCmd(*harness.io_node, 0.2, -0.1, 0.3, 0.4));
      });
  ASSERT_TRUE(followed);

  std_msgs::msg::Bool arrived;
  arrived.data = true;
  const bool stopped = spinUntil(
      harness.executor, 2s,
      [&]() {
        return latestDebugContains(harness, "reason=mission_finished") &&
               latestSetpointAndAppliedNear(harness, 0.0, 0.0, 0.0, 0.0);
      },
      [&]() {
        harness.publishReady(2.2);
        harness.cmd_pub->publish(
            plannerCmd(*harness.io_node, 0.2, -0.1, 0.3, 0.4));
        harness.arrived_pub->publish(arrived);
      });

  ASSERT_TRUE(stopped);
}
