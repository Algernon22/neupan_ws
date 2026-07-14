#include "neupan_ros/uav_node.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "std_msgs/msg/bool.hpp"

namespace {

using namespace std::chrono_literals;

std::string testConfigPath(const std::string& file_name) {
  return std::string(NEUPAN_ROS_TEST_CONFIG_DIR) + "/" + file_name;
}

std::string testDataPath(const std::string& file_name) {
  return std::string(NEUPAN_ROS_TEST_DIR) + "/" + file_name;
}

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

std::string topicPrefix(const std::string& suffix) {
  return "/neupan_ros_test_uav_" + suffix;
}

nav_msgs::msg::Odometry odomAt(rclcpp::Node& node, double x, double y,
                               double z) {
  nav_msgs::msg::Odometry msg;
  msg.header.stamp = node.get_clock()->now();
  msg.header.frame_id = "camera_init";
  msg.child_frame_id = "base_link";
  msg.pose.pose.position.x = x;
  msg.pose.pose.position.y = y;
  msg.pose.pose.position.z = z;
  msg.pose.pose.orientation.w = 1.0;
  return msg;
}

void appendFloat(std::vector<std::uint8_t>& data, float value) {
  const auto* raw = reinterpret_cast<const std::uint8_t*>(&value);
  data.insert(data.end(), raw, raw + sizeof(float));
}

sensor_msgs::msg::PointCloud2 cloudWithPoints(
    rclcpp::Node& node,
    const std::vector<std::array<float, 3>>& points) {
  sensor_msgs::msg::PointCloud2 msg;
  msg.header.stamp = node.get_clock()->now();
  msg.header.frame_id = "base_link";
  msg.height = 1;
  msg.width = static_cast<std::uint32_t>(points.size());
  msg.is_bigendian = false;
  msg.is_dense = true;
  msg.point_step = 12;
  msg.row_step = msg.point_step * msg.width;
  msg.fields.resize(3);
  msg.fields[0].name = "x";
  msg.fields[0].offset = 0;
  msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
  msg.fields[0].count = 1;
  msg.fields[1].name = "y";
  msg.fields[1].offset = 4;
  msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
  msg.fields[1].count = 1;
  msg.fields[2].name = "z";
  msg.fields[2].offset = 8;
  msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
  msg.fields[2].count = 1;
  msg.data.reserve(static_cast<std::size_t>(msg.row_step));
  for (const auto& point : points) {
    appendFloat(msg.data, point[0]);
    appendFloat(msg.data, point[1]);
    appendFloat(msg.data, point[2]);
  }
  return msg;
}

geometry_msgs::msg::TwistStamped appliedCmd(rclcpp::Node& node) {
  geometry_msgs::msg::TwistStamped msg;
  msg.header.stamp = node.get_clock()->now();
  msg.header.frame_id = "camera_init";
  msg.twist.linear.x = 0.11;
  msg.twist.linear.y = -0.02;
  msg.twist.linear.z = 0.03;
  msg.twist.angular.z = 0.04;
  return msg;
}

struct UavHarness {
  explicit UavHarness(const std::string& suffix, double update_rate,
                      double planner_rate) {
    const std::string prefix = topicPrefix(suffix);
    state_topic = prefix + "/odom";
    cloud_topic = prefix + "/cloud";
    applied_topic = prefix + "/applied";
    cmd_topic = prefix + "/cmd";
    arrived_topic = prefix + "/arrived";

    rclcpp::NodeOptions options;
    options.parameter_overrides({
        rclcpp::Parameter("robot_config_dir", testConfigPath("")),
        rclcpp::Parameter("planner_config_file",
                          testDataPath("planner_smoke.yaml")),
        rclcpp::Parameter("map_frame", "camera_init"),
        rclcpp::Parameter("state_topic", state_topic),
        rclcpp::Parameter("pointcloud_topic", cloud_topic),
        rclcpp::Parameter("applied_cmd_topic", applied_topic),
        rclcpp::Parameter("cmd_vel_topic", cmd_topic),
        rclcpp::Parameter("planner_arrived_topic", arrived_topic),
        rclcpp::Parameter("update_rate", update_rate),
        rclcpp::Parameter("planner_rate", planner_rate),
        rclcpp::Parameter("max_state_age_ms", 1000.0),
        rclcpp::Parameter("max_cloud_age_ms", 1000.0),
        rclcpp::Parameter("profile_planner", true),
    });

    uav_node = std::make_shared<neupan_ros::UavNode>(options);
    io_node = std::make_shared<rclcpp::Node>("uav_smoke_test_" + suffix);
    state_pub = io_node->create_publisher<nav_msgs::msg::Odometry>(
        state_topic, rclcpp::SensorDataQoS());
    cloud_pub = io_node->create_publisher<sensor_msgs::msg::PointCloud2>(
        cloud_topic, rclcpp::SensorDataQoS());
    applied_pub =
        io_node->create_publisher<geometry_msgs::msg::TwistStamped>(
            applied_topic, 10);
    cmd_sub =
        io_node->create_subscription<geometry_msgs::msg::TwistStamped>(
            cmd_topic, 10,
            [this](geometry_msgs::msg::TwistStamped::SharedPtr msg) {
              commands.push_back(*msg);
            });
    arrived_sub = io_node->create_subscription<std_msgs::msg::Bool>(
        arrived_topic, 10, [this](std_msgs::msg::Bool::SharedPtr msg) {
          arrived.push_back(*msg);
        });

    executor.add_node(uav_node);
    executor.add_node(io_node);
  }

  ~UavHarness() {
    executor.remove_node(io_node);
    executor.remove_node(uav_node);
    uav_node.reset();
    io_node.reset();
  }

  void publishPlanningInputs(double x, double y, double z) {
    state_pub->publish(odomAt(*io_node, x, y, z));
    cloud_pub->publish(
        cloudWithPoints(*io_node, {{{6.0F, 4.0F, 2.0F},
                                   {7.0F, -3.0F, 2.0F}}}));
    applied_pub->publish(appliedCmd(*io_node));
  }

  std::string state_topic;
  std::string cloud_topic;
  std::string applied_topic;
  std::string cmd_topic;
  std::string arrived_topic;
  std::shared_ptr<neupan_ros::UavNode> uav_node;
  std::shared_ptr<rclcpp::Node> io_node;
  rclcpp::executors::MultiThreadedExecutor executor;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr state_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr applied_pub;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_sub;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr arrived_sub;
  std::vector<geometry_msgs::msg::TwistStamped> commands;
  std::vector<std_msgs::msg::Bool> arrived;
};

}  // namespace

TEST_F(RosFixture, PlannerNodePublishesCommandAndArrivedFromFakeInputs) {
  UavHarness harness("planner", 20.0, 10.0);

  const bool ok = spinUntil(
      harness.executor, 4s,
      [&]() { return !harness.commands.empty() && !harness.arrived.empty(); },
      [&]() { harness.publishPlanningInputs(0.0, 0.0, 2.0); });

  ASSERT_TRUE(ok);
  const auto& cmd = harness.commands.back();
  EXPECT_EQ(cmd.header.frame_id, "camera_init");
  EXPECT_TRUE(std::isfinite(cmd.twist.linear.x));
  EXPECT_TRUE(std::isfinite(cmd.twist.linear.y));
  EXPECT_TRUE(std::isfinite(cmd.twist.linear.z));
  EXPECT_TRUE(std::isfinite(cmd.twist.angular.z));
  EXPECT_FALSE(harness.arrived.back().data);
}

TEST_F(RosFixture, PlannerNodeProfileLogIncludesFarfieldFields) {
  UavHarness harness("profile", 20.0, 10.0);

  const bool ok = spinUntil(
      harness.executor, 4s,
      [&]() {
        const std::string log = harness.uav_node->lastProfileLogForTest();
        return log.find("Planner profile") != std::string::npos &&
               log.find("farfield=") != std::string::npos &&
               log.find("off=") != std::string::npos &&
               log.find("cnt=") != std::string::npos;
      },
      [&]() { harness.publishPlanningInputs(0.0, 0.0, 2.0); });

  ASSERT_TRUE(ok);
}

TEST_F(RosFixture, PlannerNodePublishesArrivedWhenFakeOdomStartsAtGoal) {
  UavHarness harness("arrived", 20.0, 10.0);

  const bool ok = spinUntil(
      harness.executor, 4s,
      [&]() {
        return !harness.commands.empty() && !harness.arrived.empty() &&
               harness.arrived.back().data;
      },
      [&]() { harness.publishPlanningInputs(20.0, 0.0, 2.0); });

  ASSERT_TRUE(ok);
  const auto& cmd = harness.commands.back();
  EXPECT_EQ(cmd.header.frame_id, "camera_init");
  EXPECT_NEAR(cmd.twist.linear.x, 0.0, 1.0e-9);
  EXPECT_NEAR(cmd.twist.linear.y, 0.0, 1.0e-9);
  EXPECT_NEAR(cmd.twist.linear.z, 0.0, 1.0e-9);
  EXPECT_NEAR(cmd.twist.angular.z, 0.0, 1.0e-9);
}
