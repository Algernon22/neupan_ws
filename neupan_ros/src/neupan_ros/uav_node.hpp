#pragma once

#include "neupan_ros/config_loader.hpp"
#include "neupan_uav/types.hpp"
#include "neupan_uav/planner.hpp"

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>

#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/bool.hpp"

namespace neupan_ros {

class UavNode final : public rclcpp::Node {
 public:
  explicit UavNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~UavNode() override;
  std::string lastProfileLogForTest() const { return last_profile_log_; }

 private:
  struct LatestState {
    std::uint64_t stamp_ns = 0;
    Eigen::Matrix<double, 6, 1> state6 =
        Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::VectorXd planner_state;
    double receive_time_s = 0.0;
  };

  struct LatestCloud {
    std::uint64_t stamp_ns = 0;
    neupan_uav::PointMatrix points_body = neupan_uav::emptyPointMatrix();
    double min_body_clearance = std::numeric_limits<double>::infinity();
    double receive_time_s = 0.0;
  };

  struct PlannerJob {
    LatestState state;
    LatestCloud cloud;
    std::optional<neupan_uav::Control> applied_control;
  };

  struct PlannerResult {
    std::optional<neupan_uav::Control> command;
    bool ready = false;
    std::string reason = "waiting_for_state";
    std::uint64_t generated_stamp_ns = 0;
  };

  std::string resolvePath(const std::string& base_dir,
                          const std::string& maybe_relative) const;
  double nowSec();
  std::uint64_t nowNanoseconds();
  void logThrottledWarn(const std::string& key, double period_s,
                        const std::string& message);
  std::optional<LatestCloud> processCloud(
      const sensor_msgs::msg::PointCloud2& msg,
      const std::optional<double>& altitude_m);
  bool isTakeoffPhaseActive(double altitude_m);
  std::optional<PlannerJob> snapshotPlannerInputs();
  PlannerResult runPlannerOnce(const PlannerJob& job);
  void plannerWorkerMain();
  void publishLoop();
  void stateCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void appliedCmdCallback(
      const geometry_msgs::msg::TwistStamped::SharedPtr msg);
  void storeLatestResult(const PlannerResult& result);

  std::unique_ptr<neupan_uav::Planner> planner_;
  LoadedPlannerConfig loaded_config_;

  std::string map_frame_ = "camera_init";
  std::string state_topic_ = "/Odometry";
  std::string pointcloud_topic_ = "/cloud_registered_body";
  std::string cmd_vel_topic_ = "/neupan/planner/cmd_vel";
  std::string applied_cmd_topic_ = "/neupan/control/applied_cmd_vel";
  std::string planner_arrived_topic_ = "/neupan/planner/arrived";
  double update_rate_ = 20.0;
  double planner_rate_ = 10.0;
  double max_state_age_s_ = 0.15;
  double max_cloud_age_s_ = 0.25;
  std::optional<Eigen::Vector3d> roi_;
  Eigen::Vector3d body_half_extent_ = Eigen::Vector3d(0.23, 0.23, 0.06);
  Eigen::Vector3d self_filter_margin_xyz_ =
      Eigen::Vector3d(0.05, 0.05, 0.05);
  double takeoff_ground_clip_ = -0.02;
  bool enable_takeoff_phase_ = true;
  double takeoff_phase_release_height_ = 1.9;
  bool takeoff_phase_done_ = false;
  bool profile_planner_ = false;
  std::string last_profile_log_;

  std::mutex data_mutex_;
  std::optional<LatestState> latest_state_;
  std::optional<LatestCloud> latest_cloud_;
  std::optional<neupan_uav::Control> latest_applied_cmd_;
  PlannerResult latest_result_;
  bool stop_worker_ = false;
  std::thread planner_thread_;
  std::unordered_map<std::string, double> last_log_time_;

  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr arrived_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr state_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr applied_sub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
};

}  // namespace neupan_ros
