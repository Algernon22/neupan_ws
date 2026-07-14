#include "neupan_ros/uav_node.hpp"

#include "internal_interfaces.hpp"
#include "neupan_ros/adapters.hpp"
#include "neupan_uav/rknn_runner.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include "rclcpp/rclcpp.hpp"

namespace neupan_ros {

UavNode::UavNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("neupan_uav_node", options) {
  declare_parameter<std::string>("robot_config_dir", "");
  declare_parameter<std::string>("planner_config_file", "planner.yaml");
  declare_parameter<std::string>("dune_rknn_metadata_file", "");
  declare_parameter<std::string>("dune_rknn_core_mask", "CORE_0_1");
  declare_parameter<bool>("dune_rknn_require_device", true);
  declare_parameter<std::string>("command_frame", internal::kDefaultCommandFrame);
  declare_parameter<std::string>("state_topic", "/Odometry");
  declare_parameter<std::string>("pointcloud_topic", "/cloud_registered_body");
  declare_parameter<double>("update_rate", 20.0);
  declare_parameter<double>("planner_rate", 10.0);
  declare_parameter<double>("max_state_age_ms", 150.0);
  declare_parameter<double>("max_cloud_age_ms", 250.0);
  declare_parameter<double>("roi_auto_xy_padding", 3.0);
  declare_parameter<double>("roi_auto_xy_max", 40.0);
  declare_parameter<double>("roi_auto_z_min", 4.0);
  declare_parameter<std::vector<double>>("self_filter_margin_xyz",
                                         {0.05, 0.05, 0.05});
  declare_parameter<bool>("profile_planner", false);

  const std::string config_dir = get_parameter("robot_config_dir").as_string();
  const std::string planner_file =
      get_parameter("planner_config_file").as_string();
  const std::string planner_path = resolvePath(config_dir, planner_file);
  loaded_config_ = loadPlannerConfig(planner_path);

  command_frame_ = get_parameter("command_frame").as_string();
  state_topic_ = get_parameter("state_topic").as_string();
  pointcloud_topic_ = get_parameter("pointcloud_topic").as_string();
  update_rate_ = std::max(1.0, get_parameter("update_rate").as_double());
  planner_rate_ = std::max(1.0, get_parameter("planner_rate").as_double());
  max_state_age_s_ =
      std::max(0.0, 1.0e-3 * get_parameter("max_state_age_ms").as_double());
  max_cloud_age_s_ =
      std::max(0.0, 1.0e-3 * get_parameter("max_cloud_age_ms").as_double());
  body_half_extent_ = loaded_config_.planner.robot.body_half_extent;
  const std::vector<double> margin =
      get_parameter("self_filter_margin_xyz").as_double_array();
  if (margin.size() == 1) {
    self_filter_margin_xyz_ = Eigen::Vector3d::Constant(std::max(0.0, margin[0]));
  } else if (margin.size() == 3) {
    self_filter_margin_xyz_ =
        Eigen::Vector3d(std::max(0.0, margin[0]), std::max(0.0, margin[1]),
                        std::max(0.0, margin[2]));
  } else {
    throw std::runtime_error("self_filter_margin_xyz must be scalar or length 3");
  }
  profile_planner_ = get_parameter("profile_planner").as_bool();

  const double horizon =
      std::max(0.0, loaded_config_.planner.ref_speed) *
      std::max(0.0, loaded_config_.planner.step_time) *
      std::max(0, loaded_config_.planner.receding);
  const double roi_padding =
      std::max(0.0, get_parameter("roi_auto_xy_padding").as_double());
  const double roi_xy_max =
      std::max(0.0, get_parameter("roi_auto_xy_max").as_double());
  const double roi_z_min =
      std::max(0.0, get_parameter("roi_auto_z_min").as_double());
  if (horizon > 0.0 || roi_z_min > 0.0) {
    double required_xy = horizon;
    if (loaded_config_.planner.farfield_guide.enabled) {
      const auto& farfield = loaded_config_.planner.farfield_guide;
      const double farfield_far =
          std::max(0.0, horizon - std::max(0.0, farfield.range_backoff)) *
          std::max(1.0, farfield.range_scale);
      required_xy =
          std::max(required_xy, farfield_far) +
          std::max(0.0, farfield.lateral_width);
    }
    double xy = required_xy + roi_padding;
    if (roi_xy_max > 0.0) xy = std::min(xy, roi_xy_max);
    roi_ = Eigen::Vector3d(xy, xy, roi_z_min);
    if (loaded_config_.planner.farfield_guide.enabled && xy > 0.0) {
      loaded_config_.planner.farfield_guide.range_far_limit = xy;
    }
  }

  const std::string rknn_metadata =
      get_parameter("dune_rknn_metadata_file").as_string();
  const std::string rknn_core_mask =
      get_parameter("dune_rknn_core_mask").as_string();
  if (!rknn_metadata.empty()) {
    loaded_config_.planner.pan.rknn_mode = neupan_uav::RknnRunnerMode::kRuntime;
    loaded_config_.planner.pan.rknn_metadata_path =
        resolvePath(config_dir, rknn_metadata);
    loaded_config_.planner.pan.rknn_core_mask = rknn_core_mask;
    loaded_config_.planner.pan.rknn_require_device =
        get_parameter("dune_rknn_require_device").as_bool();
  }

  planner_ = std::make_unique<neupan_uav::Planner>(loaded_config_.planner);

  RCLCPP_INFO(get_logger(), "Loaded C++ NeuPAN planner config: %s",
              planner_path.c_str());

  cmd_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>(
      internal::kPlannerCommandTopic, 10);
  arrived_pub_ =
      create_publisher<std_msgs::msg::Bool>(internal::kPlannerArrivedTopic, 10);
  state_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      state_topic_, rclcpp::SensorDataQoS(),
      [this](nav_msgs::msg::Odometry::SharedPtr msg) { stateCallback(msg); });
  cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      pointcloud_topic_, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        cloudCallback(msg);
      });
  publish_timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / update_rate_),
      [this]() { publishLoop(); });

  planner_thread_ = std::thread([this]() { plannerWorkerMain(); });
  RCLCPP_INFO(get_logger(), "neupan_ros C++ planner node initialized");
}

UavNode::~UavNode() {
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    stop_worker_ = true;
  }
  if (planner_thread_.joinable()) planner_thread_.join();
}

std::string UavNode::resolvePath(const std::string& base_dir,
                                 const std::string& maybe_relative) const {
  std::filesystem::path path(maybe_relative);
  if (path.is_absolute()) return path.string();
  if (base_dir.empty()) return path.string();
  return (std::filesystem::path(base_dir) / path).string();
}

double UavNode::nowSec() {
  return static_cast<double>(get_clock()->now().nanoseconds()) * 1.0e-9;
}

std::uint64_t UavNode::nowNanoseconds() {
  return static_cast<std::uint64_t>(get_clock()->now().nanoseconds());
}

void UavNode::logThrottledWarn(const std::string& key, double period_s,
                               const std::string& message) {
  const double now_s = nowSec();
  const auto it = last_log_time_.find(key);
  if (it != last_log_time_.end() && now_s - it->second < period_s) return;
  last_log_time_[key] = now_s;
  RCLCPP_WARN(get_logger(), "%s", message.c_str());
}

std::optional<UavNode::LatestCloud> UavNode::processCloud(
    const sensor_msgs::msg::PointCloud2& msg) {
  auto parsed = readXyzPoints(msg);
  if (!parsed.has_value()) return std::nullopt;

  neupan_uav::PointMatrix points = std::move(*parsed);
  const std::uint64_t stamp_ns = stampToNanoseconds(msg.header.stamp);
  LatestCloud cloud;
  cloud.stamp_ns = stamp_ns;
  cloud.receive_time_s = nowSec();

  if (points.cols() == 0) return cloud;
  if (roi_.has_value()) {
    std::vector<Eigen::Vector3d> kept;
    kept.reserve(static_cast<std::size_t>(points.cols()));
    for (Eigen::Index col = 0; col < points.cols(); ++col) {
      const Eigen::Vector3d p = points.col(col);
      if (std::abs(p(0)) <= (*roi_)(0) && std::abs(p(1)) <= (*roi_)(1) &&
          std::abs(p(2)) <= (*roi_)(2)) {
        kept.push_back(p);
      }
    }
    neupan_uav::PointMatrix filtered(3, static_cast<Eigen::Index>(kept.size()));
    for (Eigen::Index i = 0; i < filtered.cols(); ++i) {
      filtered.col(i) = kept[static_cast<std::size_t>(i)];
    }
    points = filtered;
  }
  if (points.cols() == 0) return cloud;

  const Eigen::Vector3d inflated = body_half_extent_ + self_filter_margin_xyz_;
  std::vector<Eigen::Vector3d> kept;
  kept.reserve(static_cast<std::size_t>(points.cols()));
  for (Eigen::Index col = 0; col < points.cols(); ++col) {
    const Eigen::Vector3d abs_p = points.col(col).cwiseAbs();
    if (abs_p(0) > inflated(0) || abs_p(1) > inflated(1) ||
        abs_p(2) > inflated(2)) {
      kept.push_back(points.col(col));
    }
  }
  cloud.points_body.resize(3, static_cast<Eigen::Index>(kept.size()));
  for (Eigen::Index i = 0; i < cloud.points_body.cols(); ++i) {
    cloud.points_body.col(i) = kept[static_cast<std::size_t>(i)];
  }
  cloud.min_body_clearance =
      minBodyClearance(cloud.points_body, body_half_extent_);
  return cloud;
}

void UavNode::stateCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
  const std::uint64_t stamp_ns = stampToNanoseconds(msg->header.stamp);
  const OdomStates states = odometryToStates(*msg);
  Eigen::VectorXd planner_state(6);
  planner_state = states.state6;
  LatestState latest;
  latest.stamp_ns = stamp_ns;
  latest.state6 = states.state6;
  latest.planner_state = planner_state;
  latest.receive_time_s = nowSec();
  std::lock_guard<std::mutex> lock(data_mutex_);
  if (latest_state_.has_value() && stamp_ns <= latest_state_->stamp_ns) {
    return;
  }
  latest_state_ = latest;
}

void UavNode::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
  auto cloud = processCloud(*msg);
  if (!cloud.has_value()) {
    logThrottledWarn("invalid_pointcloud", 1.0,
                     "Rejected malformed PointCloud2 message");
    return;
  }
  std::lock_guard<std::mutex> lock(data_mutex_);
  if (latest_cloud_.has_value() && cloud->stamp_ns <= latest_cloud_->stamp_ns) {
    return;
  }
  latest_cloud_ = std::move(*cloud);
}

std::optional<UavNode::PlannerJob> UavNode::snapshotPlannerInputs() {
  const double now_s = nowSec();
  std::lock_guard<std::mutex> lock(data_mutex_);
  if (stop_worker_) return std::nullopt;
  if (!latest_state_.has_value()) {
    latest_result_.ready = false;
    latest_result_.reason = "waiting_for_state";
    return std::nullopt;
  }
  if (!latest_cloud_.has_value()) {
    latest_result_.ready = false;
    latest_result_.reason = "waiting_for_cloud";
    return std::nullopt;
  }

  const double state_age = std::max(0.0, now_s - latest_state_->receive_time_s);
  const double cloud_age = std::max(0.0, now_s - latest_cloud_->receive_time_s);
  if (max_state_age_s_ > 0.0 && state_age > max_state_age_s_) {
    latest_result_.ready = false;
    latest_result_.reason = "stale_state";
    return std::nullopt;
  }
  if (max_cloud_age_s_ > 0.0 && cloud_age > max_cloud_age_s_) {
    latest_result_.ready = false;
    latest_result_.reason = "stale_cloud";
    return std::nullopt;
  }

  PlannerJob job;
  job.state = *latest_state_;
  job.cloud = *latest_cloud_;
  return job;
}

UavNode::PlannerResult UavNode::runPlannerOnce(const PlannerJob& job) {
  PlannerResult result;

  neupan_uav::PlannerInput input;
  input.state = job.state.planner_state;
  input.obstacle_points =
      pointsBodyToWorld(job.cloud.points_body, job.state.state6);
  input.stamp_sec = stampNanosecondsToSeconds(job.state.stamp_ns);
  const neupan_uav::PlannerOutput out = planner_->forward(input);

  result.ready = out.ready;
  result.reason = out.reason;
  result.generated_stamp_ns = nowNanoseconds();

  if (out.ready) {
    result.command = out.command;
  }

  if (profile_planner_) {
    char profile_log[512];
    std::snprintf(profile_log, sizeof(profile_log),
                  "Planner profile total=%.1fms pre=%.1fms dune=%.1fms "
                  "nrmp=%.1fms farfield=%d off=%.2f/%.2f cnt=%d/%d/%d "
                  "obs=%zu->%zu->%zu osqp_status=%d iter=%d",
                  1000.0 * out.profile.forward_sec,
                  1000.0 * out.profile.preselect_sec,
                  1000.0 * out.profile.dune_sec,
                  1000.0 * out.profile.nrmp_sec,
                  out.profile.farfield_active ? 1 : 0,
                  out.profile.farfield_offset_m,
                  out.profile.farfield_target_offset_m,
                  out.profile.farfield_center_count,
                  out.profile.farfield_left_count,
                  out.profile.farfield_right_count,
                  out.profile.input_obstacle_count,
                  out.profile.preselected_obstacle_count,
                  out.profile.dune_selected_count, out.profile.osqp_status,
                  out.profile.osqp_iteration_count);
    RCLCPP_INFO(get_logger(),
                "%s", profile_log);
  }
  return result;
}

void UavNode::storeLatestResult(const PlannerResult& result) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  latest_result_ = result;
}

void UavNode::plannerWorkerMain() {
  const auto period = std::chrono::duration<double>(1.0 / planner_rate_);
  auto next_tick = std::chrono::steady_clock::now();
  while (true) {
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (stop_worker_) break;
    }
    std::this_thread::sleep_until(next_tick);
    next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
    auto job = snapshotPlannerInputs();
    if (!job.has_value()) continue;
    try {
      storeLatestResult(runPlannerOnce(*job));
    } catch (const std::exception& exc) {
      logThrottledWarn("planner_forward", 1.0,
                       std::string("Planner forward failed: ") + exc.what());
      PlannerResult result;
      result.ready = false;
      result.reason = "planner_error";
      storeLatestResult(result);
    }
  }
}

void UavNode::publishLoop() {
  PlannerResult result;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    result = latest_result_;
  }
  if (!result.ready || !result.command.has_value()) return;
  const rclcpp::Time stamp(static_cast<int64_t>(result.generated_stamp_ns));
  const auto cmd =
      controlToTwistStamped(*result.command, stamp, command_frame_);
  cmd_pub_->publish(cmd);

  std_msgs::msg::Bool arrived;
  arrived.data = result.reason == "arrived";
  arrived_pub_->publish(arrived);
}

}  // namespace neupan_ros

#ifndef NEUPAN_ROS_DISABLE_NODE_MAIN
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<neupan_ros::UavNode>());
  rclcpp::shutdown();
  return 0;
}
#endif
