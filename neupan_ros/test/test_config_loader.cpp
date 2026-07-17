#include "neupan_ros/config_loader.hpp"

#include "neupan_uav/planner.hpp"

#include <gtest/gtest.h>

namespace {

std::string plannerConfigPath() {
  return std::string(NEUPAN_ROS_TEST_CONFIG_DIR) + "/planner.yaml";
}

}  // namespace

TEST(ConfigLoader, LoadsCurrentPlannerYamlShape) {
  const auto loaded = neupan_ros::loadPlannerConfig(plannerConfigPath());
  const auto& config = loaded.options;

  EXPECT_EQ(config.grid.horizon_steps, 25);
  EXPECT_NEAR(config.grid.dt, 0.1, 1e-12);
  EXPECT_NEAR(config.ref_speed, 1.0, 1e-12);
  EXPECT_NEAR(config.collision_threshold, 0.15, 1e-12);
  EXPECT_NEAR(config.arrive_threshold, 0.8, 1e-12);
  EXPECT_TRUE(config.has_goal);
  ASSERT_EQ(config.initial_path.waypoints.size(), 2U);
  EXPECT_NEAR(config.initial_path.waypoints.back()(0), 5.0, 1e-12);
  EXPECT_NEAR(config.robot.body_half_extent(0), 0.32, 1e-12);
  EXPECT_NEAR(config.robot.body_half_extent(1), 0.32, 1e-12);
  EXPECT_NEAR(config.robot.body_half_extent(2), 0.27, 1e-12);
  EXPECT_NEAR(config.robot.max_control(0), 1.0, 1e-12);
  EXPECT_NEAR(config.robot.max_control(2), 1.0, 1e-12);
  EXPECT_TRUE(config.preselect.enabled);
  EXPECT_EQ(config.preselect.per_step, 4);
  EXPECT_NEAR(config.preselect.corridor_margin(0), 6.0, 1e-12);
  EXPECT_NEAR(config.preselect.exact_margin(2), 0.25, 1e-12);
  EXPECT_EQ(config.pan.iter_num, 1);
  EXPECT_NEAR(config.pan.trajectory_threshold, 0.05, 1e-12);
  EXPECT_NEAR(config.pan.dune_threshold, 0.01, 1e-12);
  ASSERT_TRUE(config.dune.has_value());
  EXPECT_TRUE(config.dune->metadata_path.empty());
  EXPECT_NEAR(config.dune->select_nearest_ratio, 0.60, 1e-12);
  EXPECT_NEAR(config.dune->select_temporal_ratio, 0.20, 1e-12);
  EXPECT_NEAR(config.dune->select_diversity_ratio, 0.20, 1e-12);
  EXPECT_EQ(config.nrmp.max_constraints, 20);
  EXPECT_NEAR(config.pan.p_u, 1.5, 1e-12);
  EXPECT_NEAR(config.pan.eta, 6.0, 1e-12);
  EXPECT_NEAR(config.pan.d_min, 0.8, 1e-12);
  EXPECT_NEAR(config.pan.d_max, 4.0, 1e-12);
  EXPECT_NEAR(config.pan.ro_obs, 260.0, 1e-12);
  EXPECT_NEAR(config.pan.bk, 0.18, 1e-12);
  EXPECT_NEAR(config.pan.smooth_du, 0.8, 1e-12);
  EXPECT_NEAR(config.pan.smooth_u0, 6.0, 1e-12);
  EXPECT_TRUE(config.nrmp.enable_control_smoothing);
  EXPECT_EQ(config.nrmp.solver.max_iter, 1000);
  EXPECT_FALSE(config.nrmp.solver.verbose);
  EXPECT_FALSE(config.nrmp.solver.polishing);
  EXPECT_TRUE(config.nrmp.solver.warm_starting);
  EXPECT_TRUE(config.farfield_guide.enabled);
  EXPECT_NEAR(config.farfield_guide.range_backoff, 1.0, 1e-12);
  EXPECT_NEAR(config.farfield_guide.range_scale, 1.5, 1e-12);
  EXPECT_NEAR(config.farfield_guide.lateral_width, 5.0, 1e-12);
  EXPECT_NEAR(config.farfield_guide.center_width, 2.0, 1e-12);
  EXPECT_NEAR(config.farfield_guide.height_window, 1.5, 1e-12);
  EXPECT_NEAR(config.farfield_guide.voxel_size(0), 1.0, 1e-12);
  EXPECT_NEAR(config.farfield_guide.voxel_size(2), 0.8, 1e-12);
  EXPECT_EQ(config.farfield_guide.trigger_count, 14);
  EXPECT_NEAR(config.farfield_guide.offset_min, 1.5, 1e-12);
  EXPECT_NEAR(config.farfield_guide.offset_max, 4.0, 1e-12);
  EXPECT_NEAR(config.farfield_guide.offset_speed_gain, 0.60, 1e-12);
  EXPECT_NEAR(config.farfield_guide.offset_alpha, 0.20, 1e-12);
  EXPECT_NEAR(config.farfield_guide.release_alpha, 0.08, 1e-12);
  EXPECT_EQ(config.farfield_guide.release_count, 4);
  EXPECT_EQ(config.farfield_guide.release_confirm_cycles, 3);
}

TEST(ConfigLoader, LoadedConfigConstructsPlanner) {
  const auto loaded = neupan_ros::loadPlannerConfig(plannerConfigPath());

  try {
    const neupan_uav::CompiledPlannerConfig config =
        neupan_ros::compileLoadedPlannerConfig(loaded);
    EXPECT_FALSE(config.pan().dune.has_value());
    EXPECT_EQ(config.receding(), 25);
    EXPECT_EQ(config.pan().nrmp.max_num, 20);
    neupan_uav::Planner planner(config);
    neupan_uav::PlannerInput input;
    input.state.position_world << 0.0, 0.0, 2.0;
    input.obstacle_points = neupan_uav::emptyPointMatrix();
    const neupan_uav::PlannerResult output = planner.forward(input);
    EXPECT_TRUE(output.isTracking());
  } catch (const std::exception& exc) {
    FAIL() << exc.what();
  }
}
