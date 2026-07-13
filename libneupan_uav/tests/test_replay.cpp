#include "neupan_uav/planner.hpp"
#include "neupan_uav/rknn_runner.hpp"

#include <gtest/gtest.h>

#include <numeric>

namespace {

neupan_uav::NrmpConfig basicNrmpConfig(bool with_obstacles) {
  neupan_uav::NrmpConfig config;
  config.receding = 3;
  config.state_dim = 4;
  config.control_dim = 4;
  config.geom_dim = 4;
  config.point_dim = 3;
  config.max_num = with_obstacles ? 1 : 0;
  config.no_obs = !with_obstacles;
  config.enable_control_smoothing = true;
  config.dynamics_A = Eigen::Matrix4d::Identity();
  config.dynamics_B = Eigen::Matrix4d::Identity() * 0.1;
  config.dynamics_C = Eigen::Vector4d::Zero();
  config.speed_bound = Eigen::Vector4d::Constant(2.0);
  config.acce_bound = Eigen::Vector4d::Constant(10.0);
  config.tracking_speed_bound = Eigen::Vector4d::Constant(
      std::numeric_limits<double>::infinity());
  return config;
}

neupan_uav::DunePostprocessorConfig basicDuneConfig() {
  neupan_uav::DunePostprocessorConfig config;
  config.receding = 3;
  config.point_dim = 3;
  config.edge_dim = 6;
  config.dune_max_num = 4;
  config.select_num = 1;
  config.select_nearest_ratio = 1.0;
  config.select_temporal_ratio = 0.0;
  config.select_diversity_ratio = 0.0;
  config.G.resize(6, 3);
  config.G << 1.0F, 0.0F, 0.0F,
              -1.0F, 0.0F, 0.0F,
              0.0F, 1.0F, 0.0F,
              0.0F, -1.0F, 0.0F,
              0.0F, 0.0F, 1.0F,
              0.0F, 0.0F, -1.0F;
  config.h.resize(6);
  config.h << 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F;
  return config;
}

neupan_uav::RknnMetadata mockMetadata() {
  neupan_uav::RknnMetadata metadata;
  metadata.receding = 3;
  metadata.dune_max_num = 4;
  metadata.max_points = 16;
  metadata.output_dim = 6;
  metadata.input_shape = {{1, 16, 3}};
  metadata.output_shape = {{1, 16, 6}};
  metadata.half_extent = Eigen::Vector3d(0.23, 0.23, 0.06);
  metadata.scene_scale = Eigen::Vector3d::Ones();
  metadata.clearance_scale = Eigen::Vector3d::Ones();
  return metadata;
}

neupan_uav::PlannerConfig fullPlannerConfig(bool with_obstacles) {
  neupan_uav::PlannerConfig config;
  config.receding = 3;
  config.step_time = 0.1;
  config.ref_speed = 1.0;
  config.placeholder_command << 0.6, 0.0, 0.0, 0.0;
  config.collision_threshold = 0.0;
  config.preselect.max_points = 4;
  config.preselect.per_step = 0;
  config.preselect.nearest_ratio = 1.0;
  config.preselect.temporal_ratio = 0.0;
  config.preselect.diversity_ratio = 0.0;
  config.preselect.corridor_margin = Eigen::Vector3d::Constant(10.0);
  config.pan.iter_num = 2;
  config.pan.iter_threshold = 1.0e-9;
  config.pan.nrmp = basicNrmpConfig(with_obstacles);
  config.pan.has_nrmp_config = true;
  config.pan.nrmp_max_num = with_obstacles ? 1 : 0;
  config.pan.dune_max_num = with_obstacles ? 4 : 0;
  config.pan.point_flow.receding = config.receding;
  config.pan.point_flow.dune_max_num =
      static_cast<std::size_t>(config.pan.dune_max_num);
  config.pan.point_flow.body_half_extent = config.robot.body_half_extent;
  config.pan.p_u = 1.0;
  config.pan.bk = 0.01;
  config.pan.smooth_u0 = 25.0;
  config.pan.smooth_du = 0.0;
  if (with_obstacles) {
    config.pan.dune = basicDuneConfig();
    config.pan.has_dune_config = true;
    config.pan.rknn_mode = neupan_uav::RknnRunnerMode::kMock;
  }
  return config;
}

}  // namespace

TEST(PlannerReplaySkeleton, EmptyAndStaticObstacleScenariosAreDeterministic) {
  neupan_uav::PlannerConfig config;
  config.placeholder_command << 0.2, -0.1, 0.05, 0.01;
  config.collision_threshold = 0.0;
  neupan_uav::Planner planner(config);

  neupan_uav::PlannerInput empty_input;
  empty_input.state = Eigen::Vector4d(0.0, 0.0, 1.0, 0.0);
  empty_input.obstacle_points = neupan_uav::emptyPointMatrix();

  const neupan_uav::PlannerOutput first = planner.forward(empty_input);
  const neupan_uav::PlannerOutput second = planner.forward(empty_input);

  EXPECT_TRUE(first.ready);
  EXPECT_TRUE(second.ready);
  EXPECT_EQ(first.reason, "planner_ok");
  EXPECT_EQ(second.reason, "planner_ok");
  EXPECT_TRUE(first.command.isApprox(config.placeholder_command));
  EXPECT_TRUE(second.command.isApprox(config.placeholder_command));
  EXPECT_TRUE(std::isinf(first.min_distance));

  neupan_uav::PlannerInput static_input = empty_input;
  static_input.obstacle_points.resize(3, 2);
  static_input.obstacle_points << 3.0, -4.0,
                                  0.0,  0.0,
                                  1.0,  1.0;

  const neupan_uav::PlannerOutput obstacle_out = planner.forward(static_input);

  EXPECT_TRUE(obstacle_out.ready);
  EXPECT_EQ(obstacle_out.reason, "planner_ok");
  EXPECT_EQ(obstacle_out.profile.input_obstacle_count, 2u);
  EXPECT_EQ(obstacle_out.profile.preselected_obstacle_count, 2u);
  EXPECT_NEAR(obstacle_out.min_distance, 2.77, 1e-12);
}

TEST(PlannerStage6, InitialPathBuildsReferenceTrajectoryAndDesiredCommand) {
  neupan_uav::PlannerConfig config;
  config.receding = 3;
  config.step_time = 0.5;
  config.ref_speed = 1.0;
  config.collision_threshold = 0.0;
  config.robot.max_control =
      neupan_uav::Control::Constant(std::numeric_limits<double>::infinity());
  config.initial_path.waypoints = {
      Eigen::Vector4d(0.0, 0.0, 1.0, 0.0),
      Eigen::Vector4d(1.0, 0.0, 1.0, 0.0),
      Eigen::Vector4d(1.0, 1.0, 1.0, 1.5707963267948966),
  };
  neupan_uav::Planner planner(config);

  neupan_uav::PlannerInput input;
  input.state = Eigen::Vector4d(0.0, 0.0, 1.0, 0.0);
  input.obstacle_points = neupan_uav::emptyPointMatrix();

  const neupan_uav::PlannerOutput out = planner.forward(input);

  ASSERT_EQ(out.reference.rows(), 4);
  ASSERT_EQ(out.reference.cols(), config.receding + 1);
  EXPECT_TRUE(out.command.isApprox(
      (neupan_uav::Control() << 1.0, 0.0, 0.0, 0.0).finished()));
  EXPECT_TRUE(out.reference.col(0).isApprox(Eigen::Vector4d(0.0, 0.0, 1.0, 0.0)));
  EXPECT_TRUE(out.reference.col(1).isApprox(Eigen::Vector4d(0.5, 0.0, 1.0, 0.0)));
  EXPECT_TRUE(out.reference.col(2).isApprox(Eigen::Vector4d(1.0, 0.0, 1.0, 0.0)));
  EXPECT_TRUE(out.reference.col(3).isApprox(
      Eigen::Vector4d(1.0, 0.5, 1.0, 0.7853981633974483)));
}

TEST(PlannerStage75, FarfieldGuideAppliesOncePerForwardCycle) {
  neupan_uav::PlannerConfig config;
  config.receding = 3;
  config.step_time = 1.0;
  config.ref_speed = 2.0;
  config.collision_threshold = 0.0;
  config.robot.max_control =
      neupan_uav::Control::Constant(std::numeric_limits<double>::infinity());
  config.initial_path.waypoints = {
      Eigen::Vector4d(0.0, 0.0, 2.0, 0.0),
      Eigen::Vector4d(10.0, 0.0, 2.0, 0.0),
  };
  config.farfield_guide.enabled = true;
  config.farfield_guide.range_backoff = 0.0;
  config.farfield_guide.range_scale = 1.5;
  config.farfield_guide.lateral_width = 3.0;
  config.farfield_guide.center_width = 0.5;
  config.farfield_guide.height_window = 1.0;
  config.farfield_guide.voxel_size = Eigen::Vector3d(0.25, 0.25, 0.25);
  config.farfield_guide.trigger_count = 2;
  config.farfield_guide.release_count = 0;
  config.farfield_guide.release_confirm_cycles = 2;
  config.farfield_guide.offset_min = 1.0;
  config.farfield_guide.offset_max = 4.0;
  config.farfield_guide.offset_speed_gain = 1.0;
  config.farfield_guide.offset_alpha = 0.25;

  neupan_uav::Planner planner(config);
  neupan_uav::PlannerInput input;
  input.state = Eigen::Vector4d(0.0, 0.0, 2.0, 0.0);
  input.obstacle_points.resize(3, 2);
  input.obstacle_points << 6.2, 6.6,
                           0.0, 0.1,
                           2.0, 2.0;

  const neupan_uav::PlannerOutput out = planner.forward(input);

  EXPECT_TRUE(out.ready);
  EXPECT_TRUE(out.profile.farfield_active);
  EXPECT_EQ(out.profile.farfield_center_count, 2);
  EXPECT_NEAR(out.profile.farfield_target_offset_m, 2.0, 1e-12);
  EXPECT_NEAR(out.profile.farfield_offset_m, 0.5, 1e-12);
  ASSERT_EQ(out.reference.cols(), config.receding + 1);
  EXPECT_NEAR(out.reference(1, config.receding), 0.5, 1e-12);
}

TEST(PlannerStage6, InitialPathArrivalLatchesUntilReset) {
  neupan_uav::PlannerConfig config;
  config.arrive_threshold = 0.1;
  config.initial_path.waypoints = {
      Eigen::Vector4d(0.0, 0.0, 0.0, 0.0),
      Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
  };
  neupan_uav::Planner planner(config);

  neupan_uav::PlannerInput input;
  input.state = Eigen::Vector4d(0.95, 0.0, 0.0, 0.0);
  input.obstacle_points = neupan_uav::emptyPointMatrix();

  const neupan_uav::PlannerOutput arrived = planner.forward(input);
  EXPECT_TRUE(arrived.arrive);
  EXPECT_EQ(arrived.reason, "arrived");

  input.state = Eigen::Vector4d(0.5, 0.0, 0.0, 0.0);
  const neupan_uav::PlannerOutput still_arrived = planner.forward(input);
  EXPECT_TRUE(still_arrived.arrive);

  planner.reset();
  const neupan_uav::PlannerOutput after_reset = planner.forward(input);
  EXPECT_FALSE(after_reset.arrive);
  EXPECT_EQ(after_reset.reason, "planner_ok");
}

#ifdef NEUPAN_UAV_WITH_OSQP

TEST(PlannerStage6, ForwardRunsFullNrmpCycleWithoutObstacles) {
  neupan_uav::PlannerConfig config = fullPlannerConfig(false);
  neupan_uav::Planner planner(config);
  neupan_uav::Control applied;
  applied << 1.0, -0.5, 0.25, 0.1;
  planner.notifyAppliedControl(applied);

  neupan_uav::PlannerInput input;
  input.state = Eigen::Vector4d(0.0, 0.0, 1.0, 0.0);
  input.obstacle_points = neupan_uav::emptyPointMatrix();

  const neupan_uav::PlannerOutput out = planner.forward(input);

  EXPECT_TRUE(out.ready);
  EXPECT_EQ(out.reason, "planner_ok");
  EXPECT_TRUE(out.seed_control.isApprox(applied));
  EXPECT_EQ(out.trajectory.rows(), config.pan.nrmp.state_dim);
  EXPECT_EQ(out.trajectory.cols(), config.receding + 1);
  EXPECT_EQ(out.reference.cols(), config.receding + 1);
  EXPECT_EQ(out.control_trajectory.cols(), config.receding);
  EXPECT_GT(out.profile.osqp_iteration_count, 0);
  EXPECT_EQ(out.profile.pan_iteration_limit, config.pan.iter_num);
  EXPECT_TRUE(out.command.allFinite());
  EXPECT_TRUE(planner.previousAppliedControl().isApprox(applied));
}

TEST(PlannerStage6, ForwardRunsMockDuneToNrmpObstacleCycle) {
  neupan_uav::PlannerConfig config = fullPlannerConfig(true);
  neupan_uav::Planner planner(config);
  auto runner = std::make_unique<neupan_uav::MockRknnRunner>(mockMetadata());
  std::vector<float> raw(
      static_cast<std::size_t>(runner->metadata().max_points *
                               runner->metadata().output_dim),
      0.0F);
  for (int step = 0; step <= config.receding; ++step) {
    const int row = step * runner->metadata().dune_max_num;
    raw[static_cast<std::size_t>(row * runner->metadata().output_dim + 1)] =
        1.0F;
  }
  runner->setOutputFull(raw);
  planner.setRknnRunner(std::move(runner));

  neupan_uav::PlannerInput input;
  input.state = Eigen::Vector4d(0.0, 0.0, 1.0, 0.0);
  input.obstacle_points.resize(3, 1);
  input.obstacle_points << 2.0, 0.0, 1.0;

  const neupan_uav::PlannerOutput out = planner.forward(input);

  EXPECT_TRUE(out.ready);
  EXPECT_EQ(out.reason, "planner_ok");
  EXPECT_EQ(out.profile.input_obstacle_count, 1u);
  EXPECT_EQ(out.profile.preselected_obstacle_count, 1u);
  EXPECT_EQ(out.profile.dune_selected_count, 1u);
  EXPECT_GT(out.profile.dune_sec, 0.0);
  EXPECT_GT(out.profile.osqp_iteration_count, 0);
  EXPECT_EQ(out.nominal_distance.size(), config.receding);
  EXPECT_TRUE(std::isfinite(out.min_distance));
  EXPECT_TRUE(out.command.allFinite());
}

#endif
