#include "neupan_uav/planner.hpp"
#include "neupan_uav/rknn_runner.hpp"

#include <gtest/gtest.h>

#include <numeric>
#include <utility>
#include <variant>

namespace {

neupan_uav::UavState stateAt(double x, double y, double z, double yaw) {
  neupan_uav::UavState state;
  state.position_world << x, y, z;
  state.attitude_world_body =
      Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
  return state;
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

neupan_uav::CompiledPlannerConfig buildConfig(
    neupan_uav::PlannerOptions options = {}) {
  return neupan_uav::compilePlannerConfig(std::move(options));
}

neupan_uav::CompiledPlannerConfig buildConfig(
    neupan_uav::PlannerOptions options,
    const neupan_uav::RknnMetadata& metadata) {
  return neupan_uav::compilePlannerConfig(std::move(options),
                                          neupan_uav::UavDynamicsConfig(),
                                          1.0, metadata);
}

neupan_uav::PlannerOptions fullPlannerOptions(bool with_obstacles) {
  neupan_uav::PlannerOptions options;
  options.grid.horizon_steps = 3;
  options.grid.dt = 0.1;
  options.robot.body_half_extent = Eigen::Vector3d(0.23, 0.23, 0.06);
  options.ref_speed = 1.0;
  options.placeholder_command << 0.6, 0.0, 0.0, 0.0;
  options.collision_threshold = 0.0;
  options.preselect.max_points = 4;
  options.preselect.per_step = 0;
  options.preselect.nearest_ratio = 1.0;
  options.preselect.temporal_ratio = 0.0;
  options.preselect.diversity_ratio = 0.0;
  options.preselect.corridor_margin = Eigen::Vector3d::Constant(10.0);
  options.pan.iter_num = 2;
  options.pan.trajectory_threshold = 1.0e-9;
  options.pan.dune_threshold = 1.0e-9;
  options.nrmp.max_constraints = 1;
  options.pan.p_u = 1.0;
  options.pan.bk = 0.01;
  options.pan.smooth_u0 = 25.0;
  options.pan.smooth_du = 0.0;
  if (with_obstacles) {
    options.dune = neupan_uav::DuneOptions();
  }
  return options;
}

neupan_uav::CompiledPlannerConfig fullPlannerConfig(bool with_obstacles) {
  neupan_uav::PlannerOptions options = fullPlannerOptions(with_obstacles);
  if (with_obstacles) {
    return buildConfig(std::move(options), mockMetadata());
  }
  return buildConfig(std::move(options));
}

const neupan_uav::Tracking& tracking(
    const neupan_uav::PlannerResult& result) {
  return std::get<neupan_uav::Tracking>(result.decision());
}

}  // namespace

TEST(PlannerReplaySkeleton, EmptyAndStaticObstacleScenariosAreDeterministic) {
  neupan_uav::PlannerOptions options;
  options.placeholder_command << 0.2, -0.1, 0.05, 0.01;
  options.collision_threshold = 0.0;
  const neupan_uav::CompiledPlannerConfig config =
      buildConfig(std::move(options));
  neupan_uav::Planner planner(config);

  neupan_uav::PlannerInput empty_input;
  empty_input.state = stateAt(0.0, 0.0, 1.0, 0.0);
  empty_input.obstacle_points = neupan_uav::emptyPointMatrix();

  const neupan_uav::PlannerResult first = planner.forward(empty_input);
  const neupan_uav::PlannerResult second = planner.forward(empty_input);

  EXPECT_TRUE(first.isTracking());
  EXPECT_TRUE(second.isTracking());
  EXPECT_TRUE(tracking(first).command.allFinite());
  EXPECT_TRUE(tracking(second).command.allFinite());
  EXPECT_TRUE(second.diagnostics().warm_start_seed.isApprox(
      tracking(first).command));
  EXPECT_TRUE(std::isinf(first.diagnostics().min_clearance.value()));

  neupan_uav::PlannerInput static_input = empty_input;
  static_input.obstacle_points.resize(3, 2);
  static_input.obstacle_points << 3.0, -4.0,
                                  0.0,  0.0,
                                  1.0,  1.0;

  const neupan_uav::PlannerResult obstacle_out = planner.forward(static_input);

  EXPECT_TRUE(obstacle_out.isTracking());
  EXPECT_EQ(obstacle_out.diagnostics().profile.input_obstacle_count, 2u);
  EXPECT_EQ(obstacle_out.diagnostics().profile.preselected_obstacle_count, 2u);
  EXPECT_NEAR(obstacle_out.diagnostics().min_clearance.value(), 3.0, 1e-12);
}

TEST(PlannerStage6, InitialPathBuildsReferenceTrajectoryAndDesiredCommand) {
  neupan_uav::PlannerOptions options;
  options.grid.horizon_steps = 3;
  options.grid.dt = 0.5;
  options.ref_speed = 1.0;
  options.collision_threshold = 0.0;
  options.robot.max_control =
      neupan_uav::Control::Constant(std::numeric_limits<double>::infinity());
  options.initial_path.waypoints = {
      Eigen::Vector4d(0.0, 0.0, 1.0, 0.0),
      Eigen::Vector4d(1.0, 0.0, 1.0, 0.0),
      Eigen::Vector4d(1.0, 1.0, 1.0, 1.5707963267948966),
  };
  const neupan_uav::CompiledPlannerConfig config =
      buildConfig(std::move(options));
  neupan_uav::Planner planner(config);

  neupan_uav::PlannerInput input;
  input.state = stateAt(0.0, 0.0, 1.0, 0.0);
  input.obstacle_points = neupan_uav::emptyPointMatrix();

  const neupan_uav::PlannerResult out = planner.forward(input);
  ASSERT_TRUE(out.isTracking());
  const auto& tracked = tracking(out);

  ASSERT_EQ(tracked.plan.reference.rows(), 8);
  ASSERT_EQ(tracked.plan.reference.cols(), config.receding() + 1);
  EXPECT_TRUE(tracked.command.allFinite());
  EXPECT_GT(tracked.command(0), 0.0);
  EXPECT_TRUE(tracked.plan.reference.col(0).head<4>().isApprox(
      Eigen::Vector4d(0.0, 0.0, 1.0, 0.0)));
  EXPECT_TRUE(tracked.plan.reference.col(1).head<4>().isApprox(
      Eigen::Vector4d(0.5, 0.0, 1.0, 0.0)));
  EXPECT_TRUE(tracked.plan.reference.col(2).head<4>().isApprox(
      Eigen::Vector4d(1.0, 0.0, 1.0, 0.0)));
  EXPECT_TRUE(tracked.plan.reference.col(3).head<4>().isApprox(
      Eigen::Vector4d(1.0, 0.5, 1.0, 0.7853981633974483)));
}

TEST(PlannerStage75, FarfieldGuideAppliesOncePerForwardCycle) {
  neupan_uav::PlannerOptions options;
  options.grid.horizon_steps = 3;
  options.grid.dt = 1.0;
  options.ref_speed = 2.0;
  options.collision_threshold = 0.0;
  options.robot.max_control =
      neupan_uav::Control::Constant(std::numeric_limits<double>::infinity());
  options.initial_path.waypoints = {
      Eigen::Vector4d(0.0, 0.0, 2.0, 0.0),
      Eigen::Vector4d(10.0, 0.0, 2.0, 0.0),
  };
  options.farfield_guide.enabled = true;
  options.farfield_guide.range_backoff = 0.0;
  options.farfield_guide.range_scale = 1.5;
  options.farfield_guide.lateral_width = 3.0;
  options.farfield_guide.center_width = 0.5;
  options.farfield_guide.height_window = 1.0;
  options.farfield_guide.voxel_size = Eigen::Vector3d(0.25, 0.25, 0.25);
  options.farfield_guide.trigger_count = 2;
  options.farfield_guide.release_count = 0;
  options.farfield_guide.release_confirm_cycles = 2;
  options.farfield_guide.offset_min = 1.0;
  options.farfield_guide.offset_max = 4.0;
  options.farfield_guide.offset_speed_gain = 1.0;
  options.farfield_guide.offset_alpha = 0.25;

  const neupan_uav::CompiledPlannerConfig config =
      buildConfig(std::move(options));
  neupan_uav::Planner planner(config);
  neupan_uav::PlannerInput input;
  input.state = stateAt(0.0, 0.0, 2.0, 0.0);
  input.obstacle_points.resize(3, 2);
  input.obstacle_points << 6.2, 6.6,
                           0.0, 0.1,
                           2.0, 2.0;

  const neupan_uav::PlannerResult out = planner.forward(input);

  EXPECT_TRUE(out.isTracking());
  EXPECT_TRUE(out.diagnostics().profile.farfield_active);
  EXPECT_EQ(out.diagnostics().profile.farfield_center_count, 2);
  EXPECT_NEAR(out.diagnostics().profile.farfield_target_offset_m, 2.0, 1e-12);
  EXPECT_NEAR(out.diagnostics().profile.farfield_offset_m, 0.5, 1e-12);
  ASSERT_EQ(tracking(out).plan.reference.cols(), config.receding() + 1);
  EXPECT_NEAR(tracking(out).plan.reference(1, config.receding()), 0.5,
              1e-12);
}

TEST(PlannerStage6, InitialPathArrivalLatchesUntilReset) {
  neupan_uav::PlannerOptions options;
  options.arrive_threshold = 0.1;
  options.initial_path.waypoints = {
      Eigen::Vector4d(0.0, 0.0, 0.0, 0.0),
      Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
  };
  const neupan_uav::CompiledPlannerConfig config =
      buildConfig(std::move(options));
  neupan_uav::Planner planner(config);

  neupan_uav::PlannerInput input;
  input.state = stateAt(0.95, 0.0, 0.0, 0.0);
  input.obstacle_points = neupan_uav::emptyPointMatrix();

  const neupan_uav::PlannerResult arrived = planner.forward(input);
  EXPECT_TRUE(std::holds_alternative<neupan_uav::GoalReached>(
      arrived.decision()));

  input.state = stateAt(0.5, 0.0, 0.0, 0.0);
  const neupan_uav::PlannerResult still_arrived = planner.forward(input);
  EXPECT_TRUE(std::holds_alternative<neupan_uav::GoalReached>(
      still_arrived.decision()));

  planner.reset();
  const neupan_uav::PlannerResult after_reset = planner.forward(input);
  EXPECT_TRUE(after_reset.isTracking());
}

TEST(PlannerStage6, ForwardRunsFullNrmpCycleWithoutObstacles) {
  const neupan_uav::CompiledPlannerConfig config = fullPlannerConfig(false);
  neupan_uav::Planner planner(config);

  neupan_uav::PlannerInput input;
  input.state = stateAt(0.0, 0.0, 1.0, 0.0);
  input.obstacle_points = neupan_uav::emptyPointMatrix();

  const neupan_uav::PlannerResult out = planner.forward(input);

  EXPECT_TRUE(out.isTracking());
  EXPECT_TRUE(out.diagnostics().warm_start_seed.isZero());
  EXPECT_EQ(tracking(out).plan.trajectory.rows(), config.pan().nrmp.state_dim);
  EXPECT_EQ(tracking(out).plan.trajectory.cols(), config.receding() + 1);
  EXPECT_EQ(tracking(out).plan.reference.cols(), config.receding() + 1);
  EXPECT_EQ(tracking(out).plan.control_trajectory.cols(), config.receding());
  EXPECT_GT(out.diagnostics().profile.osqp_iteration_count, 0);
  EXPECT_EQ(out.diagnostics().profile.pan_iteration_limit, config.pan().iter_num);
  EXPECT_TRUE(tracking(out).command.allFinite());
  EXPECT_TRUE(planner.previousCommand().isApprox(tracking(out).command));
}

TEST(PlannerStage6, PreviousCommandSeedsNextNrmpCycle) {
  neupan_uav::PlannerOptions options = fullPlannerOptions(false);
  options.nrmp.enable_control_smoothing = true;
  options.pan.smooth_u0 = 100.0;
  options.pan.smooth_du = 0.0;
  const neupan_uav::CompiledPlannerConfig enabled_config =
      buildConfig(std::move(options));

  neupan_uav::PlannerInput input;
  input.state = stateAt(0.0, 0.0, 1.0, 0.0);
  input.obstacle_points = neupan_uav::emptyPointMatrix();

  neupan_uav::Planner enabled_planner(enabled_config);
  const neupan_uav::PlannerResult enabled_seed =
      enabled_planner.forward(input);
  const neupan_uav::PlannerResult enabled = enabled_planner.forward(input);

  ASSERT_TRUE(enabled_seed.isTracking());
  ASSERT_TRUE(enabled.isTracking());
  EXPECT_TRUE(enabled.diagnostics().warm_start_seed.isApprox(
      tracking(enabled_seed).command));
  EXPECT_LT((tracking(enabled).command -
             enabled.diagnostics().warm_start_seed).norm(),
            0.25);
  EXPECT_TRUE(enabled_planner.previousCommand().isApprox(
      tracking(enabled).command));
}

TEST(PlannerStage6, ForwardRunsMockDuneToNrmpObstacleCycle) {
  const neupan_uav::CompiledPlannerConfig config = fullPlannerConfig(true);
  neupan_uav::Planner planner(config);
  auto runner = std::make_unique<neupan_uav::MockRknnRunner>(mockMetadata());
  std::vector<float> raw(
      static_cast<std::size_t>(runner->metadata().max_points *
                               runner->metadata().output_dim),
      0.0F);
  for (int step = 0; step <= config.receding(); ++step) {
    const int row = step * runner->metadata().dune_max_num;
    raw[static_cast<std::size_t>(row * runner->metadata().output_dim + 1)] =
        1.0F;
  }
  runner->setOutputFull(raw);
  planner.setRknnRunner(std::move(runner));

  neupan_uav::PlannerInput input;
  input.state = stateAt(0.0, 0.0, 1.0, 0.0);
  input.obstacle_points.resize(3, 1);
  input.obstacle_points << 2.0, 0.0, 1.0;

  const neupan_uav::PlannerResult out = planner.forward(input);

  EXPECT_TRUE(out.isTracking());
  EXPECT_EQ(out.diagnostics().profile.input_obstacle_count, 1u);
  EXPECT_EQ(out.diagnostics().profile.preselected_obstacle_count, 1u);
  EXPECT_EQ(out.diagnostics().profile.dune_selected_count, 1u);
  EXPECT_GT(out.diagnostics().profile.dune_sec, 0.0);
  EXPECT_GT(out.diagnostics().profile.osqp_iteration_count, 0);
  EXPECT_EQ(tracking(out).plan.nominal_distance.size(), config.receding());
  EXPECT_TRUE(std::isfinite(out.diagnostics().min_clearance.value()));
  EXPECT_TRUE(tracking(out).command.allFinite());
}

TEST(PlannerStage6, RejectsInjectedRknnRunnerRuntimeMismatch) {
  const neupan_uav::CompiledPlannerConfig config = fullPlannerConfig(true);
  neupan_uav::Planner planner(config);
  neupan_uav::RknnMetadata metadata = mockMetadata();
  metadata.receding = config.receding() + 1;

  auto runner = std::make_unique<neupan_uav::MockRknnRunner>(metadata);

  EXPECT_THROW(planner.setRknnRunner(std::move(runner)), std::invalid_argument);
}

TEST(PlannerStage6, StableDuneDoesNotStopWhenTrajectoryChanges) {
  neupan_uav::PlannerOptions options = fullPlannerOptions(true);
  options.pan.iter_num = 3;
  options.pan.trajectory_threshold = 1.0e-12;
  options.pan.dune_threshold = 1.0;
  const neupan_uav::CompiledPlannerConfig config =
      buildConfig(std::move(options), mockMetadata());
  neupan_uav::Planner planner(config);

  auto runner = std::make_unique<neupan_uav::MockRknnRunner>(mockMetadata());
  runner->setOutputFull(std::vector<float>(
      static_cast<std::size_t>(runner->metadata().max_points *
                               runner->metadata().output_dim),
      0.0F));
  planner.setRknnRunner(std::move(runner));

  neupan_uav::PlannerInput input;
  input.state = stateAt(0.0, 0.0, 1.0, 0.0);
  input.obstacle_points.resize(3, 1);
  input.obstacle_points << 2.0, 0.0, 1.0;

  const neupan_uav::PlannerResult out = planner.forward(input);

  ASSERT_TRUE(out.isTracking());
  EXPECT_EQ(out.diagnostics().profile.pan_iterations, config.pan().iter_num);
}
