#include "neupan_uav/config.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

namespace {

neupan_uav::RknnMetadata metadata() {
  neupan_uav::RknnMetadata out;
  out.receding = 25;
  out.dune_max_num = 160;
  out.max_points = 26 * 160;
  out.output_dim = 6;
  out.input_shape = {{1, out.max_points, 3}};
  out.output_shape = {{1, out.max_points, 6}};
  out.half_extent = Eigen::Vector3d(0.32, 0.32, 0.27);
  out.scene_scale = Eigen::Vector3d::Ones();
  out.clearance_scale = Eigen::Vector3d::Ones();
  return out;
}

neupan_uav::PlannerOptions basicOptions() {
  neupan_uav::PlannerOptions options;
  options.grid.horizon_steps = 25;
  options.grid.dt = 0.1;
  options.robot.body_half_extent = Eigen::Vector3d(0.32, 0.32, 0.27);
  options.robot.max_control << 4.0, 4.0, 2.0, 0.8;
  options.dune = neupan_uav::DuneOptions();
  options.nrmp.max_constraints = 20;
  return options;
}

neupan_uav::UavDynamicsConfig basicDynamics() {
  neupan_uav::UavDynamicsConfig dynamics;
  dynamics.max_acceleration << 2.5, 2.5, 1.5, 0.8;
  dynamics.velocity_time_constant << 0.45, 0.45, 0.35;
  dynamics.velocity_gain = Eigen::Vector3d::Ones();
  dynamics.yaw_rate_time_constant = 0.30;
  dynamics.yaw_rate_gain = 1.0;
  dynamics.velocity_weight_scale = 0.35;
  return dynamics;
}

}  // namespace

TEST(Config, CompilePlannerConfigPopulatesDerivedFields) {
  const neupan_uav::CompiledPlannerConfig config =
      neupan_uav::compilePlannerConfig(
          basicOptions(), basicDynamics(), 1.1, metadata());

  EXPECT_EQ(config.receding(), 25);
  EXPECT_NEAR(config.stepTime(), 0.1, 1e-12);
  EXPECT_EQ(config.pan().grid.horizon_steps, 25);
  EXPECT_NEAR(config.pan().grid.dt, 0.1, 1e-12);
  EXPECT_EQ(config.pan().point_flow.receding, 25);
  EXPECT_NEAR(config.pan().point_flow.dt, 0.1, 1e-12);
  EXPECT_EQ(config.pan().point_flow.dune_max_num, 160U);
  EXPECT_TRUE(config.pan().point_flow.body_half_extent.isApprox(
      config.robot().body_half_extent));
  EXPECT_EQ(config.preselect().max_points, 160U);

  ASSERT_TRUE(config.pan().dune.has_value());
  const auto& dune = *config.pan().dune;
  EXPECT_EQ(dune.receding, 25);
  EXPECT_EQ(dune.point_dim, 3);
  EXPECT_EQ(dune.edge_dim, 6);
  EXPECT_EQ(dune.dune_max_num, 160U);
  EXPECT_EQ(dune.select_num, 20U);
  ASSERT_EQ(dune.G.rows(), 6);
  ASSERT_EQ(dune.G.cols(), 3);
  EXPECT_FLOAT_EQ(dune.G(0, 0), 1.0F);
  EXPECT_FLOAT_EQ(dune.G(1, 0), -1.0F);
  EXPECT_FLOAT_EQ(dune.G(4, 2), 1.0F);
  EXPECT_FLOAT_EQ(dune.G(5, 2), -1.0F);
  ASSERT_EQ(dune.h.size(), 6);
  EXPECT_FLOAT_EQ(dune.h(0), 0.32F);
  EXPECT_FLOAT_EQ(dune.h(2), 0.32F);
  EXPECT_FLOAT_EQ(dune.h(4), 0.27F);

  EXPECT_EQ(config.pan().nrmp.receding, 25);
  EXPECT_EQ(config.pan().nrmp.state_dim, 8);
  EXPECT_EQ(config.pan().nrmp.control_dim, 4);
  EXPECT_EQ(config.pan().nrmp.geom_dim, 3);
  EXPECT_EQ(config.pan().nrmp.point_dim, 3);
  EXPECT_EQ(config.pan().nrmp.max_num, 20);
  EXPECT_FALSE(config.pan().nrmp.no_obs);
  EXPECT_EQ(config.pan().nrmp.dynamics_A.rows(), 8);
  EXPECT_EQ(config.pan().nrmp.dynamics_A.cols(), 8);
  EXPECT_EQ(config.pan().nrmp.dynamics_B.rows(), 8);
  EXPECT_EQ(config.pan().nrmp.dynamics_B.cols(), 4);
  EXPECT_EQ(config.pan().nrmp.dynamics_C.size(), 8);
  EXPECT_TRUE(config.pan().nrmp.speed_bound.isApprox(config.robot().max_control));
  EXPECT_TRUE(config.pan().nrmp.acce_bound.isApprox(
      Eigen::Vector4d(0.25, 0.25, 0.15, 0.08)));
  EXPECT_TRUE(config.pan().nrmp.tracking_speed_bound.isApprox(
      config.robot().max_control.cwiseAbs()));

  ASSERT_EQ(config.pan().state_weights.size(), 8);
  EXPECT_NEAR(config.pan().state_weights(0), 1.0, 1e-12);
  EXPECT_NEAR(config.pan().state_weights(4), 1.1 * 0.35, 1e-12);
  EXPECT_NEAR(config.pan().state_weights(7), 1.1 * 0.35, 1e-12);
}

TEST(Config, CompilePlannerConfigRejectsInvalidOptions) {
  neupan_uav::PlannerOptions options = basicOptions();
  neupan_uav::UavDynamicsConfig dynamics = basicDynamics();
  dynamics.velocity_time_constant(0) = 0.0;

  EXPECT_THROW(neupan_uav::compilePlannerConfig(options, dynamics, 1.0, metadata()),
               std::invalid_argument);
}
