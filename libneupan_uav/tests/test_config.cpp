#include "neupan_uav/config.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

namespace {

neupan_uav::UavPlannerConfigSpec basicSpec() {
  neupan_uav::UavPlannerConfigSpec spec;
  spec.planner.receding = 25;
  spec.planner.step_time = 0.1;
  spec.planner.robot.body_half_extent = Eigen::Vector3d(0.32, 0.32, 0.27);
  spec.planner.robot.max_control << 4.0, 4.0, 2.0, 0.8;
  spec.planner.pan.dune_max_num = 160;
  spec.planner.pan.nrmp_max_num = 20;
  spec.dynamics.max_acceleration << 2.5, 2.5, 1.5, 0.8;
  spec.dynamics.velocity_time_constant << 0.45, 0.45, 0.35;
  spec.dynamics.velocity_gain = Eigen::Vector3d::Ones();
  spec.dynamics.yaw_rate_time_constant = 0.30;
  spec.dynamics.yaw_rate_gain = 1.0;
  spec.dynamics.velocity_weight_scale = 0.35;
  spec.state_weight_gain = 1.1;
  return spec;
}

}  // namespace

TEST(Config, BuildUavPlannerConfigPopulatesDerivedFields) {
  const neupan_uav::PlannerConfig config =
      neupan_uav::buildUavPlannerConfig(basicSpec());

  EXPECT_EQ(config.pan.receding, config.receding);
  EXPECT_NEAR(config.pan.step_time, config.step_time, 1e-12);
  EXPECT_EQ(config.pan.point_flow.receding, config.receding);
  EXPECT_NEAR(config.pan.point_flow.dt, config.step_time, 1e-12);
  EXPECT_EQ(config.pan.point_flow.dune_max_num, 160U);
  EXPECT_TRUE(config.pan.point_flow.body_half_extent.isApprox(
      config.robot.body_half_extent));
  EXPECT_EQ(config.preselect.max_points, 160U);

  EXPECT_TRUE(config.pan.has_dune_config);
  EXPECT_EQ(config.pan.dune.receding, config.receding);
  EXPECT_EQ(config.pan.dune.point_dim, 3);
  EXPECT_EQ(config.pan.dune.edge_dim, 6);
  EXPECT_EQ(config.pan.dune.dune_max_num, 160U);
  EXPECT_EQ(config.pan.dune.select_num, 20U);
  ASSERT_EQ(config.pan.dune.G.rows(), 6);
  ASSERT_EQ(config.pan.dune.G.cols(), 3);
  EXPECT_FLOAT_EQ(config.pan.dune.G(0, 0), 1.0F);
  EXPECT_FLOAT_EQ(config.pan.dune.G(1, 0), -1.0F);
  EXPECT_FLOAT_EQ(config.pan.dune.G(4, 2), 1.0F);
  EXPECT_FLOAT_EQ(config.pan.dune.G(5, 2), -1.0F);
  ASSERT_EQ(config.pan.dune.h.size(), 6);
  EXPECT_FLOAT_EQ(config.pan.dune.h(0), 0.32F);
  EXPECT_FLOAT_EQ(config.pan.dune.h(2), 0.32F);
  EXPECT_FLOAT_EQ(config.pan.dune.h(4), 0.27F);

  EXPECT_TRUE(config.pan.has_nrmp_config);
  EXPECT_EQ(config.pan.nrmp.receding, config.receding);
  EXPECT_EQ(config.pan.nrmp.state_dim, 8);
  EXPECT_EQ(config.pan.nrmp.control_dim, 4);
  EXPECT_EQ(config.pan.nrmp.geom_dim, 3);
  EXPECT_EQ(config.pan.nrmp.point_dim, 3);
  EXPECT_EQ(config.pan.nrmp.max_num, 20);
  EXPECT_FALSE(config.pan.nrmp.no_obs);
  EXPECT_EQ(config.pan.nrmp.dynamics_A.rows(), 8);
  EXPECT_EQ(config.pan.nrmp.dynamics_A.cols(), 8);
  EXPECT_EQ(config.pan.nrmp.dynamics_B.rows(), 8);
  EXPECT_EQ(config.pan.nrmp.dynamics_B.cols(), 4);
  EXPECT_EQ(config.pan.nrmp.dynamics_C.size(), 8);
  EXPECT_TRUE(config.pan.nrmp.speed_bound.isApprox(config.robot.max_control));
  EXPECT_TRUE(config.pan.nrmp.acce_bound.isApprox(
      Eigen::Vector4d(0.25, 0.25, 0.15, 0.08)));
  EXPECT_TRUE(config.pan.nrmp.tracking_speed_bound.isApprox(
      config.robot.max_control.cwiseAbs()));

  ASSERT_EQ(config.pan.state_weights.size(), 8);
  EXPECT_NEAR(config.pan.state_weights(0), 1.0, 1e-12);
  EXPECT_NEAR(config.pan.state_weights(4), 1.1 * 0.35, 1e-12);
  EXPECT_NEAR(config.pan.state_weights(7), 1.1 * 0.35, 1e-12);
}

TEST(Config, BuildUavPlannerConfigRejectsInvalidSpec) {
  neupan_uav::UavPlannerConfigSpec spec = basicSpec();
  spec.dynamics.velocity_time_constant(0) = 0.0;

  EXPECT_THROW(neupan_uav::buildUavPlannerConfig(spec), std::invalid_argument);
}
