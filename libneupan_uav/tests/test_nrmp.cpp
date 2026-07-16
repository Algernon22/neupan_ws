#include "neupan_uav/nrmp.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace {

neupan_uav::NrmpConfig basicConfig(bool with_obstacles = false) {
  neupan_uav::NrmpConfig config;
  config.receding = 3;
  config.state_dim = 4;
  config.control_dim = 4;
  config.geom_dim = 4;
  config.point_dim = 3;
  config.max_num = with_obstacles ? 2 : 0;
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

neupan_uav::NrmpInput basicInput(const neupan_uav::NrmpConfig& config) {
  neupan_uav::NrmpInput input;
  input.seed_control << 0.0, 0.0, 0.0, 0.0;
  input.desired_control << 0.0, 0.0, 0.0, 0.0;
  input.nominal_states =
      neupan_uav::Trajectory::Zero(config.state_dim, config.receding + 1);
  input.reference_states =
      neupan_uav::Trajectory::Zero(config.state_dim, config.receding + 1);
  input.reference_controls =
      Eigen::MatrixXd::Zero(config.control_dim, config.receding);
  input.state_weights = Eigen::VectorXd::Ones(config.state_dim);
  input.p_u = 1.0;
  input.bk = 0.01;
  input.smooth_du = 0.0;
  input.smooth_u0 = 0.0;
  if (!config.no_obs) {
    input.fa_batch =
        Eigen::MatrixXd::Zero(config.receding * config.max_num,
                              config.point_dim);
    input.fb_batch = Eigen::MatrixXd::Constant(
        config.receding, config.max_num, -2.0);
  }
  return input;
}

}  // namespace

TEST(NRMP, DefaultConstructorKeepsPlaceholderCompatibility) {
  neupan_uav::NRMP nrmp;
  neupan_uav::NrmpInput input;
  input.seed_control << 1.0, 2.0, 3.0, 4.0;
  input.desired_control << -1.0, -2.0, -3.0, -4.0;

  const neupan_uav::NrmpResult result = nrmp.solve(input);

  EXPECT_FALSE(nrmp.hasBackend());
  EXPECT_TRUE(result.control.isApprox(input.desired_control));
  EXPECT_EQ(result.status, 0);
  EXPECT_EQ(result.iterations, 0);
}

#ifdef NEUPAN_UAV_WITH_OSQP

TEST(NRMP, SolvesNoObstacleProblem) {
  const neupan_uav::NrmpConfig config = basicConfig(false);
  neupan_uav::NRMP nrmp(config);
  neupan_uav::NrmpInput input = basicInput(config);
  input.reference_controls.row(0).setConstant(0.5);
  input.reference_states.row(0) << 0.0, 0.05, 0.10, 0.15;

  const neupan_uav::NrmpResult result = nrmp.solve(input);

  EXPECT_TRUE(nrmp.hasBackend());
  EXPECT_EQ(result.state_trajectory.rows(), config.state_dim);
  EXPECT_EQ(result.state_trajectory.cols(), config.receding + 1);
  EXPECT_EQ(result.control_trajectory.rows(), config.control_dim);
  EXPECT_EQ(result.control_trajectory.cols(), config.receding);
  EXPECT_GE(result.status, 1);
  EXPECT_GT(result.iterations, 0);
  EXPECT_NEAR(result.control(0), 0.5, 0.15);
}

TEST(NRMP, SolvesWithObstacleCoefficients) {
  const neupan_uav::NrmpConfig config = basicConfig(true);
  neupan_uav::NRMP nrmp(config);
  neupan_uav::NrmpInput input = basicInput(config);
  input.d_min = 0.1;
  input.d_max = 1.0;
  input.eta = 1.0;
  input.ro_obs = 10.0;
  for (int t = 0; t < config.receding; ++t) {
    input.fa_batch(t * config.max_num, 0) = 1.0;
    input.fa_batch(t * config.max_num + 1, 1) = 1.0;
    input.fb_batch(t, 0) = -0.5;
    input.fb_batch(t, 1) = -0.5;
  }

  const neupan_uav::NrmpResult result = nrmp.solve(input);

  EXPECT_EQ(result.nominal_distance.size(), config.receding);
  EXPECT_GE(result.status, 1);
  EXPECT_TRUE(result.control.allFinite());
}

TEST(NRMP, PreviousControlSmoothingUsesAppliedSeed) {
  const neupan_uav::NrmpConfig config = basicConfig(false);
  neupan_uav::NRMP nrmp(config);
  neupan_uav::NrmpInput input = basicInput(config);
  input.p_u = 0.1;
  input.smooth_u0 = 100.0;
  input.seed_control << 1.2, -0.8, 0.4, -0.2;

  const neupan_uav::NrmpResult result = nrmp.solve(input);

  EXPECT_NEAR(result.control(0), input.seed_control(0), 0.2);
  EXPECT_NEAR(result.control(1), input.seed_control(1), 0.2);
  EXPECT_NEAR(result.control(2), input.seed_control(2), 0.2);
  EXPECT_NEAR(result.control(3), input.seed_control(3), 0.2);
}

TEST(NRMP, RejectsBadRuntimeDimensions) {
  const neupan_uav::NrmpConfig config = basicConfig(false);
  neupan_uav::NRMP nrmp(config);
  neupan_uav::NrmpInput input = basicInput(config);
  input.reference_controls.resize(config.control_dim, config.receding + 1);

  EXPECT_THROW(nrmp.solve(input), std::invalid_argument);
}

TEST(NRMP, ReusesFixedSparseWorkspaceAcrossSolves) {
  const neupan_uav::NrmpConfig config = basicConfig(false);
  neupan_uav::NRMP nrmp(config);
  neupan_uav::NrmpInput input = basicInput(config);

  const neupan_uav::NrmpResult first = nrmp.solve(input);
  input.reference_controls.row(1).setConstant(-0.4);
  const neupan_uav::NrmpResult second = nrmp.solve(input);

  EXPECT_EQ(first.setup_count, 1);
  EXPECT_EQ(second.setup_count, 1);
  EXPECT_EQ(first.solve_count, 1);
  EXPECT_EQ(second.solve_count, 2);
  EXPECT_NEAR(second.control(1), -0.4, 0.15);
}

#endif
