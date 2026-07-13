#include "neupan_uav/obstacle_preselector.hpp"

#include <gtest/gtest.h>

TEST(ObstaclePreselector, HandlesEmptyPointCloud) {
  neupan_uav::ObstaclePreselector preselector(4);

  const neupan_uav::ObstacleSelection out =
      preselector.select(neupan_uav::emptyPointMatrix());

  EXPECT_EQ(out.points.rows(), 3);
  EXPECT_EQ(out.points.cols(), 0);
  EXPECT_EQ(out.profile.input_obstacle_count, 0u);
  EXPECT_EQ(out.profile.preselected_obstacle_count, 0u);
  EXPECT_TRUE(out.tags.empty());
}

TEST(ObstaclePreselector, KeepsAllPointsWhenUnderLimit) {
  neupan_uav::PointMatrix points(3, 2);
  points << 1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0;
  neupan_uav::ObstaclePreselector preselector(4);

  const neupan_uav::ObstacleSelection out = preselector.select(points);

  EXPECT_EQ(out.points.cols(), 2);
  EXPECT_TRUE(out.points.isApprox(points));
  EXPECT_EQ(out.profile.input_obstacle_count, 2u);
  EXPECT_EQ(out.profile.preselected_obstacle_count, 2u);
  EXPECT_EQ(out.tags.size(), 2u);
}

TEST(ObstaclePreselector, TruncatesPointsAtLimit) {
  neupan_uav::PointMatrix points(3, 4);
  points << 1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0;
  neupan_uav::ObstaclePreselector preselector(2);

  const neupan_uav::ObstacleSelection out = preselector.select(points);

  EXPECT_EQ(out.points.cols(), 2);
  EXPECT_TRUE(out.points.isApprox(points.leftCols(2)));
  EXPECT_EQ(out.profile.input_obstacle_count, 4u);
  EXPECT_EQ(out.profile.preselected_obstacle_count, 2u);
}

TEST(ObstaclePreselector, PreservesMatchingVelocities) {
  neupan_uav::PointMatrix points(3, 3);
  points.setRandom();
  neupan_uav::PointMatrix velocities(3, 3);
  velocities.setRandom();
  neupan_uav::ObstaclePreselector preselector(2);

  const neupan_uav::ObstacleSelection out =
      preselector.select(points, velocities);

  EXPECT_TRUE(out.points.isApprox(points.leftCols(2)));
  EXPECT_TRUE(out.velocities.isApprox(velocities.leftCols(2)));
}

TEST(ObstaclePreselector, RejectsInvalidVelocityShape) {
  neupan_uav::PointMatrix points(3, 2);
  points.setZero();
  neupan_uav::PointMatrix velocities(3, 1);
  velocities.setZero();
  neupan_uav::ObstaclePreselector preselector(0);

  EXPECT_THROW(preselector.select(points, velocities), std::invalid_argument);
}

TEST(ObstaclePreselector, TrajectorySelectionKeepsPointsWithinAabbCorridor) {
  neupan_uav::ObstaclePreselectorConfig config;
  config.max_points = 10;
  config.per_step = 0;
  config.nearest_ratio = 1.0;
  config.temporal_ratio = 0.0;
  config.diversity_ratio = 0.0;
  config.body_half_extent = Eigen::Vector3d(0.5, 0.5, 0.25);
  config.corridor_margin = Eigen::Vector3d::Zero();
  neupan_uav::ObstaclePreselector preselector(config);

  neupan_uav::Trajectory nominal(4, 2);
  nominal << 0.0, 1.0,
             0.0, 0.0,
             0.0, 0.0,
             0.0, 0.0;
  neupan_uav::PointMatrix points(3, 4);
  points << 0.5, 1.5, 1.51, 0.0,
            0.0, 0.0, 0.0, 0.51,
            0.0, 0.0, 0.0, 0.0;
  Eigen::MatrixXd attitude = Eigen::MatrixXd::Zero(3, 2);

  const neupan_uav::ObstacleSelection out =
      preselector.selectWithNominalTrajectory(nominal, points,
                                              neupan_uav::emptyPointMatrix(),
                                              attitude);

  EXPECT_EQ(out.profile.input_obstacle_count, 4u);
  EXPECT_EQ(out.profile.corridor_obstacle_count, 2u);
  EXPECT_EQ(out.profile.preselected_obstacle_count, 2u);
  EXPECT_EQ(out.points.cols(), 2);
  EXPECT_TRUE(out.points.isApprox(points.leftCols(2)));
}

TEST(ObstaclePreselector, TrajectorySelectionHandlesCountsBelowEqualAboveLimit) {
  neupan_uav::ObstaclePreselectorConfig config;
  config.max_points = 2;
  config.per_step = 0;
  config.nearest_ratio = 1.0;
  config.temporal_ratio = 0.0;
  config.diversity_ratio = 0.0;
  config.body_half_extent = Eigen::Vector3d::Zero();
  config.corridor_margin = Eigen::Vector3d::Constant(10.0);
  neupan_uav::ObstaclePreselector preselector(config);

  neupan_uav::Trajectory nominal(4, 1);
  nominal << 0.0, 0.0, 0.0, 0.0;

  neupan_uav::PointMatrix one(3, 1);
  one << 0.4, 0.0, 0.0;
  EXPECT_EQ(preselector.selectWithNominalTrajectory(nominal, one).points.cols(),
            1);

  neupan_uav::PointMatrix two(3, 2);
  two << 0.4, 0.2,
         0.0, 0.0,
         0.0, 0.0;
  EXPECT_EQ(preselector.selectWithNominalTrajectory(nominal, two).points.cols(),
            2);

  neupan_uav::PointMatrix three(3, 3);
  three << 3.0, 0.2, 1.0,
           0.0, 0.0, 0.0,
           0.0, 0.0, 0.0;
  const neupan_uav::ObstacleSelection out =
      preselector.selectWithNominalTrajectory(nominal, three);
  EXPECT_EQ(out.points.cols(), 2);
  EXPECT_TRUE(out.points.col(0).isApprox(three.col(1)));
  EXPECT_TRUE(out.points.col(1).isApprox(three.col(2)));
}

TEST(ObstaclePreselector, DisabledTrajectorySelectionKeepsInputAndClearsHistory) {
  neupan_uav::ObstaclePreselectorConfig config;
  config.enabled = false;
  config.max_points = 1;
  neupan_uav::ObstaclePreselector preselector(config);

  neupan_uav::PointMatrix previous(3, 1);
  previous << 0.0, 0.0, 0.0;
  preselector.setPreviousPoints(previous);

  neupan_uav::Trajectory nominal(4, 1);
  nominal.setZero();
  neupan_uav::PointMatrix points(3, 3);
  points << 1.0, 2.0, 3.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0;

  const neupan_uav::ObstacleSelection out =
      preselector.selectWithNominalTrajectory(nominal, points);

  EXPECT_TRUE(out.points.isApprox(points));
  EXPECT_EQ(out.profile.input_obstacle_count, 3u);
  EXPECT_EQ(out.profile.preselected_obstacle_count, 3u);
  EXPECT_EQ(preselector.previousPoints().cols(), 0);
}

TEST(ObstaclePreselector, MovingObstacleCrossingCorridorIsKept) {
  neupan_uav::ObstaclePreselectorConfig config;
  config.max_points = 4;
  config.per_step = 0;
  config.body_half_extent = Eigen::Vector3d::Zero();
  config.corridor_margin = Eigen::Vector3d::Constant(0.5);
  config.nearest_ratio = 1.0;
  config.temporal_ratio = 0.0;
  config.diversity_ratio = 0.0;
  config.dt = 1.0;
  neupan_uav::ObstaclePreselector preselector(config);

  neupan_uav::Trajectory nominal(4, 3);
  nominal << 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0,
             0.0, 0.0, 0.0,
             0.0, 0.0, 0.0;
  neupan_uav::PointMatrix points(3, 2);
  points << 2.0, 2.0,
            0.0, 0.0,
            0.0, 0.0;
  neupan_uav::PointMatrix velocities(3, 2);
  velocities << -1.0, 1.0,
                 0.0, 0.0,
                 0.0, 0.0;

  const neupan_uav::ObstacleSelection out =
      preselector.selectWithNominalTrajectory(nominal, points, velocities);

  EXPECT_EQ(out.profile.corridor_obstacle_count, 1u);
  ASSERT_EQ(out.points.cols(), 1);
  EXPECT_TRUE(out.points.col(0).isApprox(points.col(0)));
  EXPECT_TRUE(out.velocities.col(0).isApprox(velocities.col(0)));
}

TEST(ObstaclePreselector, AttitudeHorizonInflatesRotatedAabbCorridor) {
  neupan_uav::ObstaclePreselectorConfig config;
  config.max_points = 4;
  config.per_step = 0;
  config.body_half_extent = Eigen::Vector3d(2.0, 0.1, 0.1);
  config.corridor_margin = Eigen::Vector3d::Zero();
  config.nearest_ratio = 1.0;
  config.temporal_ratio = 0.0;
  config.diversity_ratio = 0.0;
  neupan_uav::ObstaclePreselector preselector(config);

  neupan_uav::Trajectory nominal(4, 1);
  nominal.setZero();
  neupan_uav::PointMatrix points(3, 2);
  points << 0.0, 0.0,
            1.5, 2.5,
            0.0, 0.0;
  Eigen::MatrixXd attitude(3, 1);
  attitude << 0.0, 0.0, M_PI / 2.0;

  const neupan_uav::ObstacleSelection out =
      preselector.selectWithNominalTrajectory(nominal, points,
                                              neupan_uav::emptyPointMatrix(),
                                              attitude);

  EXPECT_EQ(out.profile.corridor_obstacle_count, 1u);
  ASSERT_EQ(out.points.cols(), 1);
  EXPECT_TRUE(out.points.col(0).isApprox(points.col(0)));
}

TEST(ObstaclePreselector, TrajectorySelectionMarksTemporalAndDiversityTags) {
  neupan_uav::ObstaclePreselectorConfig config;
  config.max_points = 3;
  config.per_step = 0;
  config.nearest_ratio = 1.0 / 3.0;
  config.temporal_ratio = 1.0 / 3.0;
  config.diversity_ratio = 1.0 / 3.0;
  config.temporal_radius = 0.2;
  config.body_half_extent = Eigen::Vector3d::Zero();
  config.corridor_margin = Eigen::Vector3d::Constant(10.0);
  config.diversity_voxel = Eigen::Vector3d::Constant(0.5);
  neupan_uav::ObstaclePreselector preselector(config);

  neupan_uav::Trajectory nominal(4, 1);
  nominal << 0.0, 0.0, 0.0, 0.0;
  neupan_uav::PointMatrix previous(3, 1);
  previous << 1.0, 0.0, 0.0;
  preselector.setPreviousPoints(previous);

  neupan_uav::PointMatrix points(3, 5);
  points << 0.1, 1.05, 2.0, 2.1, 3.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0;

  const neupan_uav::ObstacleSelection out =
      preselector.selectWithNominalTrajectory(nominal, points);

  EXPECT_EQ(out.points.cols(), 3);
  ASSERT_EQ(out.tags.size(), 3u);
  bool saw_temporal = false;
  bool saw_diversity = false;
  for (unsigned char tag : out.tags) {
    saw_temporal = saw_temporal || neupan_uav::hasTemporalTag(tag);
    saw_diversity = saw_diversity || neupan_uav::hasDiversityTag(tag);
  }
  EXPECT_TRUE(saw_temporal);
  EXPECT_TRUE(saw_diversity);
  EXPECT_EQ(out.profile.temporal_quota, 1u);
  EXPECT_EQ(out.profile.diversity_quota, 1u);
  EXPECT_EQ(out.profile.continuity_hits, 1u);
}

TEST(ObstaclePreselector, ResetClearsTemporalContinuityHistory) {
  neupan_uav::ObstaclePreselectorConfig config;
  config.max_points = 2;
  config.per_step = 0;
  config.nearest_ratio = 0.5;
  config.temporal_ratio = 0.5;
  config.diversity_ratio = 0.0;
  config.temporal_radius = 1.0;
  config.body_half_extent = Eigen::Vector3d::Zero();
  config.corridor_margin = Eigen::Vector3d::Constant(10.0);
  neupan_uav::ObstaclePreselector preselector(config);

  neupan_uav::PointMatrix previous(3, 1);
  previous << 2.0, 0.0, 0.0;
  preselector.setPreviousPoints(previous);
  preselector.reset();

  neupan_uav::Trajectory nominal(4, 1);
  nominal.setZero();
  neupan_uav::PointMatrix points(3, 2);
  points << 0.2, 2.1,
            0.0, 0.0,
            0.0, 0.0;

  const neupan_uav::ObstacleSelection out =
      preselector.selectWithNominalTrajectory(nominal, points);

  EXPECT_EQ(out.profile.continuity_hits, 0u);
  for (unsigned char tag : out.tags) {
    EXPECT_FALSE(neupan_uav::hasTemporalTag(tag));
  }
}

TEST(PointFlowBuilder, BuildsMovingObstacleFlowAndFallbackSelection) {
  neupan_uav::PointFlowConfig config;
  config.receding = 2;
  config.dt = 0.5;
  config.dune_max_num = 2;
  config.body_half_extent = Eigen::Vector3d::Zero();
  neupan_uav::PointFlowBuilder builder(config);

  neupan_uav::Trajectory nominal(4, 3);
  nominal << 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0,
             0.0, 0.0, 0.0,
             0.0, 0.0, 0.0;
  neupan_uav::PointMatrix points(3, 3);
  points << 3.0, 1.0, 2.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0;
  neupan_uav::PointMatrix velocities(3, 3);
  velocities << 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0;

  const neupan_uav::PointFlowResult out =
      builder.generate(nominal, points, velocities);

  EXPECT_TRUE(out.fallback_applied);
  ASSERT_EQ(out.point_flow.size(), 3u);
  EXPECT_EQ(out.selected_points.cols(), 2);
  EXPECT_TRUE(out.selected_points.col(0).isApprox(points.col(1)));
  EXPECT_TRUE(out.selected_points.col(1).isApprox(points.col(2)));
  EXPECT_NEAR(out.obstacle_points_by_step[2](0, 0), 2.0, 1e-12);
  EXPECT_NEAR(out.point_flow[2](0, 0), 2.0, 1e-12);
}
