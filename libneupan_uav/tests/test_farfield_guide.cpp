#include "neupan_uav/farfield_guide.hpp"

#include <gtest/gtest.h>

namespace {

Eigen::Matrix<neupan_uav::Scalar, 4, Eigen::Dynamic> straightReference() {
  Eigen::Matrix<neupan_uav::Scalar, 4, Eigen::Dynamic> ref(4, 4);
  ref << 0.0, 1.0, 2.0, 3.0,
         0.0, 0.0, 0.0, 0.0,
         2.0, 2.0, 2.0, 2.0,
         0.0, 0.0, 0.0, 0.0;
  return ref;
}

neupan_uav::FarfieldGuideConfig baseConfig() {
  neupan_uav::FarfieldGuideConfig config;
  config.enabled = true;
  config.range_backoff = 0.0;
  config.range_scale = 1.5;
  config.lateral_width = 3.0;
  config.center_width = 0.5;
  config.height_window = 1.0;
  config.voxel_size = Eigen::Vector3d(0.25, 0.25, 0.25);
  config.trigger_count = 2;
  config.release_count = 0;
  config.release_confirm_cycles = 2;
  config.offset_min = 1.0;
  config.offset_max = 4.0;
  config.offset_speed_gain = 1.0;
  config.offset_alpha = 0.25;
  config.release_alpha = 0.5;
  return config;
}

neupan_uav::PointMatrix centerObstacles() {
  neupan_uav::PointMatrix points(3, 2);
  points << 3.2, 3.6,
            0.0, 0.1,
            2.0, 2.0;
  return points;
}

}  // namespace

TEST(FarfieldGuide, DisabledLeavesReferenceUnchanged) {
  neupan_uav::FarfieldGuideConfig config = baseConfig();
  config.enabled = false;
  neupan_uav::FarfieldGuide guide(config);
  const auto ref = straightReference();
  const Eigen::Vector4d state(0.0, 0.0, 2.0, 0.0);

  const auto result =
      guide.apply(ref, centerObstacles(), state, 2.0, 1.0, 3);

  EXPECT_FALSE(result.profile.active);
  EXPECT_TRUE(result.reference_geometry.isApprox(ref));
}

TEST(FarfieldGuide, CenterObstaclesTriggerSmoothedLeftOffset) {
  neupan_uav::FarfieldGuide guide(baseConfig());
  const auto ref = straightReference();
  const Eigen::Vector4d state(0.0, 0.0, 2.0, 0.0);

  const auto result =
      guide.apply(ref, centerObstacles(), state, 2.0, 1.0, 3);

  EXPECT_TRUE(result.profile.active);
  EXPECT_EQ(result.profile.center_count, 2);
  EXPECT_EQ(result.profile.left_count, 0);
  EXPECT_EQ(result.profile.right_count, 0);
  EXPECT_NEAR(result.profile.target_offset_m, 2.0, 1e-12);
  EXPECT_NEAR(result.profile.offset_m, 0.5, 1e-12);
  EXPECT_NEAR(result.reference_geometry(1, 0), 0.0, 1e-12);
  EXPECT_NEAR(result.reference_geometry(1, 3), 0.5, 1e-12);
}

TEST(FarfieldGuide, ChoosesClearerRightSideWhenLeftIsOccupied) {
  neupan_uav::FarfieldGuide guide(baseConfig());
  const auto ref = straightReference();
  const Eigen::Vector4d state(0.0, 0.0, 2.0, 0.0);
  neupan_uav::PointMatrix points(3, 5);
  points << 3.2, 3.6, 3.4, 3.8, 4.2,
            0.0, 0.1, 1.0, 1.4, 1.8,
            2.0, 2.0, 2.0, 2.0, 2.0;

  const auto result = guide.apply(ref, points, state, 2.0, 1.0, 3);

  EXPECT_TRUE(result.profile.active);
  EXPECT_EQ(result.profile.center_count, 2);
  EXPECT_EQ(result.profile.left_count, 3);
  EXPECT_EQ(result.profile.right_count, 0);
  EXPECT_NEAR(result.profile.target_offset_m, -2.0, 1e-12);
  EXPECT_NEAR(result.profile.offset_m, -0.5, 1e-12);
  EXPECT_NEAR(result.reference_geometry(1, 3), -0.5, 1e-12);
}

TEST(FarfieldGuide, SideCountsAreIndependentOfPointOrder) {
  const auto ref = straightReference();
  const Eigen::Vector4d state(0.0, 0.0, 2.0, 0.0);
  neupan_uav::PointMatrix side_first(3, 5);
  side_first << 3.4, 3.8, 4.2, 3.2, 3.6,
                1.0, 1.4, 1.8, 0.0, 0.1,
                2.0, 2.0, 2.0, 2.0, 2.0;
  neupan_uav::PointMatrix center_first(3, 5);
  center_first << 3.2, 3.6, 3.4, 3.8, 4.2,
                  0.0, 0.1, 1.0, 1.4, 1.8,
                  2.0, 2.0, 2.0, 2.0, 2.0;

  neupan_uav::FarfieldGuide side_first_guide(baseConfig());
  neupan_uav::FarfieldGuide center_first_guide(baseConfig());
  const auto side_first_result =
      side_first_guide.apply(ref, side_first, state, 2.0, 1.0, 3);
  const auto center_first_result =
      center_first_guide.apply(ref, center_first, state, 2.0, 1.0, 3);

  EXPECT_EQ(side_first_result.profile.center_count,
            center_first_result.profile.center_count);
  EXPECT_EQ(side_first_result.profile.left_count,
            center_first_result.profile.left_count);
  EXPECT_EQ(side_first_result.profile.right_count,
            center_first_result.profile.right_count);
  EXPECT_NEAR(side_first_result.profile.target_offset_m,
              center_first_result.profile.target_offset_m, 1e-12);
  EXPECT_NEAR(side_first_result.profile.offset_m,
              center_first_result.profile.offset_m, 1e-12);
  EXPECT_NEAR(side_first_result.reference_geometry(1, 3),
              center_first_result.reference_geometry(1, 3), 1e-12);
  EXPECT_EQ(side_first_result.profile.left_count, 3);
  EXPECT_EQ(side_first_result.profile.right_count, 0);
  EXPECT_NEAR(side_first_result.profile.target_offset_m, -2.0, 1e-12);
}

TEST(FarfieldGuide, ReleaseRequiresConfirmCyclesThenDecays) {
  neupan_uav::FarfieldGuide guide(baseConfig());
  const auto ref = straightReference();
  const Eigen::Vector4d state(0.0, 0.0, 2.0, 0.0);
  const neupan_uav::PointMatrix empty = neupan_uav::emptyPointMatrix();

  auto result = guide.apply(ref, centerObstacles(), state, 2.0, 1.0, 3);
  EXPECT_NEAR(result.profile.offset_m, 0.5, 1e-12);

  result = guide.apply(ref, empty, state, 2.0, 1.0, 3);
  EXPECT_EQ(result.profile.release_streak, 1);
  EXPECT_NEAR(result.profile.target_offset_m, 2.0, 1e-12);
  EXPECT_NEAR(result.profile.offset_m, 0.875, 1e-12);

  result = guide.apply(ref, empty, state, 2.0, 1.0, 3);
  EXPECT_EQ(result.profile.release_streak, 2);
  EXPECT_NEAR(result.profile.target_offset_m, 0.0, 1e-12);
  EXPECT_NEAR(result.profile.offset_m, 0.4375, 1e-12);

  result = guide.apply(ref, empty, state, 2.0, 1.0, 3);
  EXPECT_NEAR(result.profile.target_offset_m, 0.0, 1e-12);
  EXPECT_NEAR(result.profile.offset_m, 0.21875, 1e-12);
}

TEST(FarfieldGuide, RangeFarLimitMovesNearWindowBehindLimit) {
  neupan_uav::FarfieldGuideConfig config = baseConfig();
  config.range_backoff = 0.0;
  config.range_far_limit = 2.0;
  neupan_uav::FarfieldGuide guide(config);
  const auto ref = straightReference();
  const Eigen::Vector4d state(0.0, 0.0, 2.0, 0.0);

  const auto result =
      guide.apply(ref, centerObstacles(), state, 2.0, 1.0, 3);

  EXPECT_NEAR(result.profile.near_m, 0.0, 1e-12);
  EXPECT_NEAR(result.profile.far_m, 2.0, 1e-12);
  EXPECT_EQ(result.profile.center_count, 0);
  EXPECT_FALSE(result.profile.active);
}

TEST(FarfieldGuide, SixDimensionalStateUsesYawElementForFallbackFrame) {
  neupan_uav::FarfieldGuide guide(baseConfig());
  Eigen::Matrix<neupan_uav::Scalar, 4, Eigen::Dynamic> ref(4, 4);
  ref << 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0,
         2.0, 3.0, 4.0, 5.0,
         0.0, 0.0, 0.0, 0.0;
  Eigen::Matrix<double, 6, 1> state;
  state << 0.0, 0.0, 2.0, 0.0, 0.0, 1.5707963267948966;
  neupan_uav::PointMatrix points(3, 2);
  points << 0.0, 0.1,
            3.2, 3.6,
            2.0, 2.0;

  const auto result = guide.apply(ref, points, state, 2.0, 1.0, 3);

  EXPECT_TRUE(result.profile.active);
  EXPECT_EQ(result.profile.center_count, 2);
  EXPECT_NEAR(result.profile.offset_m, 0.5, 1e-12);
  EXPECT_NEAR(result.reference_geometry(0, 3), -0.5, 1e-12);
}
