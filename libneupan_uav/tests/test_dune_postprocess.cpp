#include "neupan_uav/dune.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char* kTestDataDir = NEUPAN_UAV_TEST_DATA_DIR;

struct ParityFixture {
  int receding = 0;
  int point_dim = 0;
  int edge_dim = 0;
  int num_points = 0;
  std::size_t dune_max_num = 0;
  std::size_t select_num = 0;
  double nearest_ratio = 0.0;
  double temporal_ratio = 0.0;
  double diversity_ratio = 0.0;
  neupan_uav::DuneMatrix G;
  Eigen::VectorXf h;
  neupan_uav::DuneMatrix raw_mu;
  std::vector<neupan_uav::PointMatrix> point_flow;
  std::vector<Eigen::Matrix3d> rotations;
  std::vector<neupan_uav::PointMatrix> obstacle_points;
  std::vector<unsigned char> tags;
  double expected_min_distance = 0.0;
  std::size_t expected_selected_count = 0;
  double expected_nearest = 0.0;
  double expected_temporal = 0.0;
  double expected_diversity = 0.0;
  neupan_uav::PointMatrix expected_selected_points;
  std::vector<neupan_uav::DuneMatrix> expected_mu;
  std::vector<neupan_uav::DuneMatrix> expected_lambda;
  std::vector<neupan_uav::DuneMatrix> expected_points;
};

neupan_uav::DunePostprocessorConfig aabbConfig(int receding = 1,
                                               std::size_t select_num = 2) {
  neupan_uav::DunePostprocessorConfig config;
  config.receding = receding;
  config.point_dim = 3;
  config.edge_dim = 6;
  config.dune_max_num = 8;
  config.select_num = select_num;
  config.select_nearest_ratio = 0.60;
  config.select_temporal_ratio = 0.20;
  config.select_diversity_ratio = 0.20;
  config.G.resize(6, 3);
  config.G << 1.0F, 0.0F, 0.0F,
              -1.0F, 0.0F, 0.0F,
              0.0F, 1.0F, 0.0F,
              0.0F, -1.0F, 0.0F,
              0.0F, 0.0F, 1.0F,
              0.0F, 0.0F, -1.0F;
  config.h.resize(6);
  config.h << 0.5F, 0.5F, 0.25F, 0.25F, 0.125F, 0.125F;
  return config;
}

std::vector<Eigen::Matrix3d> identityRotations(int receding) {
  return std::vector<Eigen::Matrix3d>(
      static_cast<std::size_t>(receding + 1), Eigen::Matrix3d::Identity());
}

std::vector<neupan_uav::PointMatrix> repeatPoints(
    const neupan_uav::PointMatrix& points, int receding) {
  return std::vector<neupan_uav::PointMatrix>(
      static_cast<std::size_t>(receding + 1), points);
}

neupan_uav::DuneMatrix rawMuRows(int rows, int edge_dim) {
  neupan_uav::DuneMatrix raw(rows, edge_dim);
  raw.setZero();
  return raw;
}

std::string fixturePath() {
  return std::string(kTestDataDir) + "/dune_postprocess_parity_fixture.txt";
}

std::string readToken(std::istream& stream) {
  std::string token;
  if (!(stream >> token)) {
    throw std::runtime_error("unexpected end of DUNE parity fixture");
  }
  return token;
}

void expectToken(std::istream& stream, const std::string& expected) {
  const std::string actual = readToken(stream);
  if (actual != expected) {
    throw std::runtime_error("expected fixture token '" + expected +
                             "', got '" + actual + "'");
  }
}

int readNamedInt(std::istream& stream, const std::string& name) {
  expectToken(stream, name);
  int value = 0;
  if (!(stream >> value)) {
    throw std::runtime_error("failed to read integer fixture field " + name);
  }
  return value;
}

double readNamedDouble(std::istream& stream, const std::string& name) {
  expectToken(stream, name);
  double value = 0.0;
  if (!(stream >> value)) {
    throw std::runtime_error("failed to read double fixture field " + name);
  }
  return value;
}

neupan_uav::DuneMatrix readMatrix(std::istream& stream,
                                  const std::string& name) {
  expectToken(stream, name);
  int rows = 0;
  int cols = 0;
  if (!(stream >> rows >> cols)) {
    throw std::runtime_error("failed to read matrix shape for " + name);
  }
  neupan_uav::DuneMatrix matrix(rows, cols);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      stream >> matrix(row, col);
    }
  }
  return matrix;
}

Eigen::VectorXf readVector(std::istream& stream, const std::string& name) {
  expectToken(stream, name);
  int size = 0;
  if (!(stream >> size)) {
    throw std::runtime_error("failed to read vector shape for " + name);
  }
  Eigen::VectorXf vector(size);
  for (int i = 0; i < size; ++i) {
    stream >> vector(i);
  }
  return vector;
}

std::vector<neupan_uav::DuneMatrix> readMatrixBatch(
    std::istream& stream, const std::string& name) {
  expectToken(stream, name);
  int steps = 0;
  int rows = 0;
  int cols = 0;
  if (!(stream >> steps >> rows >> cols)) {
    throw std::runtime_error("failed to read batch shape for " + name);
  }
  std::vector<neupan_uav::DuneMatrix> batch;
  batch.reserve(static_cast<std::size_t>(steps));
  for (int step = 0; step < steps; ++step) {
    neupan_uav::DuneMatrix matrix(rows, cols);
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        stream >> matrix(row, col);
      }
    }
    batch.push_back(std::move(matrix));
  }
  return batch;
}

std::vector<neupan_uav::PointMatrix> readPointBatch(
    std::istream& stream, const std::string& name) {
  std::vector<neupan_uav::DuneMatrix> raw = readMatrixBatch(stream, name);
  std::vector<neupan_uav::PointMatrix> batch;
  batch.reserve(raw.size());
  for (const neupan_uav::DuneMatrix& matrix : raw) {
    batch.push_back(matrix.cast<double>());
  }
  return batch;
}

std::vector<Eigen::Matrix3d> readRotationBatch(std::istream& stream) {
  std::vector<neupan_uav::DuneMatrix> raw = readMatrixBatch(stream, "rotations");
  std::vector<Eigen::Matrix3d> rotations;
  rotations.reserve(raw.size());
  for (const neupan_uav::DuneMatrix& matrix : raw) {
    if (matrix.rows() != 3 || matrix.cols() != 3) {
      throw std::runtime_error("rotation fixture entries must be 3x3");
    }
    rotations.push_back(matrix.cast<double>());
  }
  return rotations;
}

std::vector<unsigned char> readTags(std::istream& stream) {
  expectToken(stream, "tags");
  int size = 0;
  if (!(stream >> size)) {
    throw std::runtime_error("failed to read tag vector shape");
  }
  std::vector<unsigned char> tags(static_cast<std::size_t>(size), 0);
  for (int i = 0; i < size; ++i) {
    int tag = 0;
    stream >> tag;
    tags[static_cast<std::size_t>(i)] = static_cast<unsigned char>(tag);
  }
  return tags;
}

ParityFixture loadParityFixture() {
  std::ifstream stream(fixturePath());
  if (!stream) {
    throw std::runtime_error("failed to open DUNE parity fixture: " +
                             fixturePath());
  }
  expectToken(stream, "DUNE_POSTPROCESS_PARITY_V1");

  ParityFixture fixture;
  (void)readNamedInt(stream, "seed");
  fixture.receding = readNamedInt(stream, "receding");
  fixture.point_dim = readNamedInt(stream, "point_dim");
  fixture.edge_dim = readNamedInt(stream, "edge_dim");
  fixture.num_points = readNamedInt(stream, "num_points");
  fixture.dune_max_num =
      static_cast<std::size_t>(readNamedInt(stream, "dune_max_num"));
  fixture.select_num =
      static_cast<std::size_t>(readNamedInt(stream, "select_num"));
  expectToken(stream, "ratios");
  stream >> fixture.nearest_ratio >> fixture.temporal_ratio >>
      fixture.diversity_ratio;
  fixture.G = readMatrix(stream, "G");
  fixture.h = readVector(stream, "h");
  fixture.raw_mu = readMatrix(stream, "raw_mu");
  fixture.point_flow = readPointBatch(stream, "point_flow");
  fixture.rotations = readRotationBatch(stream);
  fixture.obstacle_points = readPointBatch(stream, "obstacle_points");
  fixture.tags = readTags(stream);
  fixture.expected_min_distance =
      readNamedDouble(stream, "expected_min_distance");
  fixture.expected_selected_count = static_cast<std::size_t>(
      readNamedInt(stream, "expected_selected_count"));
  expectToken(stream, "expected_profile");
  stream >> fixture.expected_nearest >> fixture.expected_temporal >>
      fixture.expected_diversity;
  fixture.expected_selected_points =
      readMatrix(stream, "expected_selected_points").cast<double>();
  fixture.expected_mu = readMatrixBatch(stream, "expected_mu");
  fixture.expected_lambda = readMatrixBatch(stream, "expected_lambda");
  fixture.expected_points = readMatrixBatch(stream, "expected_points");
  expectToken(stream, "END");
  return fixture;
}

}  // namespace

TEST(DunePostprocessor, EmptyPointsProduceInfiniteDistance) {
  neupan_uav::DunePostprocessor postprocessor;

  const neupan_uav::DuneResult result =
      postprocessor.process(neupan_uav::emptyPointMatrix());

  EXPECT_EQ(result.selected_count, 0u);
  EXPECT_TRUE(std::isinf(result.min_distance));
  EXPECT_EQ(result.selected_points.cols(), 0);
}

TEST(DunePostprocessor, ComputesNearestPointDistance) {
  neupan_uav::PointMatrix points(3, 3);
  points << 3.0, 1.0, 0.0,
            4.0, 0.0, 0.0,
            0.0, 0.0, 2.0;
  neupan_uav::DunePostprocessor postprocessor;

  const neupan_uav::DuneResult result = postprocessor.process(points);

  EXPECT_EQ(result.selected_count, 3u);
  EXPECT_DOUBLE_EQ(result.min_distance, 1.0);
  EXPECT_TRUE(result.selected_points.isApprox(points));
}

TEST(DunePostprocessor, ProjectsRawMuComputesLambdaAndDistance) {
  const neupan_uav::DunePostprocessorConfig config = aabbConfig(1, 1);
  neupan_uav::DunePostprocessor postprocessor(config);
  neupan_uav::PointMatrix points(3, 1);
  points << 2.0, 0.0, 0.0;
  neupan_uav::DuneMatrix raw = rawMuRows(2, config.edge_dim);
  raw.row(0) << 2.0F, -1.0F, 0.0F, 0.0F, 0.0F, 0.0F;
  raw.row(1) << 0.0F, 0.0F, 3.0F, 0.0F, 0.0F, 0.0F;

  const neupan_uav::DuneResult result = postprocessor.process(
      raw, repeatPoints(points, config.receding),
      identityRotations(config.receding), repeatPoints(points, config.receding));

  ASSERT_EQ(result.mu_batch.size(), 2u);
  ASSERT_EQ(result.lambda_batch.size(), 2u);
  ASSERT_EQ(result.point_batch.size(), 2u);
  ASSERT_EQ(result.mu_batch[0].cols(), 1);
  EXPECT_NEAR(result.mu_batch[0](0, 0), 1.0F, 1.0e-6F);
  EXPECT_NEAR(result.mu_batch[0](1, 0), 0.0F, 1.0e-6F);
  EXPECT_NEAR(result.lambda_batch[0](0, 0), -1.0F, 1.0e-6F);
  EXPECT_NEAR(result.lambda_batch[0](1, 0), 0.0F, 1.0e-6F);
  EXPECT_NEAR(result.lambda_batch[0](2, 0), 0.0F, 1.0e-6F);
  EXPECT_NEAR(result.min_distance, 1.5, 1.0e-6);
  EXPECT_EQ(result.selected_count, 1u);
  EXPECT_EQ(result.profile.dune_selected_count, 1u);
}

TEST(DunePostprocessor, SelectsAllWhenPointCountBelowAndEqualLimit) {
  const neupan_uav::DunePostprocessorConfig config = aabbConfig(0, 3);
  neupan_uav::DunePostprocessor postprocessor(config);

  neupan_uav::PointMatrix two(3, 2);
  two << 2.0, 1.0,
         0.0, 0.0,
         0.0, 0.0;
  neupan_uav::DuneMatrix raw_two = rawMuRows(2, config.edge_dim);
  raw_two.col(0).setOnes();
  neupan_uav::DuneResult below = postprocessor.process(
      raw_two, repeatPoints(two, config.receding),
      identityRotations(config.receding), repeatPoints(two, config.receding));
  EXPECT_EQ(below.selected_count, 2u);
  EXPECT_EQ(below.selected_points.cols(), 2);

  neupan_uav::PointMatrix three(3, 3);
  three << 3.0, 2.0, 1.0,
           0.0, 0.0, 0.0,
           0.0, 0.0, 0.0;
  neupan_uav::DuneMatrix raw_three = rawMuRows(3, config.edge_dim);
  raw_three.col(0).setOnes();
  neupan_uav::DuneResult equal = postprocessor.process(
      raw_three, repeatPoints(three, config.receding),
      identityRotations(config.receding), repeatPoints(three, config.receding));
  EXPECT_EQ(equal.selected_count, 3u);
  EXPECT_EQ(equal.selected_points.cols(), 3);
}

TEST(DunePostprocessor, StableNearestSelectionOrdersByObjectiveDistance) {
  neupan_uav::DunePostprocessorConfig config = aabbConfig(0, 2);
  config.select_nearest_ratio = 1.0;
  config.select_temporal_ratio = 0.0;
  config.select_diversity_ratio = 0.0;
  neupan_uav::DunePostprocessor postprocessor(config);

  neupan_uav::PointMatrix points(3, 4);
  points << 3.0, 1.0, 2.0, 4.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0;
  neupan_uav::DuneMatrix raw = rawMuRows(4, config.edge_dim);
  raw.col(0).setOnes();

  const neupan_uav::DuneResult result = postprocessor.process(
      raw, repeatPoints(points, config.receding),
      identityRotations(config.receding), repeatPoints(points, config.receding));

  ASSERT_EQ(result.selected_count, 2u);
  ASSERT_EQ(result.selected_points.cols(), 2);
  EXPECT_TRUE(result.selected_points.col(0).isApprox(points.col(1)));
  EXPECT_TRUE(result.selected_points.col(1).isApprox(points.col(2)));
  EXPECT_NEAR(result.selection_profile.nearest_selected_per_step, 2.0, 1.0e-12);
}

TEST(DunePostprocessor, HonorsTemporalAndDiversityQuotaTags) {
  neupan_uav::DunePostprocessorConfig config = aabbConfig(0, 3);
  config.select_nearest_ratio = 1.0 / 3.0;
  config.select_temporal_ratio = 1.0 / 3.0;
  config.select_diversity_ratio = 1.0 / 3.0;
  neupan_uav::DunePostprocessor postprocessor(config);

  neupan_uav::PointMatrix points(3, 5);
  points << 1.0, 2.0, 3.0, 4.0, 5.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0;
  neupan_uav::DuneMatrix raw = rawMuRows(5, config.edge_dim);
  raw.col(0).setOnes();
  std::vector<unsigned char> tags(5, 0);
  tags[3] |= neupan_uav::kTemporalSelectionTag;
  tags[4] |= neupan_uav::kDiversitySelectionTag;

  const neupan_uav::DuneResult result = postprocessor.process(
      raw, repeatPoints(points, config.receding),
      identityRotations(config.receding), repeatPoints(points, config.receding),
      &tags);

  ASSERT_EQ(result.selected_count, 3u);
  ASSERT_EQ(result.selected_points.cols(), 3);
  EXPECT_TRUE(result.selected_points.col(0).isApprox(points.col(0)));
  EXPECT_TRUE(result.selected_points.col(1).isApprox(points.col(3)));
  EXPECT_TRUE(result.selected_points.col(2).isApprox(points.col(4)));
  EXPECT_NEAR(result.selection_profile.nearest_selected_per_step, 1.0, 1.0e-12);
  EXPECT_NEAR(result.selection_profile.temporal_selected_per_step, 1.0, 1.0e-12);
  EXPECT_NEAR(result.selection_profile.diversity_selected_per_step, 1.0, 1.0e-12);
}

TEST(DunePostprocessor, MatchesPythonNumpyPostprocessGoldenScenario) {
  neupan_uav::DunePostprocessorConfig config = aabbConfig(1, 2);
  config.select_nearest_ratio = 0.5;
  config.select_temporal_ratio = 0.5;
  config.select_diversity_ratio = 0.0;
  neupan_uav::DunePostprocessor postprocessor(config);

  std::vector<neupan_uav::PointMatrix> point_flow;
  std::vector<neupan_uav::PointMatrix> obstacle_points_by_step;
  point_flow.resize(2);
  obstacle_points_by_step.resize(2);
  point_flow[0].resize(3, 4);
  point_flow[0] << 1.0, 2.0, 3.0, 4.0,
                   0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0;
  point_flow[1].resize(3, 4);
  point_flow[1] << 4.0, 1.0, 3.0, 2.0,
                   0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0;
  obstacle_points_by_step = point_flow;

  std::vector<Eigen::Matrix3d> rotations = identityRotations(config.receding);
  rotations[1] << 0.0, -1.0, 0.0,
                  1.0,  0.0, 0.0,
                  0.0,  0.0, 1.0;

  neupan_uav::DuneMatrix raw = rawMuRows(8, config.edge_dim);
  raw.row(0) << 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F;
  raw.row(1) << 2.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F;
  raw.row(2) << 3.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F;
  raw.row(3) << 4.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F;
  raw.row(4) << 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F;
  raw.row(5) << 0.0F, 0.0F, 2.0F, 0.0F, 0.0F, 0.0F;
  raw.row(6) << 0.0F, 0.0F, 3.0F, 0.0F, 0.0F, 0.0F;
  raw.row(7) << 0.0F, 0.0F, 4.0F, 0.0F, 0.0F, 0.0F;
  std::vector<unsigned char> tags(4, 0);
  tags[3] |= neupan_uav::kTemporalSelectionTag;

  const neupan_uav::DuneResult result = postprocessor.process(
      raw, point_flow, rotations, obstacle_points_by_step, &tags);

  ASSERT_EQ(result.selected_count, 2u);
  ASSERT_EQ(result.mu_batch.size(), 2u);
  ASSERT_EQ(result.lambda_batch.size(), 2u);
  EXPECT_NEAR(result.min_distance, 0.5, 1.0e-6);
  EXPECT_TRUE(result.selected_points.col(0).isApprox(point_flow[0].col(0)));
  EXPECT_TRUE(result.selected_points.col(1).isApprox(point_flow[0].col(3)));
  EXPECT_NEAR(result.mu_batch[0](0, 0), 1.0F, 1.0e-6F);
  EXPECT_NEAR(result.mu_batch[0](0, 1), 1.0F, 1.0e-6F);
  EXPECT_NEAR(result.lambda_batch[0](0, 0), -1.0F, 1.0e-6F);
  EXPECT_NEAR(result.lambda_batch[0](0, 1), -1.0F, 1.0e-6F);
  EXPECT_NEAR(result.mu_batch[1](2, 0), 1.0F, 1.0e-6F);
  EXPECT_NEAR(result.mu_batch[1](2, 1), 1.0F, 1.0e-6F);
  EXPECT_NEAR(result.lambda_batch[1](0, 0), 1.0F, 1.0e-6F);
  EXPECT_NEAR(result.lambda_batch[1](1, 0), 0.0F, 1.0e-6F);
  EXPECT_NEAR(result.selection_profile.nearest_selected_per_step, 1.0,
              1.0e-12);
  EXPECT_NEAR(result.selection_profile.temporal_selected_per_step, 1.0,
              1.0e-12);
}

TEST(DunePostprocessor, MatchesPythonGeneratedRandomParityFixture) {
  const ParityFixture fixture = loadParityFixture();
  neupan_uav::DunePostprocessorConfig config;
  config.receding = fixture.receding;
  config.point_dim = fixture.point_dim;
  config.edge_dim = fixture.edge_dim;
  config.dune_max_num = fixture.dune_max_num;
  config.select_num = fixture.select_num;
  config.select_nearest_ratio = fixture.nearest_ratio;
  config.select_temporal_ratio = fixture.temporal_ratio;
  config.select_diversity_ratio = fixture.diversity_ratio;
  config.G = fixture.G;
  config.h = fixture.h;
  neupan_uav::DunePostprocessor postprocessor(config);

  const neupan_uav::DuneResult result = postprocessor.process(
      fixture.raw_mu, fixture.point_flow, fixture.rotations,
      fixture.obstacle_points, &fixture.tags);

  ASSERT_EQ(result.selected_count, fixture.expected_selected_count);
  EXPECT_NEAR(result.min_distance, fixture.expected_min_distance, 2.0e-5);
  EXPECT_NEAR(result.selection_profile.nearest_selected_per_step,
              fixture.expected_nearest, 1.0e-6);
  EXPECT_NEAR(result.selection_profile.temporal_selected_per_step,
              fixture.expected_temporal, 1.0e-6);
  EXPECT_NEAR(result.selection_profile.diversity_selected_per_step,
              fixture.expected_diversity, 1.0e-6);
  ASSERT_EQ(result.selected_points.rows(),
            fixture.expected_selected_points.rows());
  ASSERT_EQ(result.selected_points.cols(),
            fixture.expected_selected_points.cols());
  EXPECT_TRUE(result.selected_points.isApprox(fixture.expected_selected_points,
                                              2.0e-5));

  ASSERT_EQ(result.mu_batch.size(), fixture.expected_mu.size());
  ASSERT_EQ(result.lambda_batch.size(), fixture.expected_lambda.size());
  ASSERT_EQ(result.point_batch.size(), fixture.expected_points.size());
  for (std::size_t step = 0; step < result.mu_batch.size(); ++step) {
    ASSERT_EQ(result.mu_batch[step].rows(), fixture.expected_mu[step].rows());
    ASSERT_EQ(result.mu_batch[step].cols(), fixture.expected_mu[step].cols());
    ASSERT_EQ(result.lambda_batch[step].rows(),
              fixture.expected_lambda[step].rows());
    ASSERT_EQ(result.lambda_batch[step].cols(),
              fixture.expected_lambda[step].cols());
    ASSERT_EQ(result.point_batch[step].rows(),
              fixture.expected_points[step].rows());
    ASSERT_EQ(result.point_batch[step].cols(),
              fixture.expected_points[step].cols());
    EXPECT_TRUE(result.mu_batch[step].isApprox(fixture.expected_mu[step],
                                               2.0e-5F));
    EXPECT_TRUE(result.lambda_batch[step].isApprox(
        fixture.expected_lambda[step], 2.0e-5F));
    EXPECT_TRUE(result.point_batch[step].isApprox(fixture.expected_points[step],
                                                  2.0e-5F));
  }
}

TEST(DunePostprocessor, EmptyRawMuBatchDoesNotRequireHardware) {
  const neupan_uav::DunePostprocessorConfig config = aabbConfig(2, 2);
  neupan_uav::DunePostprocessor postprocessor(config);
  neupan_uav::DuneMatrix raw(0, config.edge_dim);

  const neupan_uav::DuneResult result = postprocessor.process(
      raw, repeatPoints(neupan_uav::emptyPointMatrix(), config.receding),
      identityRotations(config.receding),
      repeatPoints(neupan_uav::emptyPointMatrix(), config.receding));

  EXPECT_EQ(result.selected_count, 0u);
  EXPECT_TRUE(std::isinf(result.min_distance));
  EXPECT_EQ(result.profile.dune_selected_count, 0u);
}

TEST(DunePostprocessor, RejectsBadConfigurationAndRuntimeShapes) {
  neupan_uav::DunePostprocessorConfig bad = aabbConfig(1, 1);
  bad.select_nearest_ratio = 0.8;
  bad.select_temporal_ratio = 0.3;
  EXPECT_THROW(
      {
        neupan_uav::DunePostprocessor processor{bad};
        (void)processor;
      },
      std::invalid_argument);

  const neupan_uav::DunePostprocessorConfig config = aabbConfig(1, 1);
  neupan_uav::DunePostprocessor postprocessor(config);
  neupan_uav::PointMatrix points(3, 1);
  points << 1.0, 0.0, 0.0;
  neupan_uav::DuneMatrix raw_bad_rows(1, config.edge_dim);
  raw_bad_rows.setZero();

  EXPECT_THROW(
      postprocessor.process(raw_bad_rows, repeatPoints(points, config.receding),
                            identityRotations(config.receding),
                            repeatPoints(points, config.receding)),
      std::invalid_argument);

  neupan_uav::DuneMatrix raw(2, config.edge_dim);
  raw.setZero();
  std::vector<unsigned char> bad_tags(2, 0);
  EXPECT_THROW(
      postprocessor.process(raw, repeatPoints(points, config.receding),
                            identityRotations(config.receding),
                            repeatPoints(points, config.receding), &bad_tags),
      std::invalid_argument);
}
