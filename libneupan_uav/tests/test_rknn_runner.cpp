#include "neupan_uav/rknn_runner.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

fs::path tempDir() {
  fs::path dir = fs::temp_directory_path() / "neupan_uav_rknn_runner_tests";
  fs::create_directories(dir);
  return dir;
}

fs::path writeMetadata(const fs::path& dir, const std::string& name,
                       const std::string& extra = "",
                       const std::string& overrides = "") {
  const fs::path model = dir / (name + ".rknn");
  {
    std::ofstream model_stream(model, std::ios::binary);
    model_stream << "fake";
  }

  const fs::path metadata = dir / (name + ".metadata.json");
  std::ofstream stream(metadata);
  stream << "{\n"
         << "  \"rknn\": \"" << model.string() << "\",\n"
         << "  \"receding\": 1,\n"
         << "  \"dune_max_num\": 3,\n"
         << "  \"max_points\": 6,\n"
         << "  \"output_dim\": 6,\n"
         << "  \"input_shape\": [1, 6, 3],\n"
         << "  \"output_shape\": [1, 6, 6],\n"
         << "  \"half_extent\": [0.32, 0.32, 0.27],\n"
         << "  \"scene_scale\": [40.0, 40.0, 10.0],\n"
         << "  \"clearance_scale\": [2.0, 2.0, 1.0],\n"
         << "  \"do_quantization\": false";
  if (!extra.empty()) {
    stream << ",\n" << extra;
  }
  if (!overrides.empty()) {
    stream << "\n" << overrides;
  }
  stream << "\n}\n";
  return metadata;
}

void expectMatrixNear(const neupan_uav::RknnFloatMatrix& actual,
                      const neupan_uav::RknnFloatMatrix& expected) {
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.cols(), expected.cols());
  for (Eigen::Index row = 0; row < actual.rows(); ++row) {
    for (Eigen::Index col = 0; col < actual.cols(); ++col) {
      EXPECT_FLOAT_EQ(actual(row, col), expected(row, col));
    }
  }
}

std::vector<neupan_uav::RknnFloatMatrix> samplePointFlow() {
  std::vector<neupan_uav::RknnFloatMatrix> point_flow(2);
  point_flow[0].resize(3, 2);
  point_flow[0] << 1.0F, 2.0F,
                   3.0F, 4.0F,
                   5.0F, 6.0F;
  point_flow[1].resize(3, 2);
  point_flow[1] << 7.0F, 8.0F,
                   9.0F, 10.0F,
                   11.0F, 12.0F;
  return point_flow;
}

neupan_uav::RknnRuntimeContract sampleContract() {
  neupan_uav::RknnRuntimeContract contract;
  contract.receding = 1;
  contract.dune_max_num = 3;
  contract.output_dim = 6;
  contract.body_half_extent = Eigen::Vector3d(0.32, 0.32, 0.27);
  return contract;
}

}  // namespace

TEST(RknnMetadata, LoadsAndValidatesRuntime) {
  const fs::path metadata_path = writeMetadata(tempDir(), "valid");

  const neupan_uav::RknnMetadata metadata =
      neupan_uav::RknnMetadata::load(metadata_path.string());

  EXPECT_EQ(metadata.receding, 1);
  EXPECT_EQ(metadata.dune_max_num, 3);
  EXPECT_EQ(metadata.max_points, 6);
  EXPECT_EQ(metadata.output_dim, 6);
  EXPECT_EQ(metadata.input_shape[0], 1);
  EXPECT_EQ(metadata.input_shape[1], 6);
  EXPECT_EQ(metadata.input_shape[2], 3);
  metadata.validateRuntime(sampleContract());
}

TEST(RknnMetadata, RejectsQuantizedModels) {
  const fs::path dir = tempDir();
  const fs::path model = dir / "quantized.rknn";
  {
    std::ofstream model_stream(model, std::ios::binary);
    model_stream << "fake";
  }
  const fs::path metadata = dir / "quantized.metadata.json";
  std::ofstream stream(metadata);
  stream << "{\n"
         << "  \"rknn\": \"" << model.string() << "\",\n"
         << "  \"receding\": 1,\n"
         << "  \"dune_max_num\": 3,\n"
         << "  \"max_points\": 6,\n"
         << "  \"output_dim\": 6,\n"
         << "  \"input_shape\": [1, 6, 3],\n"
         << "  \"output_shape\": [1, 6, 6],\n"
         << "  \"half_extent\": [0.32, 0.32, 0.27],\n"
         << "  \"scene_scale\": [40.0, 40.0, 10.0],\n"
         << "  \"clearance_scale\": [2.0, 2.0, 1.0],\n"
         << "  \"do_quantization\": true\n"
         << "}\n";

  EXPECT_THROW((void)neupan_uav::RknnMetadata::load(metadata.string()),
               std::invalid_argument);
}

TEST(RknnMetadata, RejectsRuntimeMismatch) {
  const neupan_uav::RknnMetadata metadata =
      neupan_uav::RknnMetadata::load(writeMetadata(tempDir(), "mismatch").string());

  neupan_uav::RknnRuntimeContract contract = sampleContract();
  contract.receding = 2;
  EXPECT_THROW(metadata.validateRuntime(contract), std::invalid_argument);

  contract = sampleContract();
  contract.dune_max_num = 4;
  EXPECT_THROW(metadata.validateRuntime(contract), std::invalid_argument);

  contract = sampleContract();
  contract.output_dim = 5;
  EXPECT_THROW(metadata.validateRuntime(contract), std::invalid_argument);

  contract = sampleContract();
  contract.body_half_extent = Eigen::Vector3d(0.30, 0.32, 0.27);
  EXPECT_THROW(metadata.validateRuntime(contract), std::invalid_argument);
}

TEST(RknnMetadata, RuntimeMismatchMessageIncludesValuesAndPath) {
  const neupan_uav::RknnMetadata metadata =
      neupan_uav::RknnMetadata::load(writeMetadata(tempDir(), "message").string());
  neupan_uav::RknnRuntimeContract contract = sampleContract();
  contract.body_half_extent = Eigen::Vector3d(0.30, 0.32, 0.27);

  try {
    metadata.validateRuntime(contract);
    FAIL() << "validateRuntime should reject mismatched half_extent";
  } catch (const std::invalid_argument& exc) {
    const std::string message = exc.what();
    EXPECT_NE(message.find("RKNN half_extent mismatch"), std::string::npos);
    EXPECT_NE(message.find("model=[0.32, 0.32, 0.27]"), std::string::npos);
    EXPECT_NE(message.find("runtime=[0.3, 0.32, 0.27]"), std::string::npos);
    EXPECT_NE(message.find("metadata_path="), std::string::npos);
  }
}

TEST(RknnPacking, PacksAndUnpacksLikePythonRunner) {
  const neupan_uav::RknnMetadata metadata =
      neupan_uav::RknnMetadata::load(writeMetadata(tempDir(), "packing").string());
  const std::vector<neupan_uav::RknnFloatMatrix> point_flow =
      samplePointFlow();

  const neupan_uav::PackedRknnInput packed =
      neupan_uav::packPointFlow(metadata, point_flow);

  ASSERT_EQ(packed.values.size(), 18U);
  const std::vector<float> expected_input{
      1.0F, 3.0F, 5.0F,
      2.0F, 4.0F, 6.0F,
      0.0F, 0.0F, 0.0F,
      7.0F, 9.0F, 11.0F,
      8.0F, 10.0F, 12.0F,
      0.0F, 0.0F, 0.0F};
  EXPECT_EQ(packed.values, expected_input);

  std::vector<float> output_full(36);
  std::iota(output_full.begin(), output_full.end(), 0.0F);
  const neupan_uav::RknnFloatMatrix raw_mu =
      neupan_uav::unpackRawMu(metadata, output_full, 2);

  neupan_uav::RknnFloatMatrix expected(4, 6);
  expected << 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
              6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F,
              18.0F, 19.0F, 20.0F, 21.0F, 22.0F, 23.0F,
              24.0F, 25.0F, 26.0F, 27.0F, 28.0F, 29.0F;
  expectMatrixNear(raw_mu, expected);
}

TEST(MockRknnRunner, SkipsInferenceForEmptyPointFlow) {
  neupan_uav::MockRknnRunner runner(
      neupan_uav::RknnMetadata::load(writeMetadata(tempDir(), "empty").string()));
  std::vector<neupan_uav::RknnFloatMatrix> point_flow(2);
  point_flow[0].resize(3, 0);
  point_flow[1].resize(3, 0);

  const neupan_uav::RknnFloatMatrix raw_mu = runner.inferRawMu(point_flow);

  EXPECT_EQ(raw_mu.rows(), 0);
  EXPECT_EQ(raw_mu.cols(), 6);
  EXPECT_EQ(runner.inferenceCount(), 0);
  EXPECT_DOUBLE_EQ(runner.profile().inference_sec, 0.0);
}

TEST(MockRknnRunner, ReturnsConfiguredOutputForValidPoints) {
  neupan_uav::MockRknnRunner runner(
      neupan_uav::RknnMetadata::load(writeMetadata(tempDir(), "mock").string()));
  std::vector<float> output_full(36);
  std::iota(output_full.begin(), output_full.end(), 0.0F);
  runner.setOutputFull(output_full);

  const neupan_uav::RknnFloatMatrix raw_mu =
      runner.inferRawMu(samplePointFlow());

  EXPECT_EQ(runner.inferenceCount(), 1);
  ASSERT_EQ(runner.lastPackedInput().size(), 18U);
  neupan_uav::RknnFloatMatrix expected(4, 6);
  expected << 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
              6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F,
              18.0F, 19.0F, 20.0F, 21.0F, 22.0F, 23.0F,
              24.0F, 25.0F, 26.0F, 27.0F, 28.0F, 29.0F;
  expectMatrixNear(raw_mu, expected);
}

TEST(ObsPointNetRknnRunner, CompileFlagMatchesBuild) {
#ifdef NEUPAN_UAV_WITH_RKNN
  EXPECT_TRUE(neupan_uav::compiledWithRknnRuntime());
#else
  EXPECT_FALSE(neupan_uav::compiledWithRknnRuntime());
  neupan_uav::RknnRunnerConfig config;
  config.metadata_path = writeMetadata(tempDir(), "no_rknn").string();
  EXPECT_THROW(neupan_uav::ObsPointNetRknnRunner runner(config),
               std::runtime_error);
#endif
}

TEST(ObsPointNetRknnRunner, HardwareSmokeWhenAvailable) {
  if (!neupan_uav::compiledWithRknnRuntime() ||
      !neupan_uav::rknnDeviceAvailable()) {
    GTEST_SKIP() << "RKNN runtime support or NPU device is not available";
  }

  const char* metadata_env = std::getenv("NEUPAN_RKNN_METADATA");
  if (metadata_env == nullptr || std::strlen(metadata_env) == 0) {
    GTEST_SKIP() << "NEUPAN_RKNN_METADATA is not set";
  }
  const fs::path metadata_path = metadata_env;
  if (!fs::is_regular_file(metadata_path)) {
    GTEST_SKIP() << "default RKNN model metadata is not available";
  }

  neupan_uav::RknnRunnerConfig config;
  config.metadata_path = metadata_path.string();
  config.core_mask = "CORE_0";
  config.require_device = true;
  neupan_uav::RknnMetadata metadata =
      neupan_uav::RknnMetadata::load(metadata_path.string());
  neupan_uav::RknnRuntimeContract contract;
  contract.receding = metadata.receding;
  contract.dune_max_num = metadata.dune_max_num;
  contract.output_dim = metadata.output_dim;
  contract.body_half_extent = metadata.half_extent;
  config.expected_runtime = contract;
  neupan_uav::ObsPointNetRknnRunner runner(config);

  std::vector<neupan_uav::RknnFloatMatrix> point_flow(
      static_cast<std::size_t>(runner.metadata().receding + 1));
  for (neupan_uav::RknnFloatMatrix& points : point_flow) {
    points.resize(3, 1);
    points << 1.0F, 0.0F, 0.2F;
  }

  const neupan_uav::RknnFloatMatrix raw_mu = runner.inferRawMu(point_flow);
  EXPECT_EQ(raw_mu.rows(), runner.metadata().receding + 1);
  EXPECT_EQ(raw_mu.cols(), runner.metadata().output_dim);
  EXPECT_GE(runner.profile().inference_sec, 0.0);
}
