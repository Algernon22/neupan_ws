#pragma once

#include "neupan_uav/types.hpp"

#include <Eigen/Dense>

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace neupan_uav {

using RknnFloatMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

struct RknnMetadata {
  std::string metadata_path;
  std::string rknn_path;
  int receding = 0;
  int dune_max_num = 0;
  int max_points = 0;
  int output_dim = 0;
  std::array<int, 3> input_shape{{0, 0, 0}};
  std::array<int, 3> output_shape{{0, 0, 0}};
  Eigen::Vector3d half_extent = Eigen::Vector3d::Zero();
  Eigen::Vector3d scene_scale = Eigen::Vector3d::Zero();
  Eigen::Vector3d clearance_scale = Eigen::Vector3d::Zero();
  bool do_quantization = false;

  static RknnMetadata load(const std::string& metadata_path);

  void validateRuntime(int receding, int dune_max_num, int output_dim,
                       const Eigen::Vector3d& robot_half_extent) const;
};

struct RknnRunnerConfig {
  std::string metadata_path;
  std::string core_mask = "CORE_0_1_2";
  bool require_device = true;
};

struct RknnProfile {
  double inference_sec = 0.0;
  std::string api_version;
  std::string driver_version;
};

struct PackedRknnInput {
  std::vector<float> values;
  int num_steps = 0;
  int num_points = 0;
};

PackedRknnInput packPointFlow(const RknnMetadata& metadata,
                              const std::vector<RknnFloatMatrix>& point_flow);

RknnFloatMatrix unpackRawMu(const RknnMetadata& metadata,
                            const std::vector<float>& output_full,
                            int num_points);

bool compiledWithRknnRuntime();
bool rknnDeviceAvailable();

class RknnRunner {
 public:
  virtual ~RknnRunner() = default;

  virtual bool available() const { return false; }
  virtual const RknnMetadata& metadata() const = 0;
  virtual const RknnProfile& profile() const = 0;
  virtual RknnFloatMatrix inferRawMu(
      const std::vector<RknnFloatMatrix>& point_flow) = 0;

  RknnFloatMatrix inferRawMu(const std::vector<PointMatrix>& point_flow);
};

class MockRknnRunner final : public RknnRunner {
 public:
  explicit MockRknnRunner(RknnMetadata metadata);

  bool available() const override { return true; }
  const RknnMetadata& metadata() const override { return metadata_; }
  const RknnProfile& profile() const override { return profile_; }
  RknnFloatMatrix inferRawMu(
      const std::vector<RknnFloatMatrix>& point_flow) override;

  void setOutputFull(std::vector<float> output_full);
  const std::vector<float>& lastPackedInput() const { return last_packed_input_; }
  int inferenceCount() const { return inference_count_; }

 private:
  RknnMetadata metadata_;
  RknnProfile profile_;
  std::vector<float> output_full_;
  std::vector<float> last_packed_input_;
  int inference_count_ = 0;
};

class ObsPointNetRknnRunner final : public RknnRunner {
 public:
  explicit ObsPointNetRknnRunner(const RknnRunnerConfig& config);
  ~ObsPointNetRknnRunner() override;

  ObsPointNetRknnRunner(const ObsPointNetRknnRunner&) = delete;
  ObsPointNetRknnRunner& operator=(const ObsPointNetRknnRunner&) = delete;

  bool available() const override;
  const RknnMetadata& metadata() const override;
  const RknnProfile& profile() const override;
  RknnFloatMatrix inferRawMu(
      const std::vector<RknnFloatMatrix>& point_flow) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace neupan_uav
