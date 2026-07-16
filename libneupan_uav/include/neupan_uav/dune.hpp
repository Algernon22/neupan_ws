#pragma once

#include "neupan_uav/obstacle_preselector.hpp"
#include "neupan_uav/types.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <limits>
#include <vector>

namespace neupan_uav {

using DuneMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

struct DunePostprocessorConfig {
  int receding = 0;
  int point_dim = 3;
  int edge_dim = 0;
  std::size_t dune_max_num = 0;
  std::size_t select_num = 0;
  double select_nearest_ratio = 0.60;
  double select_temporal_ratio = 0.20;
  double select_diversity_ratio = 0.20;
  DuneMatrix G;
  Eigen::VectorXf h;
};

struct DuneSelectionProfile {
  double nearest_selected_per_step = 0.0F;
  double temporal_selected_per_step = 0.0F;
  double diversity_selected_per_step = 0.0F;
};

struct DuneResult {
  std::size_t selected_count = 0;
  double min_distance = std::numeric_limits<double>::infinity();
  PointMatrix selected_points = emptyPointMatrix();
  std::vector<DuneMatrix> mu_batch;
  std::vector<DuneMatrix> lambda_batch;
  std::vector<DuneMatrix> point_batch;
  DuneSelectionProfile selection_profile;
  PlannerProfile profile;
};

class DunePostprocessor {
 public:
  DunePostprocessor() = default;
  explicit DunePostprocessor(DunePostprocessorConfig config);

  const DunePostprocessorConfig& config() const { return config_; }

  // Compatibility helper for the stage-2 skeleton. Full DUNE postprocessing
  // uses the raw-mu overload below.
  DuneResult process(const PointMatrix& obstacle_points) const;

  DuneResult process(
      const DuneMatrix& raw_mu,
      const std::vector<PointMatrix>& point_flow,
      const std::vector<Eigen::Matrix3d>& rotations,
      const std::vector<PointMatrix>& obstacle_points_by_step,
      const std::vector<unsigned char>* selection_tags = nullptr,
      double inference_sec = 0.0) const;

 private:
  DunePostprocessorConfig config_;
  bool configured_ = false;
};

}  // namespace neupan_uav
