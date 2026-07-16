#pragma once

#include "neupan_uav/config.hpp"
#include "neupan_uav/nrmp.hpp"
#include "neupan_uav/dune.hpp"
#include "neupan_uav/obstacle_preselector.hpp"
#include "neupan_uav/rknn_runner.hpp"
#include "neupan_uav/types.hpp"

#include <memory>
#include <limits>
#include <vector>

namespace neupan_uav {

struct PanInput {
  Control seed_control = Control::Zero();
  Control desired_control = Control::Zero();
  PointMatrix obstacle_points = emptyPointMatrix();
  PointMatrix obstacle_velocities = emptyPointMatrix();
  Trajectory nominal_states;
  Eigen::MatrixXd nominal_controls;
  Trajectory reference_states;
  Eigen::MatrixXd reference_controls;
  Eigen::MatrixXd attitude_horizon;
  std::vector<unsigned char> selection_tags;
};

struct PanOutput {
  Control command = Control::Zero();
  Trajectory trajectory;
  Trajectory reference;
  Eigen::MatrixXd control_trajectory;
  Eigen::RowVectorXd nominal_distance;
  double min_distance = std::numeric_limits<double>::infinity();
  PlannerProfile profile;
};

class PAN {
 public:
  PAN();
  explicit PAN(PanConfig config);

  void setRknnRunner(std::unique_ptr<RknnRunner> runner);

  PanOutput forward(const PanInput& input);

 private:
  DuneResult runDune(const PanInput& input);
  NrmpInput buildNrmpInput(
      const PanInput& input,
      const DuneResult* dune_result) const;
  bool stopCriteria(const NrmpResult& nrmp_result,
                    const DuneResult* dune_result);
  void resetIterationState();
  RknnRuntimeContract rknnRuntimeContract() const;
  void validateRknnRunner(const RknnRunner& runner) const;

  PanConfig config_;
  NRMP nrmp_;
  DunePostprocessor dune_;
  PointFlowBuilder point_flow_;
  std::unique_ptr<RknnRunner> rknn_runner_;
  Trajectory previous_nominal_states_;
  Eigen::MatrixXd previous_nominal_controls_;
  std::vector<DuneMatrix> previous_mu_batch_;
  std::vector<DuneMatrix> previous_lambda_batch_;
  bool has_previous_iteration_ = false;
};

}  // namespace neupan_uav
