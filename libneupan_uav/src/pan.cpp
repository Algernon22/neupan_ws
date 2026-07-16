#include "neupan_uav/pan.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace neupan_uav {

namespace {

using Clock = std::chrono::steady_clock;

double elapsedSeconds(const Clock::time_point& start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

bool finiteMatrix(const Eigen::MatrixXd& matrix) {
  for (Eigen::Index i = 0; i < matrix.size(); ++i) {
    if (!std::isfinite(matrix.data()[i])) return false;
  }
  return true;
}

bool finiteMatrixFloat(const DuneMatrix& matrix) {
  for (Eigen::Index i = 0; i < matrix.size(); ++i) {
    if (!std::isfinite(matrix.data()[i])) return false;
  }
  return true;
}

void validateTrajectory(const Eigen::MatrixXd& matrix, int rows, int cols,
                        const char* name) {
  if (matrix.rows() != rows || matrix.cols() != cols) {
    throw std::invalid_argument(std::string(name) + " must have shape " +
                                std::to_string(rows) + "x" +
                                std::to_string(cols));
  }
  if (!finiteMatrix(matrix)) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

Eigen::VectorXd defaultStateWeights(int state_dim) {
  return Eigen::VectorXd::Ones(std::max(1, state_dim));
}

double matrixDiffNorm(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    return std::numeric_limits<double>::infinity();
  }
  return (a - b).norm();
}

std::pair<double, double> duneDiff(
    const DuneResult& current,
    const std::vector<DuneMatrix>& previous_mu,
    const std::vector<DuneMatrix>& previous_lambda,
    int max_num) {
  if (current.mu_batch.empty() || previous_mu.empty() ||
      current.lambda_batch.empty() || previous_lambda.empty()) {
    return {0.0, 0.0};
  }
  const std::size_t steps = std::min(current.mu_batch.size(), previous_mu.size());
  double mu_sq = 0.0;
  double lambda_sq = 0.0;
  int effect_num = 0;
  for (std::size_t step = 0; step < steps; ++step) {
    const DuneMatrix& mu = current.mu_batch[step];
    const DuneMatrix& prev_mu = previous_mu[step];
    const DuneMatrix& lambda = current.lambda_batch[step];
    const DuneMatrix& prev_lambda = previous_lambda[step];
    const int cols = std::min(
        {static_cast<int>(mu.cols()), static_cast<int>(prev_mu.cols()),
         static_cast<int>(lambda.cols()), static_cast<int>(prev_lambda.cols()),
         max_num});
    if (cols <= 0) continue;
    if (mu.rows() != prev_mu.rows() || lambda.rows() != prev_lambda.rows()) {
      return {std::numeric_limits<double>::infinity(),
              std::numeric_limits<double>::infinity()};
    }
    mu_sq += (mu.leftCols(cols) - prev_mu.leftCols(cols))
                 .template cast<double>()
                 .squaredNorm();
    lambda_sq += (lambda.leftCols(cols) - prev_lambda.leftCols(cols))
                     .template cast<double>()
                     .squaredNorm();
    effect_num = std::max(effect_num, cols);
  }
  if (effect_num <= 0) return {0.0, 0.0};
  return {std::sqrt(mu_sq) / static_cast<double>(effect_num),
          std::sqrt(lambda_sq) / static_cast<double>(effect_num)};
}

}  // namespace

PAN::PAN()
    : config_(),
      nrmp_(),
      dune_(),
      point_flow_(PointFlowConfig{}) {}

PAN::PAN(PanConfig config)
    : config_(std::move(config)),
      nrmp_(config_.has_nrmp_config ? NRMP(config_.nrmp) : NRMP()),
      dune_(config_.has_dune_config ? DunePostprocessor(config_.dune)
                                    : DunePostprocessor()),
      point_flow_(config_.point_flow) {
  if (config_.rknn_mode == RknnRunnerMode::kRuntime) {
    RknnRunnerConfig runner_config;
    runner_config.metadata_path = config_.rknn_metadata_path;
    runner_config.core_mask = config_.rknn_core_mask;
    runner_config.require_device = config_.rknn_require_device;
    runner_config.expected_runtime = rknnRuntimeContract();
    rknn_runner_ = std::make_unique<ObsPointNetRknnRunner>(runner_config);
  }
}

void PAN::setRknnRunner(std::unique_ptr<RknnRunner> runner) {
  if (runner != nullptr) {
    validateRknnRunner(*runner);
  }
  rknn_runner_ = std::move(runner);
}

bool PAN::hasFullConfig() const {
  return config_.has_nrmp_config && nrmp_.hasBackend();
}

RknnRuntimeContract PAN::rknnRuntimeContract() const {
  RknnRuntimeContract contract;
  contract.receding = config_.receding;
  contract.dune_max_num = config_.dune_max_num;
  contract.output_dim = config_.dune.edge_dim;
  contract.body_half_extent = config_.point_flow.body_half_extent;
  return contract;
}

void PAN::validateRknnRunner(const RknnRunner& runner) const {
  runner.metadata().validateRuntime(rknnRuntimeContract());
}

PanOutput PAN::forward(const PanInput& input) {
  if (!hasFullConfig()) {
    NrmpInput nrmp_input;
    nrmp_input.seed_control = input.seed_control;
    nrmp_input.desired_control = input.desired_control;
    const NrmpResult nrmp = nrmp_.solve(nrmp_input);

    PanOutput out;
    out.command = nrmp.control;
    out.trajectory = input.nominal_states;
    out.reference = input.reference_states;
    out.control_trajectory = input.nominal_controls;
    if (input.nominal_controls.cols() > 0) {
      out.nominal_distance =
          Eigen::RowVectorXd::Zero(input.nominal_controls.cols());
    }
    out.profile.osqp_status = nrmp.status;
    out.profile.osqp_iteration_count = nrmp.iterations;
    out.profile.dune_selected_count =
        static_cast<std::size_t>(input.obstacle_points.cols());
    return out;
  }

  const int state_dim = config_.nrmp.state_dim;
  const int control_dim = config_.nrmp.control_dim;
  const int receding = config_.receding;
  validateTrajectory(input.nominal_states, state_dim, receding + 1,
                     "PanInput::nominal_states");
  validateTrajectory(input.nominal_controls, control_dim, receding,
                     "PanInput::nominal_controls");
  validateTrajectory(input.reference_states, state_dim, receding + 1,
                     "PanInput::reference_states");
  validateTrajectory(input.reference_controls, control_dim, receding,
                     "PanInput::reference_controls");

  const bool has_obstacles =
      input.obstacle_points.rows() == 3 && input.obstacle_points.cols() > 0 &&
      config_.nrmp_max_num > 0 && config_.dune_max_num > 0;
  if (!has_obstacles) resetObstacleState();

  PanOutput out;
  out.trajectory = input.nominal_states;
  out.reference = input.reference_states;
  out.control_trajectory = input.nominal_controls;
  out.profile.pan_iteration_limit = config_.iter_num;

  PanInput current = input;
  DuneResult dune_result;
  bool has_dune_result = false;
  NrmpResult nrmp_result;
  resetIterationState();

  const auto total_start = Clock::now();
  for (int iteration = 0; iteration < std::max(1, config_.iter_num);
       ++iteration) {
    if (has_obstacles) {
      const auto dune_start = Clock::now();
      dune_result = runDune(current);
      has_dune_result = true;
      out.profile.dune_sec += elapsedSeconds(dune_start);
      out.profile.dune_inference_sec += dune_result.profile.dune_inference_sec;
      out.profile.dune_select_sec += dune_result.profile.dune_select_sec;
      out.profile.dune_selected_count = dune_result.profile.dune_selected_count;
      out.profile.nearest_selected = dune_result.profile.nearest_selected;
      out.profile.continuity_selected =
          dune_result.profile.continuity_selected;
      out.profile.diversity_selected = dune_result.profile.diversity_selected;
      out.min_distance = dune_result.min_distance;
      out.dune_points = dune_result.selected_points;
    }

    const auto nrmp_start = Clock::now();
    nrmp_result = nrmp_.solve(
        buildNrmpInput(current, has_dune_result ? &dune_result : nullptr));
    out.profile.nrmp_sec += elapsedSeconds(nrmp_start);
    out.profile.osqp_solve_sec += nrmp_result.solve_sec;
    out.profile.osqp_run_time_sec += nrmp_result.run_time_sec;
    out.profile.osqp_iteration_count += nrmp_result.iterations;
    out.profile.osqp_status = nrmp_result.status;
    out.profile.pan_iterations = iteration + 1;

    out.command = nrmp_result.control;
    out.trajectory = nrmp_result.state_trajectory;
    out.control_trajectory = nrmp_result.control_trajectory;
    out.nominal_distance = nrmp_result.nominal_distance;
    current.nominal_states = nrmp_result.state_trajectory;
    current.nominal_controls = nrmp_result.control_trajectory;

    if (stopCriteria(nrmp_result, has_dune_result ? &dune_result : nullptr)) {
      break;
    }
  }

  out.profile.forward_sec = elapsedSeconds(total_start);
  if (has_dune_result) {
    out.nrmp_points = dune_result.selected_points;
  }
  return out;
}

DuneResult PAN::runDune(const PanInput& input) {
  if (!config_.has_dune_config) {
    DunePostprocessor fallback;
    return fallback.process(input.obstacle_points);
  }
  if (rknn_runner_ == nullptr) {
    throw std::runtime_error(
        "PAN requires an RKNN runner when obstacle DUNE is enabled");
  }

  const PointFlowResult flow = point_flow_.generate(
      input.nominal_states, input.obstacle_points, input.obstacle_velocities,
      input.attitude_horizon);
  const RknnFloatMatrix raw_mu = rknn_runner_->inferRawMu(flow.point_flow);
  return dune_.process(raw_mu, flow.point_flow, flow.rotations,
                       flow.obstacle_points_by_step,
                       input.selection_tags.empty() ? nullptr
                                                    : &input.selection_tags,
                       rknn_runner_->profile().inference_sec);
}

NrmpInput PAN::buildNrmpInput(const PanInput& input,
                              const DuneResult* dune_result) const {
  NrmpInput nrmp_input;
  nrmp_input.seed_control = input.seed_control;
  nrmp_input.desired_control = input.desired_control;
  nrmp_input.nominal_states = input.nominal_states;
  nrmp_input.reference_states = input.reference_states;
  nrmp_input.reference_controls = input.reference_controls;
  nrmp_input.state_weights =
      config_.state_weights.size() == 0
          ? defaultStateWeights(config_.nrmp.state_dim)
          : config_.state_weights;
  nrmp_input.p_u = config_.p_u;
  nrmp_input.eta = config_.eta;
  nrmp_input.d_min = config_.d_min;
  nrmp_input.d_max = config_.d_max;
  nrmp_input.ro_obs = config_.ro_obs;
  nrmp_input.bk = config_.bk;
  nrmp_input.smooth_du =
      config_.nrmp.enable_control_smoothing ? config_.smooth_du : 0.0;
  nrmp_input.smooth_u0 =
      config_.nrmp.enable_control_smoothing ? config_.smooth_u0 : 0.0;

  if (config_.nrmp.no_obs || config_.nrmp.max_num <= 0) {
    return nrmp_input;
  }

  const int T = config_.nrmp.receding;
  const int max_num = config_.nrmp.max_num;
  const int point_dim = config_.nrmp.point_dim;
  const double inactive_margin = std::max(config_.d_max, 0.0) + 1.0;
  nrmp_input.fa_batch =
      Eigen::MatrixXd::Zero(T * max_num, point_dim);
  nrmp_input.fb_batch =
      Eigen::MatrixXd::Constant(T, max_num, -inactive_margin);

  if (dune_result == nullptr || dune_result->mu_batch.size() <= 1 ||
      dune_result->lambda_batch.size() <= 1 ||
      dune_result->point_batch.size() <= 1) {
    return nrmp_input;
  }

  const int time_count = std::min(
      {T, static_cast<int>(dune_result->mu_batch.size()) - 1,
       static_cast<int>(dune_result->lambda_batch.size()) - 1,
       static_cast<int>(dune_result->point_batch.size()) - 1});
  for (int t = 0; t < time_count; ++t) {
    const DuneMatrix& mu = dune_result->mu_batch[static_cast<std::size_t>(t + 1)];
    const DuneMatrix& lambda =
        dune_result->lambda_batch[static_cast<std::size_t>(t + 1)];
    const DuneMatrix& points =
        dune_result->point_batch[static_cast<std::size_t>(t + 1)];
    if (!finiteMatrixFloat(mu) || !finiteMatrixFloat(lambda) ||
        !finiteMatrixFloat(points)) {
      throw std::invalid_argument("DUNE result contains non-finite values");
    }
    const int point_count =
        std::min({max_num, static_cast<int>(mu.cols()),
                  static_cast<int>(lambda.cols()),
                  static_cast<int>(points.cols())});
    for (int k = 0; k < point_count; ++k) {
      const int row = t * max_num + k;
      for (int d = 0; d < point_dim; ++d) {
        nrmp_input.fa_batch(row, d) = lambda(d, k);
      }
      const Eigen::VectorXf h = config_.dune.h;
      const double lambda_dot_point =
          lambda.col(k).head(point_dim).dot(points.col(k).head(point_dim));
      const double mu_dot_h = mu.col(k).dot(h);
      nrmp_input.fb_batch(t, k) = lambda_dot_point + mu_dot_h;
    }
  }
  return nrmp_input;
}

bool PAN::stopCriteria(const NrmpResult& nrmp_result,
                       const DuneResult* dune_result) {
  if (!has_previous_iteration_) {
    previous_nominal_states_ = nrmp_result.state_trajectory;
    previous_nominal_controls_ = nrmp_result.control_trajectory;
    previous_mu_batch_ =
        dune_result == nullptr ? std::vector<DuneMatrix>() : dune_result->mu_batch;
    previous_lambda_batch_ = dune_result == nullptr ? std::vector<DuneMatrix>()
                                                    : dune_result->lambda_batch;
    has_previous_iteration_ = true;
    return false;
  }

  const double state_diff =
      matrixDiffNorm(nrmp_result.state_trajectory, previous_nominal_states_);
  const double control_diff =
      matrixDiffNorm(nrmp_result.control_trajectory, previous_nominal_controls_);
  double diff = state_diff * state_diff + control_diff * control_diff;

  if (dune_result != nullptr && !dune_result->mu_batch.empty() &&
      !previous_mu_batch_.empty()) {
    const auto [mu_diff, lambda_diff] =
        duneDiff(*dune_result, previous_mu_batch_, previous_lambda_batch_,
                 config_.nrmp_max_num);
    diff = mu_diff * mu_diff + lambda_diff * lambda_diff;
  }

  previous_nominal_states_ = nrmp_result.state_trajectory;
  previous_nominal_controls_ = nrmp_result.control_trajectory;
  previous_mu_batch_ =
      dune_result == nullptr ? std::vector<DuneMatrix>() : dune_result->mu_batch;
  previous_lambda_batch_ = dune_result == nullptr ? std::vector<DuneMatrix>()
                                                  : dune_result->lambda_batch;
  return diff < config_.iter_threshold;
}

void PAN::resetIterationState() {
  previous_nominal_states_.resize(0, 0);
  previous_nominal_controls_.resize(0, 0);
  previous_mu_batch_.clear();
  previous_lambda_batch_.clear();
  has_previous_iteration_ = false;
}

void PAN::resetObstacleState() {
  previous_mu_batch_.clear();
  previous_lambda_batch_.clear();
}

}  // namespace neupan_uav
