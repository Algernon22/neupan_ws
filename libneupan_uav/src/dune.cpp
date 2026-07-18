#include "neupan_uav/dune.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace neupan_uav {

namespace {

using Clock = std::chrono::steady_clock;

double elapsedSeconds(const Clock::time_point& start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

bool allFinite(const Eigen::MatrixXd& matrix) {
  for (Eigen::Index i = 0; i < matrix.size(); ++i) {
    if (!std::isfinite(matrix.data()[i])) return false;
  }
  return true;
}

bool allFiniteFloat(const DuneMatrix& matrix) {
  for (Eigen::Index i = 0; i < matrix.size(); ++i) {
    if (!std::isfinite(matrix.data()[i])) return false;
  }
  return true;
}

bool allFiniteFloatVector(const Eigen::VectorXf& vector) {
  for (Eigen::Index i = 0; i < vector.size(); ++i) {
    if (!std::isfinite(vector(i))) return false;
  }
  return true;
}

double asRatio(double value, const char* name) {
  if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and within [0, 1]");
  }
  return value;
}

std::vector<int> quotaCounts(int total, const std::vector<double>& ratios) {
  total = std::max(0, total);
  std::vector<int> counts(ratios.size(), 0);
  if (total == 0 || ratios.empty()) return counts;

  std::vector<double> raw(ratios.size(), 0.0);
  double raw_sum = 0.0;
  int count_sum = 0;
  for (std::size_t i = 0; i < ratios.size(); ++i) {
    raw[i] = std::clamp(ratios[i], 0.0, 1.0) * static_cast<double>(total);
    raw_sum += raw[i];
    counts[i] = static_cast<int>(std::floor(raw[i]));
    count_sum += counts[i];
  }

  const int target =
      std::min(total, static_cast<int>(std::floor(raw_sum + 1.0e-9)));
  int remaining = target - count_sum;
  std::vector<int> order(ratios.size(), 0);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    const double ra = raw[static_cast<std::size_t>(a)] -
                      counts[static_cast<std::size_t>(a)];
    const double rb = raw[static_cast<std::size_t>(b)] -
                      counts[static_cast<std::size_t>(b)];
    if (ra == rb) return a < b;
    return ra > rb;
  });
  for (int i = 0; i < remaining; ++i) {
    counts[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])] += 1;
  }
  return counts;
}

std::vector<int> stableArgsort(const std::vector<float>& values) {
  std::vector<int> order(values.size(), 0);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return values[static_cast<std::size_t>(a)] <
           values[static_cast<std::size_t>(b)];
  });
  return order;
}

int appendRanked(std::vector<int>& selected, std::vector<uint8_t>& selected_mask,
                 const std::vector<int>& candidates, int target_max,
                 int limit = -1,
                 const std::vector<uint8_t>* candidate_mask = nullptr) {
  if (target_max <= 0 || static_cast<int>(selected.size()) >= target_max) {
    return static_cast<int>(selected.size());
  }

  int room = target_max - static_cast<int>(selected.size());
  if (limit >= 0) room = std::min(room, std::max(0, limit));
  if (room <= 0) return static_cast<int>(selected.size());

  for (int idx : candidates) {
    if (idx < 0 || idx >= static_cast<int>(selected_mask.size()) ||
        selected_mask[static_cast<std::size_t>(idx)] != 0) {
      continue;
    }
    if (candidate_mask != nullptr &&
        (*candidate_mask)[static_cast<std::size_t>(idx)] == 0) {
      continue;
    }
    selected_mask[static_cast<std::size_t>(idx)] = 1;
    selected.push_back(idx);
    --room;
    if (room <= 0 || static_cast<int>(selected.size()) >= target_max) break;
  }
  return static_cast<int>(selected.size());
}

DunePostprocessorConfig normalizeConfig(DunePostprocessorConfig config) {
  if (config.receding < 0) {
    throw std::invalid_argument("DunePostprocessorConfig::receding must be non-negative");
  }
  if (config.point_dim != 3) {
    throw std::invalid_argument(
        "DunePostprocessorConfig::point_dim must be 3 for UAV DUNE");
  }
  if (config.edge_dim <= 0) {
    throw std::invalid_argument("DunePostprocessorConfig::edge_dim must be positive");
  }
  if (config.select_num == 0) {
    config.select_num = config.dune_max_num;
  }
  if (config.dune_max_num > 0) {
    config.select_num = std::min(config.select_num, config.dune_max_num);
  }
  if (config.select_num == 0) {
    throw std::invalid_argument("DunePostprocessorConfig::select_num must be positive");
  }

  config.select_nearest_ratio =
      asRatio(config.select_nearest_ratio,
              "DunePostprocessorConfig::select_nearest_ratio");
  config.select_temporal_ratio =
      asRatio(config.select_temporal_ratio,
              "DunePostprocessorConfig::select_temporal_ratio");
  config.select_diversity_ratio =
      asRatio(config.select_diversity_ratio,
              "DunePostprocessorConfig::select_diversity_ratio");
  const double ratio_sum = config.select_nearest_ratio +
                           config.select_temporal_ratio +
                           config.select_diversity_ratio;
  if (ratio_sum > 1.0 + 1.0e-6) {
    throw std::invalid_argument(
        "DunePostprocessorConfig selection ratios must sum to <= 1");
  }

  if (config.G.rows() != config.edge_dim ||
      config.G.cols() != config.point_dim) {
    throw std::invalid_argument(
        "DunePostprocessorConfig::G must have shape edge_dim x point_dim");
  }
  if (config.h.size() != config.edge_dim) {
    throw std::invalid_argument(
        "DunePostprocessorConfig::h must have edge_dim elements");
  }
  if (!allFiniteFloat(config.G) || !allFiniteFloatVector(config.h)) {
    throw std::invalid_argument("DunePostprocessorConfig::G and h must be finite");
  }
  return config;
}

void validatePointBatch(const std::vector<PointMatrix>& batch,
                        int expected_steps, int point_count,
                        const char* name) {
  if (static_cast<int>(batch.size()) != expected_steps) {
    throw std::invalid_argument(std::string(name) +
                                " must contain receding+1 steps");
  }
  for (const PointMatrix& points : batch) {
    if (points.rows() != 3 || points.cols() != point_count) {
      throw std::invalid_argument(std::string(name) +
                                  " must contain 3xN point matrices");
    }
    if (!allFinite(points)) {
      throw std::invalid_argument(std::string(name) + " must be finite");
    }
  }
}

void validateRotations(const std::vector<Eigen::Matrix3d>& rotations,
                       int expected_steps) {
  if (static_cast<int>(rotations.size()) != expected_steps) {
    throw std::invalid_argument("rotations must contain receding+1 matrices");
  }
  for (const Eigen::Matrix3d& rotation : rotations) {
    if (!allFinite(rotation)) {
      throw std::invalid_argument("rotations must be finite");
    }
  }
}

std::vector<std::vector<int>> quotaSelectIndices(
    const std::vector<std::vector<float>>& margin_batch,
    const std::vector<unsigned char>* selection_tags, int select_num,
    double nearest_ratio, double temporal_ratio, double diversity_ratio,
    DuneSelectionProfile* profile) {
  const int num_steps = static_cast<int>(margin_batch.size());
  const int num_points =
      num_steps == 0 ? 0 : static_cast<int>(margin_batch.front().size());
  const int k = std::min(num_points, select_num);
  std::vector<std::vector<int>> selected_by_step(
      static_cast<std::size_t>(num_steps));
  if (k <= 0) return selected_by_step;

  const std::vector<int> quotas =
      quotaCounts(k, {nearest_ratio, temporal_ratio, diversity_ratio});
  const int nearest_quota = quotas.size() > 0 ? quotas[0] : 0;
  const int temporal_quota = quotas.size() > 1 ? quotas[1] : 0;
  const int diversity_quota = quotas.size() > 2 ? quotas[2] : 0;

  bool temporal_active = false;
  bool diversity_active = false;
  if (selection_tags != nullptr) {
    for (unsigned char tag : *selection_tags) {
      temporal_active = temporal_active || hasTemporalTag(tag);
      diversity_active = diversity_active || hasDiversityTag(tag);
    }
  }
  temporal_active = temporal_active && temporal_quota > 0;
  diversity_active = diversity_active && diversity_quota > 0;

  double nearest_total = 0.0;
  double temporal_total = 0.0;
  double diversity_total = 0.0;

  for (int step = 0; step < num_steps; ++step) {
    const std::vector<int> ordered =
        stableArgsort(margin_batch[static_cast<std::size_t>(step)]);

    if (!temporal_active && !diversity_active) {
      std::vector<int> selected(ordered.begin(), ordered.begin() + k);
      selected_by_step[static_cast<std::size_t>(step)] = std::move(selected);
      nearest_total += std::min(nearest_quota, k);
      if (selection_tags != nullptr) {
        for (int idx : selected_by_step[static_cast<std::size_t>(step)]) {
          const unsigned char tag = (*selection_tags)[static_cast<std::size_t>(idx)];
          temporal_total += hasTemporalTag(tag) ? 1.0 : 0.0;
          diversity_total += hasDiversityTag(tag) ? 1.0 : 0.0;
        }
      }
      continue;
    }

    std::vector<uint8_t> temporal_mask(static_cast<std::size_t>(num_points), 0);
    std::vector<uint8_t> diversity_mask(static_cast<std::size_t>(num_points), 0);
    if (selection_tags != nullptr) {
      for (int idx = 0; idx < num_points; ++idx) {
        const unsigned char tag =
            (*selection_tags)[static_cast<std::size_t>(idx)];
        temporal_mask[static_cast<std::size_t>(idx)] =
            hasTemporalTag(tag) ? 1 : 0;
        diversity_mask[static_cast<std::size_t>(idx)] =
            hasDiversityTag(tag) ? 1 : 0;
      }
    }

    std::vector<int> selected;
    selected.reserve(static_cast<std::size_t>(k));
    std::vector<uint8_t> selected_mask(static_cast<std::size_t>(num_points), 0);
    appendRanked(selected, selected_mask, ordered, k, nearest_quota);
    nearest_total += selected.size();

    if (temporal_active) {
      int already_temporal = 0;
      for (int idx : selected) {
        already_temporal +=
            temporal_mask[static_cast<std::size_t>(idx)] != 0 ? 1 : 0;
      }
      appendRanked(selected, selected_mask, ordered, k,
                   std::max(0, temporal_quota - already_temporal),
                   &temporal_mask);
    }

    if (diversity_active) {
      int already_diverse = 0;
      for (int idx : selected) {
        already_diverse +=
            diversity_mask[static_cast<std::size_t>(idx)] != 0 ? 1 : 0;
      }
      appendRanked(selected, selected_mask, ordered, k,
                   std::max(0, diversity_quota - already_diverse),
                   &diversity_mask);
    }

    appendRanked(selected, selected_mask, ordered, k);

    for (int idx : selected) {
      temporal_total += temporal_mask[static_cast<std::size_t>(idx)] != 0 ? 1.0 : 0.0;
      diversity_total += diversity_mask[static_cast<std::size_t>(idx)] != 0 ? 1.0 : 0.0;
    }
    selected_by_step[static_cast<std::size_t>(step)] = std::move(selected);
  }

  if (profile != nullptr && num_steps > 0) {
    profile->nearest_selected_per_step =
        nearest_total / static_cast<double>(num_steps);
    profile->temporal_selected_per_step =
        temporal_total / static_cast<double>(num_steps);
    profile->diversity_selected_per_step =
        diversity_total / static_cast<double>(num_steps);
  }
  return selected_by_step;
}

}  // namespace

DunePostprocessor::DunePostprocessor(DunePostprocessorConfig config)
    : config_(normalizeConfig(std::move(config))), configured_(true) {}

DuneResult DunePostprocessor::process(
    const DuneMatrix& raw_mu, const std::vector<PointMatrix>& point_flow,
    const std::vector<Eigen::Matrix3d>& rotations,
    const std::vector<PointMatrix>& obstacle_points_by_step,
    const std::vector<unsigned char>* selection_tags,
    double inference_sec) const {
  if (!configured_) {
    throw std::logic_error(
        "DunePostprocessor must be constructed with DunePostprocessorConfig "
        "for raw-mu postprocessing");
  }
  if (!std::isfinite(inference_sec) || inference_sec < 0.0) {
    throw std::invalid_argument("DUNE inference time must be finite and non-negative");
  }

  const auto select_start = Clock::now();
  const int num_steps = config_.receding + 1;
  const int point_count =
      point_flow.empty() ? 0 : static_cast<int>(point_flow.front().cols());
  if (config_.dune_max_num > 0 &&
      point_count > static_cast<int>(config_.dune_max_num)) {
    throw std::invalid_argument("DUNE point count exceeds dune_max_num");
  }
  if (selection_tags != nullptr &&
      static_cast<int>(selection_tags->size()) != point_count) {
    throw std::invalid_argument("DUNE selection tags must have one entry per point");
  }
  validatePointBatch(point_flow, num_steps, point_count, "point_flow");
  validateRotations(rotations, num_steps);
  validatePointBatch(obstacle_points_by_step, num_steps, point_count,
                     "obstacle_points_by_step");

  DuneResult result;
  result.profile.dune_inference_sec = inference_sec;

  if (point_count == 0) {
    if (raw_mu.rows() != 0 || raw_mu.cols() != config_.edge_dim) {
      throw std::invalid_argument(
          "raw_mu must have shape ((receding+1)*N) x edge_dim");
    }
    result.min_margin = std::numeric_limits<double>::infinity();
    result.profile.dune_select_sec = elapsedSeconds(select_start);
    result.profile.dune_selected_count = 0;
    return result;
  }

  if (raw_mu.rows() != num_steps * point_count ||
      raw_mu.cols() != config_.edge_dim) {
    throw std::invalid_argument(
        "raw_mu must have shape ((receding+1)*N) x edge_dim");
  }
  if (!allFiniteFloat(raw_mu)) {
    throw std::invalid_argument("raw_mu must be finite");
  }

  DuneMatrix total_mu = raw_mu.cwiseMax(0.0F);
  for (Eigen::Index row = 0; row < total_mu.rows(); ++row) {
    const Eigen::RowVectorXf dual_vector =
        total_mu.row(row) * config_.G;
    const float scale = std::max(dual_vector.norm(), 1.0F);
    total_mu.row(row) /= scale;
  }

  std::vector<DuneMatrix> mu_batch;
  std::vector<DuneMatrix> lambda_batch;
  std::vector<std::vector<float>> margin_batch;
  mu_batch.reserve(static_cast<std::size_t>(num_steps));
  lambda_batch.reserve(static_cast<std::size_t>(num_steps));
  margin_batch.reserve(static_cast<std::size_t>(num_steps));

  for (int step = 0; step < num_steps; ++step) {
    DuneMatrix mu(config_.edge_dim, point_count);
    for (int point = 0; point < point_count; ++point) {
      const int raw_row = step * point_count + point;
      mu.col(point) = total_mu.row(raw_row).transpose();
    }
    mu_batch.push_back(mu);

    const DuneMatrix rotation =
        rotations[static_cast<std::size_t>(step)].cast<float>();
    lambda_batch.push_back(-(rotation * config_.G.transpose() * mu));

    std::vector<float> margin(static_cast<std::size_t>(point_count), 0.0F);
    const DuneMatrix flow =
        point_flow[static_cast<std::size_t>(step)].cast<float>();
    for (int point = 0; point < point_count; ++point) {
      const Eigen::VectorXf temp =
          config_.G * flow.col(point) - config_.h;
      margin[static_cast<std::size_t>(point)] =
          mu.col(point).dot(temp);
    }
    margin_batch.push_back(std::move(margin));
  }

  result.min_margin =
      margin_batch.empty()
          ? std::numeric_limits<double>::infinity()
          : static_cast<double>(
                *std::min_element(margin_batch.front().begin(),
                                  margin_batch.front().end()));

  DuneSelectionProfile selection_profile;
  const int select_num = std::min(
      point_count, static_cast<int>(config_.select_num));
  const std::vector<std::vector<int>> selected_by_step = quotaSelectIndices(
      margin_batch, selection_tags, select_num, config_.select_nearest_ratio,
      config_.select_temporal_ratio, config_.select_diversity_ratio,
      &selection_profile);

  result.selection_profile = selection_profile;
  result.selected_count = static_cast<std::size_t>(select_num);
  result.profile.dune_selected_count = result.selected_count;
  result.profile.nearest_selected =
      static_cast<std::size_t>(std::llround(selection_profile.nearest_selected_per_step));
  result.profile.continuity_selected =
      static_cast<std::size_t>(std::llround(selection_profile.temporal_selected_per_step));
  result.profile.diversity_selected =
      static_cast<std::size_t>(std::llround(selection_profile.diversity_selected_per_step));

  result.mu_batch.reserve(static_cast<std::size_t>(num_steps));
  result.lambda_batch.reserve(static_cast<std::size_t>(num_steps));
  result.point_batch.reserve(static_cast<std::size_t>(num_steps));

  for (int step = 0; step < num_steps; ++step) {
    const std::vector<int>& selected =
        selected_by_step[static_cast<std::size_t>(step)];
    const int k = static_cast<int>(selected.size());
    DuneMatrix mu_selected(config_.edge_dim, k);
    DuneMatrix lambda_selected(config_.point_dim, k);
    DuneMatrix point_selected(config_.point_dim, k);
    for (int out_col = 0; out_col < k; ++out_col) {
      const int src_col = selected[static_cast<std::size_t>(out_col)];
      mu_selected.col(out_col) =
          mu_batch[static_cast<std::size_t>(step)].col(src_col);
      lambda_selected.col(out_col) =
          lambda_batch[static_cast<std::size_t>(step)].col(src_col);
      point_selected.col(out_col) =
          obstacle_points_by_step[static_cast<std::size_t>(step)]
              .col(src_col)
              .cast<float>();
    }
    result.mu_batch.push_back(std::move(mu_selected));
    result.lambda_batch.push_back(std::move(lambda_selected));
    result.point_batch.push_back(std::move(point_selected));
  }

  result.selected_points.resize(3, result.selected_count);
  if (result.selected_count > 0) {
    result.selected_points =
        result.point_batch.front().template cast<double>();
  }
  result.profile.dune_select_sec = elapsedSeconds(select_start);
  return result;
}

}  // namespace neupan_uav
