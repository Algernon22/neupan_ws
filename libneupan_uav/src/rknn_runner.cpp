#include "neupan_uav/rknn_runner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifdef NEUPAN_UAV_WITH_RKNN
#include <rknn_api.h>
#endif

namespace neupan_uav {

namespace {

namespace fs = std::filesystem;

std::string readTextFile(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("failed to open RKNN metadata: " + path);
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

std::string vectorToString(const Eigen::Vector3d& value) {
  std::ostringstream stream;
  stream << "[" << value(0) << ", " << value(1) << ", " << value(2) << "]";
  return stream.str();
}

std::string metadataSuffix(const std::string& metadata_path) {
  return ", metadata_path=" + metadata_path;
}

#ifdef NEUPAN_UAV_WITH_RKNN

using Clock = std::chrono::steady_clock;

double elapsedSeconds(const Clock::time_point& start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

std::vector<unsigned char> readBinaryFile(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("failed to open RKNN model: " + path);
  }
  return std::vector<unsigned char>(std::istreambuf_iterator<char>(stream),
                                    std::istreambuf_iterator<char>());
}

#endif

std::size_t findJsonKey(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  const std::size_t key_pos = json.find(needle);
  if (key_pos == std::string::npos) {
    throw std::invalid_argument("RKNN metadata is missing key: " + key);
  }
  const std::size_t colon = json.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    throw std::invalid_argument("RKNN metadata key has no value: " + key);
  }
  return colon + 1;
}

std::size_t skipSpace(const std::string& text, std::size_t pos) {
  while (pos < text.size() &&
         std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
    ++pos;
  }
  return pos;
}

std::string parseJsonString(const std::string& json, const std::string& key) {
  std::size_t pos = skipSpace(json, findJsonKey(json, key));
  if (pos >= json.size() || json[pos] != '"') {
    throw std::invalid_argument("RKNN metadata key must be a string: " + key);
  }
  ++pos;
  std::string value;
  bool escaping = false;
  for (; pos < json.size(); ++pos) {
    const char ch = json[pos];
    if (escaping) {
      value.push_back(ch);
      escaping = false;
      continue;
    }
    if (ch == '\\') {
      escaping = true;
      continue;
    }
    if (ch == '"') {
      return value;
    }
    value.push_back(ch);
  }
  throw std::invalid_argument("unterminated string in RKNN metadata key: " + key);
}

std::string parseJsonToken(const std::string& json, const std::string& key) {
  std::size_t pos = skipSpace(json, findJsonKey(json, key));
  const std::size_t begin = pos;
  while (pos < json.size() && json[pos] != ',' && json[pos] != '}' &&
         json[pos] != ']' &&
         std::isspace(static_cast<unsigned char>(json[pos])) == 0) {
    ++pos;
  }
  if (begin == pos) {
    throw std::invalid_argument("RKNN metadata key has an empty value: " + key);
  }
  return json.substr(begin, pos - begin);
}

std::vector<double> parseNumberArray(const std::string& json,
                                     const std::string& key,
                                     std::size_t expected_size) {
  std::size_t pos = skipSpace(json, findJsonKey(json, key));
  if (pos >= json.size() || json[pos] != '[') {
    throw std::invalid_argument("RKNN metadata key must be an array: " + key);
  }
  ++pos;
  std::vector<double> values;
  while (pos < json.size()) {
    pos = skipSpace(json, pos);
    if (pos < json.size() && json[pos] == ']') {
      ++pos;
      break;
    }
    std::size_t consumed = 0;
    const double value = std::stod(json.substr(pos), &consumed);
    if (consumed == 0) {
      throw std::invalid_argument("failed to parse RKNN metadata array: " + key);
    }
    values.push_back(value);
    pos += consumed;
    pos = skipSpace(json, pos);
    if (pos < json.size() && json[pos] == ',') {
      ++pos;
      continue;
    }
    if (pos < json.size() && json[pos] == ']') {
      ++pos;
      break;
    }
    throw std::invalid_argument("malformed RKNN metadata array: " + key);
  }
  if (values.size() != expected_size) {
    throw std::invalid_argument("RKNN metadata key has unexpected array size: " +
                                key);
  }
  return values;
}

int parsePositiveInt(const std::string& json, const std::string& key) {
  const int value = std::stoi(parseJsonToken(json, key));
  if (value <= 0) {
    throw std::invalid_argument("RKNN metadata key must be positive: " + key);
  }
  return value;
}

bool parseBool(const std::string& json, const std::string& key) {
  const std::string token = parseJsonToken(json, key);
  if (token == "true" || token == "True" || token == "1") return true;
  if (token == "false" || token == "False" || token == "0") return false;
  throw std::invalid_argument("RKNN metadata key must be boolean: " + key);
}

std::array<int, 3> parseShape3(const std::string& json,
                               const std::string& key) {
  const std::vector<double> raw = parseNumberArray(json, key, 3);
  std::array<int, 3> shape{{0, 0, 0}};
  for (std::size_t i = 0; i < 3; ++i) {
    const double value = raw[i];
    const int parsed = static_cast<int>(value);
    if (value != static_cast<double>(parsed) || parsed <= 0) {
      throw std::invalid_argument(
          "RKNN metadata shape values must be positive integers: " + key);
    }
    shape[i] = parsed;
  }
  return shape;
}

Eigen::Vector3d parsePositiveVector3(const std::string& json,
                                     const std::string& key) {
  const std::vector<double> raw = parseNumberArray(json, key, 3);
  Eigen::Vector3d value;
  for (std::size_t i = 0; i < 3; ++i) {
    if (!std::isfinite(raw[i]) || raw[i] <= 0.0) {
      throw std::invalid_argument(
          "RKNN metadata vector values must be positive and finite: " + key);
    }
    value(static_cast<Eigen::Index>(i)) = raw[i];
  }
  return value;
}

std::string expandUser(const std::string& path) {
  if (path.empty() || path[0] != '~') return path;
  const char* home = std::getenv("HOME");
  if (home == nullptr || std::strlen(home) == 0) return path;
  if (path.size() == 1) return std::string(home);
  if (path[1] == '/') return std::string(home) + path.substr(1);
  return path;
}

std::string resolveRelatedPath(const std::string& value,
                               const std::string& metadata_path) {
  const fs::path raw(expandUser(value));
  if (raw.is_absolute()) return fs::weakly_canonical(raw).string();

  std::vector<fs::path> candidates;
  candidates.emplace_back(fs::path(metadata_path).parent_path());
  candidates.emplace_back(fs::current_path());
  if (const char* env_root = std::getenv("NEUPAN_ROOT")) {
    if (std::strlen(env_root) > 0) candidates.emplace_back(env_root);
  }

  for (const fs::path& base : candidates) {
    const fs::path candidate = base / raw;
    if (fs::exists(candidate)) {
      return fs::weakly_canonical(candidate).string();
    }
  }
  return fs::absolute(candidates.front() / raw).lexically_normal().string();
}

bool allFiniteFloat(const RknnFloatMatrix& matrix) {
  for (Eigen::Index i = 0; i < matrix.size(); ++i) {
    if (!std::isfinite(matrix.data()[i])) return false;
  }
  return true;
}

void validatePointFlow(const RknnMetadata& metadata,
                       const std::vector<RknnFloatMatrix>& point_flow) {
  const int expected_steps = metadata.receding + 1;
  if (static_cast<int>(point_flow.size()) != expected_steps) {
    throw std::invalid_argument(
        "point_flow must contain receding+1 matrices");
  }

  int num_points = -1;
  for (const RknnFloatMatrix& step_points : point_flow) {
    if (step_points.rows() != 3) {
      throw std::invalid_argument("point_flow matrices must have shape 3xN");
    }
    if (num_points < 0) {
      num_points = static_cast<int>(step_points.cols());
    } else if (step_points.cols() != num_points) {
      throw std::invalid_argument(
          "point_flow matrices must have a consistent point count");
    }
    if (!allFiniteFloat(step_points)) {
      throw std::invalid_argument("point_flow values must be finite");
    }
  }
  if (num_points < 0) {
    throw std::invalid_argument("point_flow must not be empty");
  }
  if (num_points > metadata.dune_max_num) {
    throw std::invalid_argument(
        "point_flow point count exceeds RKNN dune_max_num");
  }
}

std::vector<RknnFloatMatrix> castPointFlow(
    const std::vector<PointMatrix>& point_flow) {
  std::vector<RknnFloatMatrix> out;
  out.reserve(point_flow.size());
  for (const PointMatrix& step_points : point_flow) {
    out.push_back(step_points.cast<float>());
  }
  return out;
}

#ifdef NEUPAN_UAV_WITH_RKNN

std::string rknnError(const std::string& call, int ret) {
  return call + " failed: " + std::to_string(ret);
}

rknn_core_mask parseCoreMask(const std::string& name) {
  std::string normalized;
  normalized.reserve(name.size());
  for (char ch : name) {
    normalized.push_back(static_cast<char>(
        std::toupper(static_cast<unsigned char>(ch))));
  }
  if (normalized == "AUTO" || normalized == "NPU_CORE_AUTO" ||
      normalized == "RKNN_NPU_CORE_AUTO") {
    return RKNN_NPU_CORE_AUTO;
  }
  if (normalized == "CORE_0" || normalized == "0" ||
      normalized == "NPU_CORE_0" || normalized == "RKNN_NPU_CORE_0") {
    return RKNN_NPU_CORE_0;
  }
  if (normalized == "CORE_1" || normalized == "1" ||
      normalized == "NPU_CORE_1" || normalized == "RKNN_NPU_CORE_1") {
    return RKNN_NPU_CORE_1;
  }
  if (normalized == "CORE_2" || normalized == "2" ||
      normalized == "NPU_CORE_2" || normalized == "RKNN_NPU_CORE_2") {
    return RKNN_NPU_CORE_2;
  }
  if (normalized == "CORE_0_1" || normalized == "0_1" ||
      normalized == "NPU_CORE_0_1" || normalized == "RKNN_NPU_CORE_0_1") {
    return RKNN_NPU_CORE_0_1;
  }
  if (normalized == "CORE_0_1_2" || normalized == "0_1_2" ||
      normalized == "NPU_CORE_0_1_2" ||
      normalized == "RKNN_NPU_CORE_0_1_2") {
    return RKNN_NPU_CORE_0_1_2;
  }
  if (normalized == "ALL" || normalized == "NPU_CORE_ALL" ||
      normalized == "RKNN_NPU_CORE_ALL") {
    return RKNN_NPU_CORE_ALL;
  }
  throw std::invalid_argument("unsupported RKNN core mask: " + name);
}

#endif

}  // namespace

RknnMetadata RknnMetadata::load(const std::string& metadata_path) {
  const fs::path resolved_metadata =
      fs::weakly_canonical(fs::absolute(expandUser(metadata_path)));
  const std::string json = readTextFile(resolved_metadata.string());

  RknnMetadata metadata;
  metadata.metadata_path = resolved_metadata.string();
  metadata.receding = parsePositiveInt(json, "receding");
  metadata.dune_max_num = parsePositiveInt(json, "dune_max_num");
  metadata.max_points = parsePositiveInt(json, "max_points");
  metadata.output_dim = parsePositiveInt(json, "output_dim");
  metadata.input_shape = parseShape3(json, "input_shape");
  metadata.output_shape = parseShape3(json, "output_shape");
  metadata.half_extent = parsePositiveVector3(json, "half_extent");
  metadata.scene_scale = parsePositiveVector3(json, "scene_scale");
  metadata.clearance_scale = parsePositiveVector3(json, "clearance_scale");
  metadata.do_quantization = parseBool(json, "do_quantization");

  const int expected_max_points =
      (metadata.receding + 1) * metadata.dune_max_num;
  if (metadata.max_points != expected_max_points) {
    throw std::invalid_argument(
        "RKNN metadata max_points does not match (receding+1)*dune_max_num");
  }
  const std::array<int, 3> expected_input{
      {1, expected_max_points, 3}};
  const std::array<int, 3> expected_output{
      {1, expected_max_points, metadata.output_dim}};
  if (metadata.input_shape != expected_input) {
    throw std::invalid_argument("RKNN metadata input_shape is inconsistent");
  }
  if (metadata.output_shape != expected_output) {
    throw std::invalid_argument("RKNN metadata output_shape is inconsistent");
  }
  if (metadata.do_quantization) {
    throw std::invalid_argument(
        "quantized RKNN models are not allowed in this FP runtime");
  }

  metadata.rknn_path =
      resolveRelatedPath(parseJsonString(json, "rknn"), metadata.metadata_path);
  if (!fs::is_regular_file(metadata.rknn_path)) {
    throw std::runtime_error("RKNN model file not found: " +
                             metadata.rknn_path);
  }
  return metadata;
}

void RknnMetadata::validateRuntime(const RknnRuntimeContract& expected) const {
  if (receding != expected.receding) {
    throw std::invalid_argument(
        "RKNN receding mismatch: model=" + std::to_string(receding) +
        ", runtime=" + std::to_string(expected.receding) +
        metadataSuffix(metadata_path));
  }
  if (dune_max_num != expected.dune_max_num) {
    throw std::invalid_argument(
        "RKNN dune_max_num mismatch: model=" + std::to_string(dune_max_num) +
        ", runtime=" + std::to_string(expected.dune_max_num) +
        metadataSuffix(metadata_path));
  }
  if (output_dim != expected.output_dim) {
    throw std::invalid_argument(
        "RKNN output_dim mismatch: model=" + std::to_string(output_dim) +
        ", runtime=" + std::to_string(expected.output_dim) +
        metadataSuffix(metadata_path));
  }
  if (!half_extent.isApprox(expected.body_half_extent, 1.0e-5)) {
    throw std::invalid_argument(
        "RKNN half_extent mismatch: model=" + vectorToString(half_extent) +
        ", runtime=" + vectorToString(expected.body_half_extent) +
        metadataSuffix(metadata_path));
  }
}

PackedRknnInput packPointFlow(
    const RknnMetadata& metadata,
    const std::vector<RknnFloatMatrix>& point_flow) {
  validatePointFlow(metadata, point_flow);
  const int num_steps = metadata.receding + 1;
  const int num_points =
      point_flow.empty() ? 0 : static_cast<int>(point_flow.front().cols());

  PackedRknnInput packed;
  packed.num_steps = num_steps;
  packed.num_points = num_points;
  packed.values.assign(static_cast<std::size_t>(metadata.max_points * 3),
                       0.0F);
  for (int step = 0; step < num_steps; ++step) {
    const RknnFloatMatrix& points =
        point_flow[static_cast<std::size_t>(step)];
    for (int point = 0; point < num_points; ++point) {
      for (int dim = 0; dim < 3; ++dim) {
        const int packed_row = step * metadata.dune_max_num + point;
        packed.values[static_cast<std::size_t>(packed_row * 3 + dim)] =
            points(dim, point);
      }
    }
  }
  return packed;
}

RknnFloatMatrix unpackRawMu(const RknnMetadata& metadata,
                            const std::vector<float>& output_full,
                            int num_points) {
  if (num_points < 0 || num_points > metadata.dune_max_num) {
    throw std::invalid_argument("invalid RKNN raw output point count");
  }
  const std::size_t expected_size =
      static_cast<std::size_t>(metadata.max_points * metadata.output_dim);
  if (output_full.size() != expected_size) {
    throw std::invalid_argument("unexpected RKNN raw output size");
  }
  const int num_steps = metadata.receding + 1;
  RknnFloatMatrix raw_mu(num_steps * num_points, metadata.output_dim);
  for (int step = 0; step < num_steps; ++step) {
    for (int point = 0; point < num_points; ++point) {
      const int output_row = step * metadata.dune_max_num + point;
      const int raw_row = step * num_points + point;
      for (int dim = 0; dim < metadata.output_dim; ++dim) {
        raw_mu(raw_row, dim) =
            output_full[static_cast<std::size_t>(
                output_row * metadata.output_dim + dim)];
      }
    }
  }
  return raw_mu;
}

bool compiledWithRknnRuntime() {
#ifdef NEUPAN_UAV_WITH_RKNN
  return true;
#else
  return false;
#endif
}

bool rknnDeviceAvailable() {
  if (fs::exists("/dev/dri")) {
    for (const fs::directory_entry& entry : fs::directory_iterator("/dev/dri")) {
      if (entry.path().filename().string().rfind("renderD", 0) == 0) {
        return true;
      }
    }
  }
  if (fs::exists("/dev")) {
    for (const fs::directory_entry& entry : fs::directory_iterator("/dev")) {
      if (entry.path().filename().string().rfind("rknpu", 0) == 0) {
        return true;
      }
    }
  }
  return false;
}

RknnFloatMatrix RknnRunner::inferRawMu(
    const std::vector<PointMatrix>& point_flow) {
  return inferRawMu(castPointFlow(point_flow));
}

MockRknnRunner::MockRknnRunner(RknnMetadata metadata)
    : metadata_(std::move(metadata)) {
  output_full_.assign(
      static_cast<std::size_t>(metadata_.max_points * metadata_.output_dim),
      0.0F);
}

RknnFloatMatrix MockRknnRunner::inferRawMu(
    const std::vector<RknnFloatMatrix>& point_flow) {
  const PackedRknnInput packed = packPointFlow(metadata_, point_flow);
  last_packed_input_ = packed.values;
  if (packed.num_points == 0) {
    profile_.inference_sec = 0.0;
    return RknnFloatMatrix(0, metadata_.output_dim);
  }
  ++inference_count_;
  profile_.inference_sec = 0.0;
  return unpackRawMu(metadata_, output_full_, packed.num_points);
}

void MockRknnRunner::setOutputFull(std::vector<float> output_full) {
  const std::size_t expected_size =
      static_cast<std::size_t>(metadata_.max_points * metadata_.output_dim);
  if (output_full.size() != expected_size) {
    throw std::invalid_argument("mock RKNN output has unexpected size");
  }
  output_full_ = std::move(output_full);
}

#ifdef NEUPAN_UAV_WITH_RKNN

struct ObsPointNetRknnRunner::Impl {
  explicit Impl(const RknnRunnerConfig& config)
      : metadata(RknnMetadata::load(config.metadata_path)) {
    if (config.expected_runtime.has_value()) {
      metadata.validateRuntime(*config.expected_runtime);
    }
    model = readBinaryFile(metadata.rknn_path);
    if (config.require_device && !rknnDeviceAvailable()) {
      throw std::runtime_error(
          "RKNN NPU device was not found. Expected /dev/rknpu* or "
          "/dev/dri/renderD*.");
    }
    if (model.size() > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("RKNN model is too large for rknn_init");
    }

    const int init_ret =
        rknn_init(&context, model.data(), static_cast<uint32_t>(model.size()),
                  0, nullptr);
    if (init_ret != RKNN_SUCC) {
      context = 0;
      throw std::runtime_error(rknnError("rknn_init", init_ret));
    }

    const int mask_ret = rknn_set_core_mask(context, parseCoreMask(config.core_mask));
    if (mask_ret != RKNN_SUCC) {
      throw std::runtime_error(rknnError("rknn_set_core_mask", mask_ret));
    }

    rknn_sdk_version version;
    std::memset(&version, 0, sizeof(version));
    const int version_ret =
        rknn_query(context, RKNN_QUERY_SDK_VERSION, &version, sizeof(version));
    if (version_ret == RKNN_SUCC) {
      profile.api_version = version.api_version;
      profile.driver_version = version.drv_version;
    }

    rknn_input_output_num io_num;
    std::memset(&io_num, 0, sizeof(io_num));
    const int io_ret =
        rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (io_ret != RKNN_SUCC) {
      throw std::runtime_error(rknnError("rknn_query(IN_OUT_NUM)", io_ret));
    }
    if (io_num.n_input != 1 || io_num.n_output != 1) {
      throw std::runtime_error(
          "RKNN ObsPointNet model must have exactly one input and one output");
    }
  }

  ~Impl() {
    if (context != 0) {
      (void)rknn_destroy(context);
      context = 0;
    }
  }

  RknnMetadata metadata;
  RknnProfile profile;
  std::vector<unsigned char> model;
  rknn_context context = 0;
};

ObsPointNetRknnRunner::ObsPointNetRknnRunner(
    const RknnRunnerConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

ObsPointNetRknnRunner::~ObsPointNetRknnRunner() = default;

const RknnMetadata& ObsPointNetRknnRunner::metadata() const {
  return impl_->metadata;
}

const RknnProfile& ObsPointNetRknnRunner::profile() const {
  return impl_->profile;
}

RknnFloatMatrix ObsPointNetRknnRunner::inferRawMu(
    const std::vector<RknnFloatMatrix>& point_flow) {
  const PackedRknnInput packed = packPointFlow(impl_->metadata, point_flow);
  if (packed.num_points == 0) {
    impl_->profile.inference_sec = 0.0;
    return RknnFloatMatrix(0, impl_->metadata.output_dim);
  }

  rknn_input input;
  std::memset(&input, 0, sizeof(input));
  input.index = 0;
  input.buf = const_cast<float*>(packed.values.data());
  input.size = static_cast<uint32_t>(packed.values.size() * sizeof(float));
  input.pass_through = 0;
  input.type = RKNN_TENSOR_FLOAT32;
  input.fmt = RKNN_TENSOR_NHWC;

  const int input_ret = rknn_inputs_set(impl_->context, 1, &input);
  if (input_ret != RKNN_SUCC) {
    throw std::runtime_error(rknnError("rknn_inputs_set", input_ret));
  }

  rknn_output output;
  std::memset(&output, 0, sizeof(output));
  output.index = 0;
  output.want_float = 1;
  output.is_prealloc = 0;

  const auto start = Clock::now();
  const int run_ret = rknn_run(impl_->context, nullptr);
  if (run_ret != RKNN_SUCC) {
    throw std::runtime_error(rknnError("rknn_run", run_ret));
  }
  const int output_ret = rknn_outputs_get(impl_->context, 1, &output, nullptr);
  impl_->profile.inference_sec = elapsedSeconds(start);
  if (output_ret != RKNN_SUCC) {
    throw std::runtime_error(rknnError("rknn_outputs_get", output_ret));
  }

  const std::size_t expected_size =
      static_cast<std::size_t>(impl_->metadata.max_points *
                               impl_->metadata.output_dim);
  const std::size_t actual_size = output.size / sizeof(float);
  if (actual_size != expected_size) {
    (void)rknn_outputs_release(impl_->context, 1, &output);
    throw std::runtime_error("unexpected RKNN output size");
  }

  const auto* output_data = static_cast<const float*>(output.buf);
  std::vector<float> output_full(output_data, output_data + expected_size);
  const int release_ret = rknn_outputs_release(impl_->context, 1, &output);
  if (release_ret != RKNN_SUCC) {
    throw std::runtime_error(rknnError("rknn_outputs_release", release_ret));
  }

  return unpackRawMu(impl_->metadata, output_full, packed.num_points);
}

#else

struct ObsPointNetRknnRunner::Impl {};

ObsPointNetRknnRunner::ObsPointNetRknnRunner(const RknnRunnerConfig&) {
  throw std::runtime_error(
      "libneupan_uav was built without RKNN runtime support");
}

ObsPointNetRknnRunner::~ObsPointNetRknnRunner() = default;

const RknnMetadata& ObsPointNetRknnRunner::metadata() const {
  throw std::runtime_error(
      "libneupan_uav was built without RKNN runtime support");
}

const RknnProfile& ObsPointNetRknnRunner::profile() const {
  throw std::runtime_error(
      "libneupan_uav was built without RKNN runtime support");
}

RknnFloatMatrix ObsPointNetRknnRunner::inferRawMu(
    const std::vector<RknnFloatMatrix>&) {
  throw std::runtime_error(
      "libneupan_uav was built without RKNN runtime support");
}

#endif

}  // namespace neupan_uav
