#include "neupan_uav/planner.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string fixture_path;
  std::string output_path;
};

void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " --fixture PATH --output PATH\n";
}

std::string requireValue(int argc, char** argv, int& index,
                         const std::string& option) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(option + " requires a value");
  }
  ++index;
  return argv[index];
}

Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--fixture") {
      args.fixture_path = requireValue(argc, argv, i, option);
    } else if (option == "--output") {
      args.output_path = requireValue(argc, argv, i, option);
    } else if (option == "--help" || option == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + option);
    }
  }
  if (args.fixture_path.empty() || args.output_path.empty()) {
    throw std::invalid_argument("--fixture and --output are required");
  }
  return args;
}

std::string readTextFile(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("failed to open: " + path);
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

class JsonReader {
 public:
  explicit JsonReader(std::string text) : text_(std::move(text)) {}

  void expect(char expected) {
    skipSpace();
    if (pos_ >= text_.size() || text_[pos_] != expected) {
      throw std::runtime_error(std::string("expected '") + expected + "'");
    }
    ++pos_;
  }

  bool consume(char value) {
    skipSpace();
    if (pos_ < text_.size() && text_[pos_] == value) {
      ++pos_;
      return true;
    }
    return false;
  }

  std::string string() {
    skipSpace();
    if (pos_ >= text_.size() || text_[pos_] != '"') {
      throw std::runtime_error("expected JSON string");
    }
    ++pos_;
    std::string out;
    while (pos_ < text_.size()) {
      const char ch = text_[pos_++];
      if (ch == '"') return out;
      if (ch == '\\') {
        if (pos_ >= text_.size()) throw std::runtime_error("bad JSON escape");
        const char esc = text_[pos_++];
        switch (esc) {
          case '"':
          case '\\':
          case '/':
            out.push_back(esc);
            break;
          case 'b':
            out.push_back('\b');
            break;
          case 'f':
            out.push_back('\f');
            break;
          case 'n':
            out.push_back('\n');
            break;
          case 'r':
            out.push_back('\r');
            break;
          case 't':
            out.push_back('\t');
            break;
          default:
            throw std::runtime_error("unsupported JSON escape");
        }
      } else {
        out.push_back(ch);
      }
    }
    throw std::runtime_error("unterminated JSON string");
  }

  double number() {
    skipSpace();
    const char* start = text_.c_str() + pos_;
    char* end = nullptr;
    const double value = std::strtod(start, &end);
    if (end == start) throw std::runtime_error("expected JSON number");
    pos_ = static_cast<std::size_t>(end - text_.c_str());
    return value;
  }

  bool boolean() {
    skipSpace();
    if (text_.compare(pos_, 4, "true") == 0) {
      pos_ += 4;
      return true;
    }
    if (text_.compare(pos_, 5, "false") == 0) {
      pos_ += 5;
      return false;
    }
    throw std::runtime_error("expected JSON boolean");
  }

  void skipValue() {
    skipSpace();
    if (pos_ >= text_.size()) throw std::runtime_error("unexpected EOF");
    const char ch = text_[pos_];
    if (ch == '{') {
      expect('{');
      if (consume('}')) return;
      do {
        (void)string();
        expect(':');
        skipValue();
      } while (consume(','));
      expect('}');
    } else if (ch == '[') {
      expect('[');
      if (consume(']')) return;
      do {
        skipValue();
      } while (consume(','));
      expect(']');
    } else if (ch == '"') {
      (void)string();
    } else if (ch == 't' || ch == 'f') {
      (void)boolean();
    } else if (ch == 'n') {
      if (text_.compare(pos_, 4, "null") != 0) {
        throw std::runtime_error("expected null");
      }
      pos_ += 4;
    } else {
      (void)number();
    }
  }

 private:
  void skipSpace() {
    while (pos_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
  }

  std::string text_;
  std::size_t pos_ = 0;
};

struct ReplayPlannerInputs {
  neupan_uav::PlannerOptions options;
  neupan_uav::UavDynamicsConfig dynamics;
  neupan_uav::DuneOptions dune_options;
  double state_weight_gain = 1.0;
  int dune_max_num = 0;

  ReplayPlannerInputs() {
    options.grid.horizon_steps = 4;
    options.grid.dt = 0.1;
    options.ref_speed = 1.0;
    options.collision_threshold = 0.05;
    options.arrive_threshold = 0.15;
    options.robot.body_half_extent = Eigen::Vector3d(0.23, 0.23, 0.06);
    options.preselect.enabled = true;
    options.preselect.max_points = 0;
    options.preselect.per_step = 0;
    options.preselect.nearest_ratio = 1.0;
    options.preselect.temporal_ratio = 0.0;
    options.preselect.diversity_ratio = 0.0;
    options.preselect.corridor_margin = Eigen::Vector3d::Constant(10.0);
    options.preselect.exact_margin = Eigen::Vector3d::Zero();
    options.pan.iter_num = 1;
    options.pan.trajectory_threshold = 1.0e-9;
    options.pan.dune_threshold = 1.0e-9;
    dynamics.yaw_rate_time_constant = 0.25;
  }
};

struct FrameFixture {
  Eigen::VectorXd state;
  neupan_uav::PointMatrix obstacle_points = neupan_uav::emptyPointMatrix();
};

struct Fixture {
  ReplayPlannerInputs planner;
  std::vector<FrameFixture> frames;
};

neupan_uav::Control readControlArray(JsonReader& reader) {
  neupan_uav::Control out;
  reader.expect('[');
  for (int i = 0; i < 4; ++i) {
    if (i > 0) reader.expect(',');
    out(i) = reader.number();
  }
  reader.expect(']');
  return out;
}

Eigen::Vector3d readVector3(JsonReader& reader) {
  Eigen::Vector3d out;
  reader.expect('[');
  for (int i = 0; i < 3; ++i) {
    if (i > 0) reader.expect(',');
    out(i) = reader.number();
  }
  reader.expect(']');
  return out;
}

Eigen::Vector4d readVector4(JsonReader& reader) {
  Eigen::Vector4d out;
  reader.expect('[');
  for (int i = 0; i < 4; ++i) {
    if (i > 0) reader.expect(',');
    out(i) = reader.number();
  }
  reader.expect(']');
  return out;
}

Eigen::VectorXd readVector(JsonReader& reader) {
  std::vector<double> values;
  reader.expect('[');
  if (!reader.consume(']')) {
    do {
      values.push_back(reader.number());
    } while (reader.consume(','));
    reader.expect(']');
  }
  Eigen::VectorXd out(values.size());
  for (Eigen::Index i = 0; i < out.size(); ++i) {
    out(i) = values[static_cast<std::size_t>(i)];
  }
  return out;
}

neupan_uav::PointMatrix readPointRows(JsonReader& reader) {
  std::vector<Eigen::Vector3d> rows;
  reader.expect('[');
  if (!reader.consume(']')) {
    do {
      rows.push_back(readVector3(reader));
    } while (reader.consume(','));
    reader.expect(']');
  }
  neupan_uav::PointMatrix points(3, static_cast<Eigen::Index>(rows.size()));
  for (Eigen::Index col = 0; col < points.cols(); ++col) {
    points.col(col) = rows[static_cast<std::size_t>(col)];
  }
  return points;
}

void readSolverArgs(JsonReader& reader, neupan_uav::NrmpSolverOptions& out) {
  reader.expect('{');
  if (!reader.consume('}')) {
    do {
      const std::string key = reader.string();
      reader.expect(':');
      if (key == "eps_abs") {
        out.eps_abs = reader.number();
      } else if (key == "eps_rel") {
        out.eps_rel = reader.number();
      } else if (key == "max_iter") {
        out.max_iter = static_cast<int>(reader.number());
      } else if (key == "verbose") {
        out.verbose = reader.boolean();
      } else if (key == "polishing") {
        out.polishing = reader.boolean();
      } else if (key == "warm_starting") {
        out.warm_starting = reader.boolean();
      } else {
        reader.skipValue();
      }
    } while (reader.consume(','));
    reader.expect('}');
  }
}

void readRobot(JsonReader& reader, ReplayPlannerInputs& planner) {
  double length = 2.0 * planner.options.robot.body_half_extent(0);
  double width = 2.0 * planner.options.robot.body_half_extent(1);
  double height = 2.0 * planner.options.robot.body_half_extent(2);
  reader.expect('{');
  if (!reader.consume('}')) {
    do {
      const std::string key = reader.string();
      reader.expect(':');
      if (key == "length") {
        length = reader.number();
      } else if (key == "width") {
        width = reader.number();
      } else if (key == "height") {
        height = reader.number();
      } else if (key == "max_speed") {
        planner.options.robot.max_control = readControlArray(reader);
      } else if (key == "max_acce") {
        planner.dynamics.max_acceleration = readControlArray(reader);
      } else if (key == "tau_velocity") {
        planner.dynamics.velocity_time_constant = readVector3(reader);
      } else if (key == "gain_velocity") {
        planner.dynamics.velocity_gain = readVector3(reader);
      } else if (key == "tau_yaw_rate") {
        planner.dynamics.yaw_rate_time_constant = reader.number();
      } else if (key == "gain_yaw_rate") {
        planner.dynamics.yaw_rate_gain = reader.number();
      } else if (key == "velocity_weight_scale") {
        planner.dynamics.velocity_weight_scale = reader.number();
      } else {
        reader.skipValue();
      }
    } while (reader.consume(','));
    reader.expect('}');
  }
  planner.options.robot.body_half_extent =
      Eigen::Vector3d(std::max(0.0, length * 0.5),
                      std::max(0.0, width * 0.5),
                      std::max(0.0, height * 0.5));
}

void readIpath(JsonReader& reader, ReplayPlannerInputs& planner) {
  reader.expect('{');
  if (!reader.consume('}')) {
    do {
      const std::string key = reader.string();
      reader.expect(':');
      if (key == "waypoints") {
        planner.options.initial_path.waypoints.clear();
        reader.expect('[');
        if (!reader.consume(']')) {
          do {
            planner.options.initial_path.waypoints.push_back(readVector4(reader));
          } while (reader.consume(','));
          reader.expect(']');
        }
      } else {
        reader.skipValue();
      }
    } while (reader.consume(','));
    reader.expect('}');
  }
  if (!planner.options.initial_path.waypoints.empty()) {
    planner.options.has_goal = true;
    planner.options.goal_position =
        planner.options.initial_path.waypoints.back().head<3>();
  }
}

void readPreselect(JsonReader& reader,
                   neupan_uav::ObstaclePreselectorConfig& out) {
  reader.expect('{');
  if (!reader.consume('}')) {
    do {
      const std::string key = reader.string();
      reader.expect(':');
      if (key == "enable") {
        out.enabled = reader.boolean();
      } else if (key == "max_points" || key == "max_num") {
        out.max_points = static_cast<std::size_t>(reader.number());
      } else if (key == "per_step") {
        out.per_step = static_cast<int>(reader.number());
      } else if (key == "nearest_ratio" || key == "nearest_quota_ratio") {
        out.nearest_ratio = reader.number();
      } else if (key == "temporal_enable") {
        out.temporal_enabled = reader.boolean();
      } else if (key == "temporal_ratio" || key == "temporal_quota_ratio") {
        out.temporal_ratio = reader.number();
      } else if (key == "diversity_enable") {
        out.diversity_enabled = reader.boolean();
      } else if (key == "diversity_ratio" || key == "diversity_quota_ratio") {
        out.diversity_ratio = reader.number();
      } else if (key == "corridor_margin") {
        out.corridor_margin = readVector3(reader);
      } else if (key == "exact_margin") {
        out.exact_margin = readVector3(reader);
      } else {
        reader.skipValue();
      }
    } while (reader.consume(','));
    reader.expect('}');
  }
}

void readFarfield(JsonReader& reader, neupan_uav::FarfieldGuideConfig& out) {
  reader.expect('{');
  if (!reader.consume('}')) {
    do {
      const std::string key = reader.string();
      reader.expect(':');
      if (key == "enable") {
        out.enabled = reader.boolean();
      } else if (key == "range_backoff") {
        out.range_backoff = reader.number();
      } else if (key == "range_scale") {
        out.range_scale = reader.number();
      } else if (key == "range_far_limit") {
        out.range_far_limit = reader.number();
      } else if (key == "lateral_width") {
        out.lateral_width = reader.number();
      } else if (key == "center_width") {
        out.center_width = reader.number();
      } else if (key == "height_window") {
        out.height_window = reader.number();
      } else if (key == "voxel_size") {
        out.voxel_size = readVector3(reader);
      } else if (key == "trigger_count") {
        out.trigger_count = static_cast<int>(reader.number());
      } else if (key == "offset_min") {
        out.offset_min = reader.number();
      } else if (key == "offset_max") {
        out.offset_max = reader.number();
      } else if (key == "offset_speed_gain") {
        out.offset_speed_gain = reader.number();
      } else if (key == "offset_alpha") {
        out.offset_alpha = reader.number();
      } else if (key == "release_alpha") {
        out.release_alpha = reader.number();
      } else if (key == "release_count") {
        out.release_count = static_cast<int>(reader.number());
      } else if (key == "release_confirm_cycles") {
        out.release_confirm_cycles = static_cast<int>(reader.number());
      } else {
        reader.skipValue();
      }
    } while (reader.consume(','));
    reader.expect('}');
  }
}

void readPan(JsonReader& reader, ReplayPlannerInputs& planner) {
  reader.expect('{');
  if (!reader.consume('}')) {
    do {
      const std::string key = reader.string();
      reader.expect(':');
      if (key == "iter_num") {
        planner.options.pan.iter_num = static_cast<int>(reader.number());
      } else if (key == "dune_max_num") {
        planner.dune_max_num = static_cast<int>(reader.number());
      } else if (key == "nrmp_max_num") {
        planner.options.nrmp.max_constraints = static_cast<int>(reader.number());
      } else if (key == "dune_select_nearest_ratio") {
        planner.dune_options.select_nearest_ratio = reader.number();
      } else if (key == "dune_select_temporal_ratio") {
        planner.dune_options.select_temporal_ratio = reader.number();
      } else if (key == "dune_select_diversity_ratio") {
        planner.dune_options.select_diversity_ratio = reader.number();
      } else {
        reader.skipValue();
      }
    } while (reader.consume(','));
    reader.expect('}');
  }
}

void readAdjust(JsonReader& reader, ReplayPlannerInputs& planner) {
  reader.expect('{');
  if (!reader.consume('}')) {
    do {
      const std::string key = reader.string();
      reader.expect(':');
      if (key == "q_s") {
        planner.state_weight_gain = reader.number();
      } else if (key == "p_u") {
        planner.options.pan.p_u = reader.number();
      } else if (key == "eta") {
        planner.options.pan.eta = reader.number();
      } else if (key == "d_min") {
        planner.options.pan.d_min = reader.number();
      } else if (key == "d_max") {
        planner.options.pan.d_max = reader.number();
      } else if (key == "ro_obs") {
        planner.options.pan.ro_obs = reader.number();
      } else if (key == "bk") {
        planner.options.pan.bk = reader.number();
      } else if (key == "enable_control_smoothing") {
        planner.options.nrmp.enable_control_smoothing = reader.boolean();
      } else if (key == "smooth_du") {
        planner.options.pan.smooth_du = reader.number();
      } else if (key == "smooth_u0") {
        planner.options.pan.smooth_u0 = reader.number();
      } else if (key == "solver_args") {
        readSolverArgs(reader, planner.options.nrmp.solver);
      } else {
        reader.skipValue();
      }
    } while (reader.consume(','));
    reader.expect('}');
  }
}

void readConfig(JsonReader& reader, Fixture& fixture) {
  reader.expect('{');
  if (!reader.consume('}')) {
    do {
      const std::string key = reader.string();
      reader.expect(':');
      if (key == "receding") {
        fixture.planner.options.grid.horizon_steps =
            static_cast<int>(reader.number());
      } else if (key == "step_time") {
        fixture.planner.options.grid.dt = reader.number();
      } else if (key == "ref_speed") {
        fixture.planner.options.ref_speed = reader.number();
      } else if (key == "collision_threshold") {
        fixture.planner.options.collision_threshold = reader.number();
      } else if (key == "arrive_threshold") {
        fixture.planner.options.arrive_threshold = reader.number();
      } else if (key == "robot") {
        readRobot(reader, fixture.planner);
      } else if (key == "ipath") {
        readIpath(reader, fixture.planner);
      } else if (key == "preselect") {
        readPreselect(reader, fixture.planner.options.preselect);
      } else if (key == "farfield_guide") {
        readFarfield(reader, fixture.planner.options.farfield_guide);
      } else if (key == "pan") {
        readPan(reader, fixture.planner);
      } else if (key == "adjust") {
        readAdjust(reader, fixture.planner);
      } else {
        reader.skipValue();
      }
    } while (reader.consume(','));
    reader.expect('}');
  }
}

FrameFixture readFrame(JsonReader& reader) {
  FrameFixture frame;
  reader.expect('{');
  if (!reader.consume('}')) {
    do {
      const std::string key = reader.string();
      reader.expect(':');
      if (key == "state") {
        frame.state = readVector(reader);
      } else if (key == "obstacle_points") {
        frame.obstacle_points = readPointRows(reader);
      } else {
        reader.skipValue();
      }
    } while (reader.consume(','));
    reader.expect('}');
  }
  return frame;
}

neupan_uav::UavState uavStateFromFixture(const Eigen::VectorXd& vector) {
  if (vector.size() != 4 && vector.size() != 8) {
    throw std::runtime_error("frame state must be [x,y,z,yaw] or dynamics state");
  }
  neupan_uav::UavState state;
  state.position_world = vector.head<3>();
  state.attitude_world_body =
      Eigen::Quaterniond(Eigen::AngleAxisd(vector(3), Eigen::Vector3d::UnitZ()));
  if (vector.size() == 8) {
    state.velocity_world = vector.segment<3>(4);
    state.yaw_rate = vector(7);
  }
  return state;
}

void readFrames(JsonReader& reader, Fixture& fixture) {
  reader.expect('[');
  if (!reader.consume(']')) {
    do {
      fixture.frames.push_back(readFrame(reader));
    } while (reader.consume(','));
    reader.expect(']');
  }
}

Fixture readFixture(const std::string& path) {
  JsonReader reader(readTextFile(path));
  Fixture fixture;
  reader.expect('{');
  if (!reader.consume('}')) {
    do {
      const std::string key = reader.string();
      reader.expect(':');
      if (key == "config") {
        readConfig(reader, fixture);
      } else if (key == "frames") {
        readFrames(reader, fixture);
      } else {
        reader.skipValue();
      }
    } while (reader.consume(','));
    reader.expect('}');
  }
  if (fixture.frames.empty()) {
    throw std::runtime_error("fixture must contain at least one frame");
  }
  return fixture;
}

std::optional<neupan_uav::RknnMetadata> makeReplayMetadata(
    const ReplayPlannerInputs& planner) {
  std::optional<neupan_uav::RknnMetadata> metadata;
  if (planner.dune_max_num > 0) {
    metadata = neupan_uav::RknnMetadata();
    metadata->receding = planner.options.grid.horizon_steps;
    metadata->dune_max_num = planner.dune_max_num;
    metadata->max_points = static_cast<int>(std::max(
        planner.options.preselect.max_points,
        static_cast<std::size_t>(planner.dune_max_num)));
    metadata->output_dim = 6;
    metadata->input_shape = {{1, metadata->max_points, 3}};
    metadata->output_shape = {{1, metadata->max_points, 6}};
    metadata->half_extent = planner.options.robot.body_half_extent;
    metadata->scene_scale = Eigen::Vector3d::Ones();
    metadata->clearance_scale = Eigen::Vector3d::Ones();
  }
  return metadata;
}

void validateReplayPlannerConfig(const neupan_uav::CompiledPlannerConfig& config) {
  const neupan_uav::NrmpConfig& nrmp = config.pan().nrmp;
  if (nrmp.state_dim != 8 || nrmp.control_dim != 4 ||
      nrmp.geom_dim != 3 || nrmp.point_dim != 3) {
    throw std::runtime_error(
        "compiled replay NRMP dimensions must be state=8 control=4 geom=3 point=3");
  }
}

neupan_uav::CompiledPlannerConfig makePlannerConfig(const Fixture& fixture) {
  ReplayPlannerInputs planner = fixture.planner;
  std::optional<neupan_uav::RknnMetadata> metadata =
      makeReplayMetadata(planner);
  if (metadata.has_value()) {
    planner.options.dune = planner.dune_options;
  }

  neupan_uav::CompiledPlannerConfig config = neupan_uav::compilePlannerConfig(
      planner.options, planner.dynamics, planner.state_weight_gain, metadata);
  validateReplayPlannerConfig(config);
  return config;
}

void writeVector(std::ostream& out, const Eigen::VectorXd& vector) {
  out << "[";
  for (Eigen::Index i = 0; i < vector.size(); ++i) {
    if (i > 0) out << ",";
    out << vector(i);
  }
  out << "]";
}

void writeJsonNumber(std::ostream& out, double value) {
  if (std::isinf(value)) {
    out << (value > 0.0 ? "Infinity" : "-Infinity");
  } else if (std::isnan(value)) {
    out << "NaN";
  } else {
    out << value;
  }
}

void writeMatrix(std::ostream& out, const Eigen::MatrixXd& matrix) {
  out << "[";
  for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
    if (row > 0) out << ",";
    out << "[";
    for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
      if (col > 0) out << ",";
      out << matrix(row, col);
    }
    out << "]";
  }
  out << "]";
}

void writeReport(const std::string& path,
                 const std::vector<neupan_uav::PlannerResult>& outputs) {
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ofstream out(path);
  if (!out) throw std::runtime_error("failed to open output: " + path);
  out << std::setprecision(12);
  out << "{\n";
  out << "  \"runtime\": \"cpp\",\n";
  out << "  \"frame_count\": " << outputs.size() << ",\n";
  out << "  \"frames\": [\n";
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto& output = outputs[i];
    const auto* tracking = std::get_if<neupan_uav::Tracking>(&output.decision());
    const auto* goal =
        std::get_if<neupan_uav::GoalReached>(&output.decision());
    const auto* safety_stop =
        std::get_if<neupan_uav::SafetyStop>(&output.decision());
    const auto* fault = std::get_if<neupan_uav::FaultStop>(&output.decision());
    if (i > 0) out << ",\n";
    out << "    {\n";
    out << "      \"frame_index\": " << i << ",\n";
    out << "      \"state\": \"";
    if (tracking != nullptr) {
      out << "tracking";
    } else if (goal != nullptr) {
      out << "goal_reached";
    } else if (safety_stop != nullptr) {
      out << "safety_stop";
    } else if (fault != nullptr) {
      out << "fault";
    } else {
      out << "unknown";
    }
    out << "\",\n";
    if (safety_stop != nullptr) {
      out << "      \"safety_stop_cause\": \""
          << neupan_uav::toString(safety_stop->cause) << "\",\n";
      out << "      \"observed_clearance\": ";
      writeJsonNumber(out, safety_stop->observed_clearance);
      out << ",\n      \"required_clearance\": ";
      writeJsonNumber(out, safety_stop->required_clearance);
      out << ",\n";
    }
    if (fault != nullptr) {
      out << "      \"fault\": \"" << neupan_uav::toString(fault->fault)
          << "\",\n";
      out << "      \"detail\": \"" << fault->detail << "\",\n";
    }
    out << "      \"command\": ";
    writeVector(out, output.commandToPublish());
    out << ",\n      \"trajectory\": ";
    writeMatrix(out, tracking != nullptr ? tracking->plan.trajectory
                                         : neupan_uav::Trajectory());
    out << ",\n      \"reference\": ";
    writeMatrix(out, tracking != nullptr ? tracking->plan.reference
                                         : neupan_uav::Trajectory());
    out << ",\n      \"control_trajectory\": ";
    writeMatrix(out, tracking != nullptr ? tracking->plan.control_trajectory
                                         : Eigen::MatrixXd());
    out << ",\n      \"nominal_distance\": ";
    writeVector(out,
                tracking != nullptr ? tracking->plan.nominal_distance.transpose()
                                    : Eigen::VectorXd());
    out << ",\n      \"min_distance\": ";
    if (output.diagnostics().min_clearance.has_value()) {
      writeJsonNumber(out, *output.diagnostics().min_clearance);
    } else {
      out << "null";
    }
    out << ",\n";
    out << "      \"profile\": {\n";
    out << "        \"forward_sec\": " << output.diagnostics().profile.forward_sec << ",\n";
    out << "        \"preselect_sec\": " << output.diagnostics().profile.preselect_sec << ",\n";
    out << "        \"dune_sec\": " << output.diagnostics().profile.dune_sec << ",\n";
    out << "        \"dune_inference_sec\": " << output.diagnostics().profile.dune_inference_sec << ",\n";
    out << "        \"dune_select_sec\": " << output.diagnostics().profile.dune_select_sec << ",\n";
    out << "        \"nrmp_sec\": " << output.diagnostics().profile.nrmp_sec << ",\n";
    out << "        \"input_obstacle_count\": " << output.diagnostics().profile.input_obstacle_count << ",\n";
    out << "        \"preselected_obstacle_count\": " << output.diagnostics().profile.preselected_obstacle_count << ",\n";
    out << "        \"dune_selected_count\": " << output.diagnostics().profile.dune_selected_count << ",\n";
    out << "        \"osqp_status\": " << output.diagnostics().profile.osqp_status << ",\n";
    out << "        \"osqp_iteration_count\": " << output.diagnostics().profile.osqp_iteration_count << ",\n";
    out << "        \"osqp_solve_sec\": " << output.diagnostics().profile.osqp_solve_sec << ",\n";
    out << "        \"pan_iterations\": " << output.diagnostics().profile.pan_iterations << ",\n";
    out << "        \"pan_iteration_limit\": " << output.diagnostics().profile.pan_iteration_limit << ",\n";
    out << "        \"farfield_sec\": " << output.diagnostics().profile.farfield_sec << ",\n";
    out << "        \"farfield_active\": " << (output.diagnostics().profile.farfield_active ? "true" : "false") << ",\n";
    out << "        \"farfield_near_m\": " << output.diagnostics().profile.farfield_near_m << ",\n";
    out << "        \"farfield_far_m\": " << output.diagnostics().profile.farfield_far_m << ",\n";
    out << "        \"farfield_offset_m\": " << output.diagnostics().profile.farfield_offset_m << ",\n";
    out << "        \"farfield_target_offset_m\": " << output.diagnostics().profile.farfield_target_offset_m << ",\n";
    out << "        \"farfield_center_count\": " << output.diagnostics().profile.farfield_center_count << ",\n";
    out << "        \"farfield_left_count\": " << output.diagnostics().profile.farfield_left_count << ",\n";
    out << "        \"farfield_right_count\": " << output.diagnostics().profile.farfield_right_count << ",\n";
    out << "        \"farfield_release_streak\": " << output.diagnostics().profile.farfield_release_streak << "\n";
    out << "      }\n";
    out << "    }";
  }
  out << "\n  ]\n";
  out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parseArgs(argc, argv);
    const Fixture fixture = readFixture(args.fixture_path);
    neupan_uav::Planner planner(makePlannerConfig(fixture));
    std::vector<neupan_uav::PlannerResult> outputs;
    outputs.reserve(fixture.frames.size());
    for (const FrameFixture& frame : fixture.frames) {
      neupan_uav::PlannerInput input;
      input.state = uavStateFromFixture(frame.state);
      input.obstacle_points = frame.obstacle_points;
      outputs.push_back(planner.forward(input));
    }
    writeReport(args.output_path, outputs);
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "run_phase6_5_cpp_replay: " << exc.what() << "\n";
    usage(argv[0]);
    return 2;
  }
}
