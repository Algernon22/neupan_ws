#include "neupan_uav/planner.hpp"

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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

struct RobotFixture {
  double length = 0.46;
  double width = 0.46;
  double height = 0.12;
  neupan_uav::Control max_speed =
      neupan_uav::Control::Constant(std::numeric_limits<double>::infinity());
  neupan_uav::Control max_acce =
      neupan_uav::Control::Constant(std::numeric_limits<double>::infinity());
  Eigen::Vector3d tau_velocity = Eigen::Vector3d(0.35, 0.35, 0.45);
  Eigen::Vector3d gain_velocity = Eigen::Vector3d::Ones();
  double tau_yaw_rate = 0.25;
  double gain_yaw_rate = 1.0;
  double velocity_weight_scale = 0.35;
};

struct AdjustFixture {
  double q_s = 1.0;
  double p_u = 1.0;
  bool enable_control_smoothing = false;
  double smooth_du = 0.0;
  double smooth_u0 = 0.0;
  double eps_abs = 1.0e-4;
  double eps_rel = 1.0e-4;
  int max_iter = 4000;
  bool verbose = false;
  bool polishing = false;
  bool warm_starting = true;
};

struct PreselectFixture {
  bool enable = true;
  std::size_t max_points = 0;
  int per_step = 0;
  double nearest_ratio = 1.0;
  double temporal_ratio = 0.0;
  double diversity_ratio = 0.0;
  Eigen::Vector3d corridor_margin = Eigen::Vector3d::Constant(10.0);
  Eigen::Vector3d exact_margin = Eigen::Vector3d::Zero();
};

struct FarfieldFixture {
  bool enable = false;
  double range_backoff = 1.0;
  double range_scale = 1.5;
  double range_far_limit = std::numeric_limits<double>::infinity();
  double lateral_width = 5.0;
  double center_width = 2.0;
  double height_window = 1.5;
  Eigen::Vector3d voxel_size = Eigen::Vector3d(1.0, 1.0, 0.8);
  int trigger_count = 8;
  double offset_min = 1.5;
  double offset_max = 4.0;
  double offset_speed_gain = 0.6;
  double offset_alpha = 0.20;
  double release_alpha = 0.08;
  int release_count = -1;
  int release_confirm_cycles = 3;
};

struct PanFixture {
  int iter_num = 1;
  int dune_max_num = 0;
  int nrmp_max_num = 0;
};

struct FrameFixture {
  double stamp_sec = 0.0;
  Eigen::VectorXd state;
  neupan_uav::PointMatrix obstacle_points = neupan_uav::emptyPointMatrix();
};

struct Fixture {
  int receding = 4;
  double step_time = 0.1;
  double ref_speed = 1.0;
  double collision_threshold = 0.05;
  double arrive_threshold = 0.15;
  RobotFixture robot;
  std::vector<Eigen::Vector4d> waypoints;
  PreselectFixture preselect;
  FarfieldFixture farfield;
  PanFixture pan;
  AdjustFixture adjust;
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

void readSolverArgs(JsonReader& reader, AdjustFixture& out) {
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

void readRobot(JsonReader& reader, RobotFixture& out) {
  reader.expect('{');
  if (!reader.consume('}')) {
    do {
      const std::string key = reader.string();
      reader.expect(':');
      if (key == "length") {
        out.length = reader.number();
      } else if (key == "width") {
        out.width = reader.number();
      } else if (key == "height") {
        out.height = reader.number();
      } else if (key == "max_speed") {
        out.max_speed = readControlArray(reader);
      } else if (key == "max_acce") {
        out.max_acce = readControlArray(reader);
      } else if (key == "tau_velocity") {
        out.tau_velocity = readVector3(reader);
      } else if (key == "gain_velocity") {
        out.gain_velocity = readVector3(reader);
      } else if (key == "tau_yaw_rate") {
        out.tau_yaw_rate = reader.number();
      } else if (key == "gain_yaw_rate") {
        out.gain_yaw_rate = reader.number();
      } else if (key == "velocity_weight_scale") {
        out.velocity_weight_scale = reader.number();
      } else {
        reader.skipValue();
      }
    } while (reader.consume(','));
    reader.expect('}');
  }
}

void readIpath(JsonReader& reader, Fixture& fixture) {
  reader.expect('{');
  if (!reader.consume('}')) {
    do {
      const std::string key = reader.string();
      reader.expect(':');
      if (key == "waypoints") {
        reader.expect('[');
        if (!reader.consume(']')) {
          do {
            fixture.waypoints.push_back(readVector4(reader));
          } while (reader.consume(','));
          reader.expect(']');
        }
      } else {
        reader.skipValue();
      }
    } while (reader.consume(','));
    reader.expect('}');
  }
}

void readPreselect(JsonReader& reader, PreselectFixture& out) {
  reader.expect('{');
  if (!reader.consume('}')) {
    do {
      const std::string key = reader.string();
      reader.expect(':');
      if (key == "enable") {
        out.enable = reader.boolean();
      } else if (key == "max_points" || key == "max_num") {
        out.max_points = static_cast<std::size_t>(reader.number());
      } else if (key == "per_step") {
        out.per_step = static_cast<int>(reader.number());
      } else if (key == "nearest_ratio" || key == "nearest_quota_ratio") {
        out.nearest_ratio = reader.number();
      } else if (key == "temporal_ratio" || key == "temporal_quota_ratio") {
        out.temporal_ratio = reader.number();
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

void readFarfield(JsonReader& reader, FarfieldFixture& out) {
  reader.expect('{');
  if (!reader.consume('}')) {
    do {
      const std::string key = reader.string();
      reader.expect(':');
      if (key == "enable") {
        out.enable = reader.boolean();
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

void readPan(JsonReader& reader, PanFixture& out) {
  reader.expect('{');
  if (!reader.consume('}')) {
    do {
      const std::string key = reader.string();
      reader.expect(':');
      if (key == "iter_num") {
        out.iter_num = static_cast<int>(reader.number());
      } else if (key == "dune_max_num") {
        out.dune_max_num = static_cast<int>(reader.number());
      } else if (key == "nrmp_max_num") {
        out.nrmp_max_num = static_cast<int>(reader.number());
      } else {
        reader.skipValue();
      }
    } while (reader.consume(','));
    reader.expect('}');
  }
}

void readAdjust(JsonReader& reader, AdjustFixture& out) {
  reader.expect('{');
  if (!reader.consume('}')) {
    do {
      const std::string key = reader.string();
      reader.expect(':');
      if (key == "q_s") {
        out.q_s = reader.number();
      } else if (key == "p_u") {
        out.p_u = reader.number();
      } else if (key == "enable_control_smoothing") {
        out.enable_control_smoothing = reader.boolean();
      } else if (key == "smooth_du") {
        out.smooth_du = reader.number();
      } else if (key == "smooth_u0") {
        out.smooth_u0 = reader.number();
      } else if (key == "solver_args") {
        readSolverArgs(reader, out);
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
        fixture.receding = static_cast<int>(reader.number());
      } else if (key == "step_time") {
        fixture.step_time = reader.number();
      } else if (key == "ref_speed") {
        fixture.ref_speed = reader.number();
      } else if (key == "collision_threshold") {
        fixture.collision_threshold = reader.number();
      } else if (key == "arrive_threshold") {
        fixture.arrive_threshold = reader.number();
      } else if (key == "robot") {
        readRobot(reader, fixture.robot);
      } else if (key == "ipath") {
        readIpath(reader, fixture);
      } else if (key == "preselect") {
        readPreselect(reader, fixture.preselect);
      } else if (key == "farfield_guide") {
        readFarfield(reader, fixture.farfield);
      } else if (key == "pan") {
        readPan(reader, fixture.pan);
      } else if (key == "adjust") {
        readAdjust(reader, fixture.adjust);
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
      if (key == "stamp_sec") {
        frame.stamp_sec = reader.number();
      } else if (key == "state") {
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

Eigen::MatrixXd continuousA(const RobotFixture& robot) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(8, 8);
  A(0, 4) = 1.0;
  A(1, 5) = 1.0;
  A(2, 6) = 1.0;
  A(3, 7) = 1.0;
  for (int i = 0; i < 3; ++i) {
    A(4 + i, 4 + i) = -1.0 / robot.tau_velocity(i);
  }
  A(7, 7) = -1.0 / robot.tau_yaw_rate;
  return A;
}

Eigen::MatrixXd continuousB(const RobotFixture& robot) {
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(8, 4);
  for (int i = 0; i < 3; ++i) {
    B(4 + i, i) = robot.gain_velocity(i) / robot.tau_velocity(i);
  }
  B(7, 3) = robot.gain_yaw_rate / robot.tau_yaw_rate;
  return B;
}

void discretizeDynamics(const RobotFixture& robot, double dt,
                        Eigen::MatrixXd& A, Eigen::MatrixXd& B,
                        Eigen::VectorXd& C) {
  const Eigen::MatrixXd Ac = continuousA(robot);
  const Eigen::MatrixXd Bc = continuousB(robot);
  Eigen::MatrixXd aug = Eigen::MatrixXd::Zero(12, 12);
  aug.block(0, 0, 8, 8) = Ac;
  aug.block(0, 8, 8, 4) = Bc;
  const Eigen::MatrixXd expm = (aug * dt).exp();
  A = expm.block(0, 0, 8, 8);
  B = expm.block(0, 8, 8, 4);
  C = Eigen::VectorXd::Zero(8);
}

neupan_uav::PlannerConfig makePlannerConfig(const Fixture& fixture) {
  neupan_uav::PlannerConfig config;
  config.receding = fixture.receding;
  config.step_time = fixture.step_time;
  config.ref_speed = fixture.ref_speed;
  config.collision_threshold = fixture.collision_threshold;
  config.arrive_threshold = fixture.arrive_threshold;
  config.robot.body_half_extent =
      Eigen::Vector3d(fixture.robot.length * 0.5,
                      fixture.robot.width * 0.5,
                      fixture.robot.height * 0.5);
  config.robot.max_control = fixture.robot.max_speed;
  config.initial_path.waypoints = fixture.waypoints;
  config.preselect.enabled = fixture.preselect.enable;
  config.preselect.max_points = fixture.preselect.max_points;
  config.preselect.per_step = fixture.preselect.per_step;
  config.preselect.nearest_ratio = fixture.preselect.nearest_ratio;
  config.preselect.temporal_ratio = fixture.preselect.temporal_ratio;
  config.preselect.diversity_ratio = fixture.preselect.diversity_ratio;
  config.preselect.corridor_margin = fixture.preselect.corridor_margin;
  config.preselect.exact_margin = fixture.preselect.exact_margin;
  config.farfield_guide.enabled = fixture.farfield.enable;
  config.farfield_guide.range_backoff = fixture.farfield.range_backoff;
  config.farfield_guide.range_scale = fixture.farfield.range_scale;
  config.farfield_guide.range_far_limit = fixture.farfield.range_far_limit;
  config.farfield_guide.lateral_width = fixture.farfield.lateral_width;
  config.farfield_guide.center_width = fixture.farfield.center_width;
  config.farfield_guide.height_window = fixture.farfield.height_window;
  config.farfield_guide.voxel_size = fixture.farfield.voxel_size;
  config.farfield_guide.trigger_count = fixture.farfield.trigger_count;
  config.farfield_guide.offset_min = fixture.farfield.offset_min;
  config.farfield_guide.offset_max = fixture.farfield.offset_max;
  config.farfield_guide.offset_speed_gain =
      fixture.farfield.offset_speed_gain;
  config.farfield_guide.offset_alpha = fixture.farfield.offset_alpha;
  config.farfield_guide.release_alpha = fixture.farfield.release_alpha;
  config.farfield_guide.release_count = fixture.farfield.release_count;
  config.farfield_guide.release_confirm_cycles =
      fixture.farfield.release_confirm_cycles;

  config.pan.iter_num = fixture.pan.iter_num;
  config.pan.iter_threshold = 1.0e-9;
  config.pan.dune_max_num = fixture.pan.dune_max_num;
  config.pan.nrmp_max_num = fixture.pan.nrmp_max_num;
  config.pan.has_nrmp_config = true;
  config.pan.nrmp.receding = fixture.receding;
  config.pan.nrmp.state_dim = 8;
  config.pan.nrmp.control_dim = 4;
  config.pan.nrmp.geom_dim = 4;
  config.pan.nrmp.point_dim = 3;
  config.pan.nrmp.max_num = fixture.pan.nrmp_max_num;
  config.pan.nrmp.no_obs = fixture.pan.nrmp_max_num <= 0;
  config.pan.nrmp.enable_control_smoothing =
      fixture.adjust.enable_control_smoothing;
  discretizeDynamics(fixture.robot, fixture.step_time,
                     config.pan.nrmp.dynamics_A,
                     config.pan.nrmp.dynamics_B,
                     config.pan.nrmp.dynamics_C);
  config.pan.nrmp.speed_bound = fixture.robot.max_speed;
  config.pan.nrmp.acce_bound = fixture.robot.max_acce * fixture.step_time;
  config.pan.nrmp.tracking_speed_bound =
      fixture.robot.max_speed.cwiseAbs();
  config.pan.nrmp.solver_options.eps_abs = fixture.adjust.eps_abs;
  config.pan.nrmp.solver_options.eps_rel = fixture.adjust.eps_rel;
  config.pan.nrmp.solver_options.max_iter = fixture.adjust.max_iter;
  config.pan.nrmp.solver_options.verbose = fixture.adjust.verbose;
  config.pan.nrmp.solver_options.polishing = fixture.adjust.polishing;
  config.pan.nrmp.solver_options.warm_starting = fixture.adjust.warm_starting;
  config.pan.state_weights = Eigen::VectorXd::Ones(8);
  const double dynamic_weight = fixture.adjust.q_s *
                                fixture.robot.velocity_weight_scale;
  config.pan.state_weights.tail<4>().setConstant(dynamic_weight);
  config.pan.p_u = fixture.adjust.p_u;
  config.pan.smooth_du = fixture.adjust.smooth_du;
  config.pan.smooth_u0 = fixture.adjust.smooth_u0;
  config.pan.point_flow.receding = fixture.receding;
  config.pan.point_flow.dune_max_num =
      static_cast<std::size_t>(fixture.pan.dune_max_num);
  config.pan.point_flow.body_half_extent = config.robot.body_half_extent;
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

void writeReport(const std::string& path, const Fixture& fixture,
                 const std::vector<neupan_uav::PlannerOutput>& outputs) {
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
    if (i > 0) out << ",\n";
    out << "    {\n";
    out << "      \"frame_index\": " << i << ",\n";
    out << "      \"stamp_sec\": " << fixture.frames[i].stamp_sec << ",\n";
    out << "      \"ready\": " << (output.ready ? "true" : "false") << ",\n";
    out << "      \"reason\": \"" << output.reason << "\",\n";
    out << "      \"arrive\": " << (output.arrive ? "true" : "false") << ",\n";
    out << "      \"stop\": " << (output.stop ? "true" : "false") << ",\n";
    out << "      \"command\": ";
    writeVector(out, output.command);
    out << ",\n      \"trajectory\": ";
    writeMatrix(out, output.trajectory);
    out << ",\n      \"reference\": ";
    writeMatrix(out, output.reference);
    out << ",\n      \"control_trajectory\": ";
    writeMatrix(out, output.control_trajectory);
    out << ",\n      \"nominal_distance\": ";
    writeVector(out, output.nominal_distance.transpose());
    out << ",\n      \"min_distance\": ";
    writeJsonNumber(out, output.min_distance);
    out << ",\n";
    out << "      \"profile\": {\n";
    out << "        \"forward_sec\": " << output.profile.forward_sec << ",\n";
    out << "        \"preselect_sec\": " << output.profile.preselect_sec << ",\n";
    out << "        \"dune_sec\": " << output.profile.dune_sec << ",\n";
    out << "        \"dune_inference_sec\": " << output.profile.dune_inference_sec << ",\n";
    out << "        \"dune_select_sec\": " << output.profile.dune_select_sec << ",\n";
    out << "        \"nrmp_sec\": " << output.profile.nrmp_sec << ",\n";
    out << "        \"input_obstacle_count\": " << output.profile.input_obstacle_count << ",\n";
    out << "        \"preselected_obstacle_count\": " << output.profile.preselected_obstacle_count << ",\n";
    out << "        \"dune_selected_count\": " << output.profile.dune_selected_count << ",\n";
    out << "        \"osqp_status\": " << output.profile.osqp_status << ",\n";
    out << "        \"osqp_iteration_count\": " << output.profile.osqp_iteration_count << ",\n";
    out << "        \"osqp_solve_sec\": " << output.profile.osqp_solve_sec << ",\n";
    out << "        \"pan_iterations\": " << output.profile.pan_iterations << ",\n";
    out << "        \"pan_iteration_limit\": " << output.profile.pan_iteration_limit << ",\n";
    out << "        \"farfield_sec\": " << output.profile.farfield_sec << ",\n";
    out << "        \"farfield_active\": " << (output.profile.farfield_active ? "true" : "false") << ",\n";
    out << "        \"farfield_near_m\": " << output.profile.farfield_near_m << ",\n";
    out << "        \"farfield_far_m\": " << output.profile.farfield_far_m << ",\n";
    out << "        \"farfield_offset_m\": " << output.profile.farfield_offset_m << ",\n";
    out << "        \"farfield_target_offset_m\": " << output.profile.farfield_target_offset_m << ",\n";
    out << "        \"farfield_center_count\": " << output.profile.farfield_center_count << ",\n";
    out << "        \"farfield_left_count\": " << output.profile.farfield_left_count << ",\n";
    out << "        \"farfield_right_count\": " << output.profile.farfield_right_count << ",\n";
    out << "        \"farfield_release_streak\": " << output.profile.farfield_release_streak << "\n";
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
    std::vector<neupan_uav::PlannerOutput> outputs;
    outputs.reserve(fixture.frames.size());
    for (const FrameFixture& frame : fixture.frames) {
      neupan_uav::PlannerInput input;
      input.state = uavStateFromFixture(frame.state);
      input.obstacle_points = frame.obstacle_points;
      input.stamp_sec = frame.stamp_sec;
      outputs.push_back(planner.forward(input));
    }
    writeReport(args.output_path, fixture, outputs);
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "run_phase6_5_cpp_replay: " << exc.what() << "\n";
    usage(argv[0]);
    return 2;
  }
}
