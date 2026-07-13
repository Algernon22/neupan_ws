#include "neupan_uav/rknn_runner.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " --metadata PATH --input PATH --output PATH"
               " [--core-mask CORE_0]\n";
}

std::string requireValue(int argc, char** argv, int& index,
                         const std::string& option) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(option + " requires a value");
  }
  ++index;
  return argv[index];
}

struct Args {
  std::string metadata_path;
  std::string input_path;
  std::string output_path;
  std::string core_mask = "CORE_0";
};

Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--metadata") {
      args.metadata_path = requireValue(argc, argv, i, option);
    } else if (option == "--input") {
      args.input_path = requireValue(argc, argv, i, option);
    } else if (option == "--output") {
      args.output_path = requireValue(argc, argv, i, option);
    } else if (option == "--core-mask") {
      args.core_mask = requireValue(argc, argv, i, option);
    } else if (option == "--help" || option == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + option);
    }
  }
  if (args.metadata_path.empty() || args.input_path.empty() ||
      args.output_path.empty()) {
    throw std::invalid_argument(
        "--metadata, --input, and --output are required");
  }
  return args;
}

std::vector<neupan_uav::RknnFloatMatrix> readPointFlow(
    const std::string& path, const neupan_uav::RknnMetadata& metadata) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("failed to open input point flow: " + path);
  }

  std::string magic;
  int steps = 0;
  int point_dim = 0;
  int num_points = 0;
  stream >> magic >> steps >> point_dim >> num_points;
  if (magic != "NEUPAN_RKNN_POINT_FLOW_V1") {
    throw std::runtime_error("unexpected point-flow input magic: " + magic);
  }
  if (steps != metadata.receding + 1) {
    throw std::runtime_error("point-flow step count does not match metadata");
  }
  if (point_dim != 3) {
    throw std::runtime_error("point-flow point_dim must be 3");
  }
  if (num_points < 0 || num_points > metadata.dune_max_num) {
    throw std::runtime_error("point-flow num_points is out of range");
  }

  std::vector<neupan_uav::RknnFloatMatrix> point_flow(
      static_cast<std::size_t>(steps));
  for (int step = 0; step < steps; ++step) {
    point_flow[static_cast<std::size_t>(step)].resize(3, num_points);
    for (int dim = 0; dim < 3; ++dim) {
      for (int point = 0; point < num_points; ++point) {
        float value = 0.0F;
        if (!(stream >> value)) {
          throw std::runtime_error("failed to read point-flow value");
        }
        point_flow[static_cast<std::size_t>(step)](dim, point) = value;
      }
    }
  }
  return point_flow;
}

void writeRawMu(const std::string& path,
                const neupan_uav::RknnFloatMatrix& raw_mu) {
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ofstream stream(path);
  if (!stream) {
    throw std::runtime_error("failed to open raw_mu output: " + path);
  }
  stream << "NEUPAN_RKNN_RAW_MU_V1\n";
  stream << raw_mu.rows() << " " << raw_mu.cols() << "\n";
  stream << std::setprecision(10);
  for (Eigen::Index row = 0; row < raw_mu.rows(); ++row) {
    for (Eigen::Index col = 0; col < raw_mu.cols(); ++col) {
      if (col > 0) stream << " ";
      stream << raw_mu(row, col);
    }
    stream << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parseArgs(argc, argv);
    if (!neupan_uav::compiledWithRknnRuntime()) {
      throw std::runtime_error(
          "libneupan_uav was built without RKNN runtime support");
    }
    neupan_uav::RknnRunnerConfig config;
    config.metadata_path = args.metadata_path;
    config.core_mask = args.core_mask;
    config.require_device = true;
    neupan_uav::ObsPointNetRknnRunner runner(config);
    const std::vector<neupan_uav::RknnFloatMatrix> point_flow =
        readPointFlow(args.input_path, runner.metadata());
    const neupan_uav::RknnFloatMatrix raw_mu = runner.inferRawMu(point_flow);
    writeRawMu(args.output_path, raw_mu);
    std::cerr << "C++ RKNN raw_mu rows=" << raw_mu.rows()
              << " cols=" << raw_mu.cols()
              << " inference_sec=" << runner.profile().inference_sec << "\n";
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "dump_rknn_raw_mu: " << exc.what() << "\n";
    usage(argv[0]);
    return 2;
  }
}
