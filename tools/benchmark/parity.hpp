#pragma once

#include <filesystem>

namespace squatris::benchmark {

struct ParityOptions {
  std::filesystem::path compare_cpu_report;
  std::filesystem::path compare_cuda_report;
  std::filesystem::path parity_output;
};

int compare_parity_reports(const ParityOptions& options);

}
