#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include <nlohmann/json_fwd.hpp>

#include "squatris/core/result.hpp"

namespace squatris::backends::rtmpose {

struct CandidateCudaProfileRequest {
  std::filesystem::path model;
  std::string expected_sha256;
  std::filesystem::path output_directory;
  std::int64_t width{};
  std::int64_t height{};
};

[[nodiscard]] Result<nlohmann::json>
profile_candidate_cuda(const CandidateCudaProfileRequest& request);

}
