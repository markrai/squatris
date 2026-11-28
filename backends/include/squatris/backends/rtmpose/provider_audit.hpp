#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "squatris/core/result.hpp"

namespace squatris::backends::rtmpose {

struct ProviderProfile {
  std::string session;
  std::filesystem::path path;
};

[[nodiscard]] Result<nlohmann::json>
inspect_provider_profile(const ProviderProfile& profile);

[[nodiscard]] Result<nlohmann::json>
analyze_provider_profiles(const std::vector<ProviderProfile>& profiles,
                          const std::filesystem::path& policy_manifest,
                          const std::string& detector_sha256, const std::string& pose_sha256,
                          const std::string& runtime_version,
                          const std::string& runtime_header_version,
                          const nlohmann::json& runtime_artifacts,
                          bool verified_sessions_retained);

}
