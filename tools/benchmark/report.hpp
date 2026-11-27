#pragma once

#include <filesystem>

#include <nlohmann/json_fwd.hpp>

#include "squatris/core/result.hpp"

namespace squatris::benchmark {

[[nodiscard]] Result<nlohmann::json> load_json(const std::filesystem::path& path);
[[nodiscard]] Status write_report(const std::filesystem::path& path,
                                  const nlohmann::json& report);
[[nodiscard]] Status validate_report(const nlohmann::json& report);
int validate_report_file(const std::filesystem::path& path);

}
