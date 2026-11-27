#pragma once

#include <filesystem>
#include <string>

#include "squatris/core/result.hpp"

namespace squatris::tools {

[[nodiscard]] Result<std::string> sha256_file(const std::filesystem::path& path);

}
