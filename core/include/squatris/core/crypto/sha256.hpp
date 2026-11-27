#pragma once

#include <filesystem>
#include <string>

#include "squatris/core/result.hpp"

namespace squatris::crypto {

[[nodiscard]] Result<std::string> sha256_file(const std::filesystem::path& path);

}
