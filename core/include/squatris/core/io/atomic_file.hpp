#pragma once

#include <filesystem>
#include <string_view>

#include "squatris/core/result.hpp"

namespace squatris::io {

[[nodiscard]] Status write_text_atomically(const std::filesystem::path& path,
                                           std::string_view content);

}
