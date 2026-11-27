#include "squatris/core/io/atomic_file.hpp"

#include <chrono>
#include <fstream>
#include <functional>
#include <string>
#include <system_error>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace squatris::io {

Status write_text_atomically(const std::filesystem::path& path, std::string_view content) {
  const auto suffix = std::to_string(
                          std::chrono::steady_clock::now().time_since_epoch().count()) +
                      "-" + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));
  const std::filesystem::path temporary(path.string() + ".tmp." + suffix);
  const auto remove_temporary = [&] {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
  };
  {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) {
      remove_temporary();
      return {ErrorCode::io_error, "Could not create temporary output file: " + path.string()};
    }
    stream.write(content.data(), static_cast<std::streamsize>(content.size()));
    stream.flush();
    if (!stream) {
      stream.close();
      remove_temporary();
      return {ErrorCode::io_error, "Could not write temporary output file: " + path.string()};
    }
  }
#ifdef _WIN32
  if (!MoveFileExW(temporary.c_str(), path.c_str(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    remove_temporary();
    return {ErrorCode::io_error, "Could not atomically publish output file: " + path.string()};
  }
#else
  std::error_code rename_error;
  std::filesystem::rename(temporary, path, rename_error);
  if (rename_error) {
    remove_temporary();
    return {ErrorCode::io_error,
            "Could not atomically publish output file: " + rename_error.message()};
  }
#endif
  return Status::success();
}

}
