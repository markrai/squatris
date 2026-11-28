#include <algorithm>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "squatris/backends/rtmpose/candidate_cuda_profile.hpp"
#include "squatris/core/crypto/sha256.hpp"
#include "squatris/core/io/atomic_file.hpp"

namespace {

using squatris::backends::rtmpose::CandidateCudaProfileRequest;

std::optional<CandidateCudaProfileRequest> parse(int argc, char** argv) {
  CandidateCudaProfileRequest request;
  if (argc != 11) {
    return std::nullopt;
  }
  for (int index = 1; index < argc; index += 2) {
    const std::string_view key(argv[index]);
    const std::string value(argv[index + 1]);
    if (key == "--model" && request.model.empty()) {
      request.model = value;
    } else if (key == "--sha256" && request.expected_sha256.empty()) {
      request.expected_sha256 = value;
    } else if (key == "--output-directory" && request.output_directory.empty()) {
      request.output_directory = value;
    } else if (key == "--width" || key == "--height") {
      auto& dimension = key == "--width" ? request.width : request.height;
      if (dimension != 0) {
        return std::nullopt;
      }
      const auto parsed = std::from_chars(value.data(), value.data() + value.size(), dimension);
      if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
          dimension <= 0 || dimension > 2048) {
        return std::nullopt;
      }
    } else {
      return std::nullopt;
    }
  }
  if (request.model.empty() || request.output_directory.empty() || request.width == 0 ||
      request.height == 0 || request.expected_sha256.size() != 64 ||
      !std::ranges::all_of(request.expected_sha256, [](char character) {
        return (character >= '0' && character <= '9') ||
               (character >= 'a' && character <= 'f');
      })) {
    return std::nullopt;
  }
  return request;
}

int run(int argc, char** argv) {
  const auto request = parse(argc, argv);
  if (!request) {
    std::cerr << "Usage: SquatrisCandidateProfiler --model PATH --sha256 LOWERCASE_HEX "
                 "--width N --height N --output-directory NEW_DIRECTORY\n";
    return 2;
  }
  const auto executable = std::filesystem::absolute(argv[0]);
  const auto hash = squatris::crypto::sha256_file(executable);
  if (!hash) {
    std::cerr << hash.error().message << '\n';
    return 1;
  }
  auto result = squatris::backends::rtmpose::profile_candidate_cuda(*request);
  if (!result) {
    std::cerr << result.error().message << '\n';
    return 1;
  }
  auto& report = result.value();
  report["executable"] = {{"path", executable.string()}, {"sha256", hash.value()}};
  const auto destination = request->output_directory / "candidate-profile.json";
  const auto written = squatris::io::write_text_atomically(destination, report.dump(2) + '\n');
  if (!written.ok()) {
    std::cerr << written.message << '\n';
    return 1;
  }
  std::cout << "Diagnostic report: " << destination << '\n'
            << report.at("observed_node_counts").dump(2) << '\n'
            << "Review required; no production execution policy was authorized.\n";
  return 0;
}

}

int main(int argc, char** argv) noexcept {
  try {
    return run(argc, argv);
  } catch (const std::exception& exception) {
    std::fputs(exception.what(), stderr);
    std::fputc('\n', stderr);
    return 1;
  }
}
