#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "squatris/core/input/frame.hpp"
#include "squatris/core/result.hpp"

namespace squatris::input {

struct CaptureFormat {
  std::uint32_t width{};
  std::uint32_t height{};
  double requested_fps{};
  double negotiated_fps{};
  std::string backend;
  std::string encoding;
};

struct SourceStatistics {
  double delivered_fps{};
  std::uint64_t frames_delivered{};
  std::uint64_t frames_overwritten{};
};

class IFrameSource {
public:
  virtual ~IFrameSource() = default;
  [[nodiscard]] virtual Status start() = 0;
  virtual void stop() noexcept = 0;
  [[nodiscard]] virtual Result<std::shared_ptr<const Frame>>
  wait_for_frame(std::uint64_t after_sequence, std::chrono::milliseconds timeout) = 0;
  [[nodiscard]] virtual CaptureFormat format() const = 0;
  [[nodiscard]] virtual SourceStatistics statistics() const = 0;
  [[nodiscard]] virtual std::optional<Status> terminal_error() const = 0;
};

}
