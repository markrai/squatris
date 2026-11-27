#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

#include "squatris/core/input/frame.hpp"
#include "squatris/core/result.hpp"

namespace squatris::input {

class LatestFrameBuffer {
public:
  void publish(std::shared_ptr<const Frame> frame) {
    {
      std::lock_guard lock(mutex_);
      if (closed_) {
        return;
      }
      if (frame_ && frame_->sequence > consumed_sequence_) {
        ++overwritten_count_;
      }
      frame_ = std::move(frame);
    }
    changed_.notify_all();
  }

  [[nodiscard]] Result<std::shared_ptr<const Frame>>
  wait_for_newer(std::uint64_t after_sequence, std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    const auto ready = changed_.wait_for(lock, timeout, [this, after_sequence] {
      return closed_ || (frame_ && frame_->sequence > after_sequence);
    });
    if (!ready) {
      return Status{ErrorCode::timeout, "Timed out waiting for a fresh frame"};
    }
    if (frame_ && frame_->sequence > after_sequence) {
      consumed_sequence_ = frame_->sequence;
      return frame_;
    }
    return Status{ErrorCode::stopped, "Frame buffer is closed"};
  }

  void close() noexcept {
    {
      std::lock_guard lock(mutex_);
      closed_ = true;
    }
    changed_.notify_all();
  }

  [[nodiscard]] std::uint64_t overwritten_count() const noexcept {
    std::lock_guard lock(mutex_);
    return overwritten_count_;
  }

private:
  mutable std::mutex mutex_;
  std::condition_variable changed_;
  std::shared_ptr<const Frame> frame_;
  std::uint64_t consumed_sequence_{};
  std::uint64_t overwritten_count_{};
  bool closed_{};
};

}
