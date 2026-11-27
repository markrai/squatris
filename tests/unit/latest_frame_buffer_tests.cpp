#include <chrono>
#include <memory>

#include <gtest/gtest.h>

#include "squatris/core/input/latest_frame_buffer.hpp"

namespace {

std::shared_ptr<squatris::input::Frame> frame(std::uint64_t sequence) {
  auto value = std::make_shared<squatris::input::Frame>();
  value->sequence = sequence;
  return value;
}

TEST(LatestFrameBuffer, KeepsOnlyTheNewestFrame) {
  squatris::input::LatestFrameBuffer buffer;
  buffer.publish(frame(1));
  buffer.publish(frame(2));
  const auto latest = buffer.wait_for_newer(0, std::chrono::milliseconds(1));
  ASSERT_TRUE(latest);
  EXPECT_EQ(latest.value()->sequence, 2);
  EXPECT_EQ(buffer.overwritten_count(), 1);
}

TEST(LatestFrameBuffer, DoesNotReturnStaleFrames) {
  squatris::input::LatestFrameBuffer buffer;
  buffer.publish(frame(4));
  const auto result = buffer.wait_for_newer(4, std::chrono::milliseconds(1));
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, squatris::ErrorCode::timeout);
}

TEST(LatestFrameBuffer, CloseUnblocksWaitersWithExplicitStatus) {
  squatris::input::LatestFrameBuffer buffer;
  buffer.close();
  const auto result = buffer.wait_for_newer(0, std::chrono::milliseconds(1));
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, squatris::ErrorCode::stopped);
}

}
