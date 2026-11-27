#include <chrono>

#include <gtest/gtest.h>

#include "squatris/backends/camera/sequential_frame_source.hpp"
#include "squatris/backends/camera/synthetic_source.hpp"
#include "squatris/backends/inference/mock_pose_estimator.hpp"
#include "squatris/core/pose/pose.hpp"

namespace {

TEST(SyntheticPipeline, ProducesValidPoseWithoutHardware) {
  squatris::backends::camera::SyntheticSource source({320, 240, 120.0});
  ASSERT_TRUE(source.start().ok());
  const auto frame = source.wait_for_frame(0, std::chrono::seconds(1));
  ASSERT_TRUE(frame);
  squatris::backends::inference::MockPoseEstimator estimator;
  const auto result = estimator.infer(*frame.value());
  ASSERT_TRUE(result);
  ASSERT_TRUE(result.value().pose.has_value());
  const auto pose = result.value().pose.value_or(squatris::pose::PoseFrame{});
  EXPECT_TRUE(squatris::pose::validate_pose(pose).ok());
  EXPECT_EQ(pose.sequence, frame.value()->sequence);
  EXPECT_EQ(result.value().timings.frame_acquisition, frame.value()->acquisition_duration);
  source.stop();
}

TEST(SyntheticPipeline, CleanShutdownIsIdempotent) {
  squatris::backends::camera::SyntheticSource source({64, 48, 60.0});
  ASSERT_TRUE(source.start().ok());
  ASSERT_TRUE(source.wait_for_frame(0, std::chrono::seconds(1)));
  source.stop();
  source.stop();
  const auto restart = source.start();
  EXPECT_FALSE(restart.ok());
  EXPECT_EQ(restart.code, squatris::ErrorCode::initialization_failed);
  const auto stopped = source.wait_for_frame(100, std::chrono::milliseconds(1));
  ASSERT_FALSE(stopped);
  EXPECT_EQ(stopped.error().code, squatris::ErrorCode::stopped);
}

TEST(SyntheticPipeline, PropagatesDisconnection) {
  squatris::backends::camera::SyntheticSourceConfig config{64, 48, 120.0, 1};
  squatris::backends::camera::SyntheticSource source(config);
  ASSERT_TRUE(source.start().ok());
  const auto first = source.wait_for_frame(0, std::chrono::seconds(1));
  ASSERT_TRUE(first);
  const auto ended = source.wait_for_frame(first.value()->sequence, std::chrono::seconds(1));
  ASSERT_FALSE(ended);
  const auto error = source.terminal_error();
  ASSERT_TRUE(error.has_value());
  EXPECT_EQ(error.value_or(squatris::Status{}).code, squatris::ErrorCode::disconnected);
  source.stop();
}

TEST(SyntheticPipeline, TranslatesWorkerExceptionIntoTerminalError) {
  squatris::backends::camera::SyntheticSourceConfig config{64, 48, 120.0, std::nullopt, 1};
  squatris::backends::camera::SyntheticSource source(config);
  ASSERT_TRUE(source.start().ok());
  const auto first = source.wait_for_frame(0, std::chrono::seconds(1));
  ASSERT_TRUE(first);
  const auto ended = source.wait_for_frame(first.value()->sequence, std::chrono::seconds(1));
  ASSERT_FALSE(ended);
  const auto error = source.terminal_error();
  ASSERT_TRUE(error.has_value());
  EXPECT_EQ(error.value_or(squatris::Status{}).code, squatris::ErrorCode::io_error);
  source.stop();
}

TEST(SequentialPipeline, DeliversEveryFrameInExactOrderWithoutSkipping) {
  squatris::backends::camera::SequentialFrameSource source({{}, 32, 24, 50.0});
  ASSERT_TRUE(source.start().ok());
  std::uint64_t sequence{};
  for (std::uint64_t expected_index = 0; expected_index < 5; ++expected_index) {
    const auto frame = source.wait_for_frame(sequence, std::chrono::seconds(1));
    ASSERT_TRUE(frame);
    sequence = frame.value()->sequence;
    ASSERT_TRUE(frame.value()->source_frame_index.has_value());
    EXPECT_EQ(*frame.value()->source_frame_index, expected_index);
    ASSERT_TRUE(frame.value()->source_media_timestamp.has_value());
    EXPECT_EQ(*frame.value()->source_media_timestamp,
              std::chrono::milliseconds(expected_index * 20));
  }
  EXPECT_EQ(source.statistics().frames_delivered, 5);
  EXPECT_EQ(source.statistics().frames_overwritten, 0);
  source.stop();
  EXPECT_FALSE(source.start().ok());
}

}
