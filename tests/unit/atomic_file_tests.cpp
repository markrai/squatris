#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>

#include <gtest/gtest.h>

#include "squatris/core/io/atomic_file.hpp"

namespace {

class AtomicFileTest : public testing::Test {
protected:
  void SetUp() override {
    directory_ = std::filesystem::temp_directory_path() /
                 ("squatris-atomic-" + std::to_string(
                                          std::chrono::steady_clock::now().time_since_epoch().count()));
    ASSERT_TRUE(std::filesystem::create_directory(directory_));
  }

  void TearDown() override {
    if (!directory_.empty()) {
      std::error_code error;
      std::filesystem::remove_all(directory_, error);
    }
  }

  std::filesystem::path directory_;
};

TEST_F(AtomicFileTest, ReplacesCompleteContentsWithoutTemporaryFiles) {
  const auto path = directory_ / "report.json";
  ASSERT_TRUE(squatris::io::write_text_atomically(path, "original").ok());
  ASSERT_TRUE(squatris::io::write_text_atomically(path, "new").ok());
  std::ifstream stream(path, std::ios::binary);
  ASSERT_TRUE(stream.good());
  const std::string content{std::istreambuf_iterator<char>(stream),
                            std::istreambuf_iterator<char>()};
  EXPECT_EQ(content, "new");
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(directory_),
                          std::filesystem::directory_iterator()), 1);
}

TEST_F(AtomicFileTest, FailedReplacementDoesNotRemoveDestinationOrLeakTemporaryFiles) {
  const auto path = directory_ / "existing-directory";
  ASSERT_TRUE(std::filesystem::create_directory(path));
  EXPECT_FALSE(squatris::io::write_text_atomically(path, "report").ok());
  EXPECT_TRUE(std::filesystem::is_directory(path));
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(directory_),
                          std::filesystem::directory_iterator()), 1);
}

TEST_F(AtomicFileTest, MissingParentFailsWithoutCreatingDestination) {
  const auto path = directory_ / "missing" / "report.json";
  EXPECT_FALSE(squatris::io::write_text_atomically(path, "report").ok());
  EXPECT_FALSE(std::filesystem::exists(path));
}

}
