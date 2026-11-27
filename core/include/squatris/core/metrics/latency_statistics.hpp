#pragma once

#include <chrono>
#include <cstddef>
#include <vector>

namespace squatris::metrics {

struct Percentiles {
  double median_ms{};
  double p95_ms{};
  double p99_ms{};
  std::size_t samples{};
};

class LatencyStatistics {
public:
  explicit LatencyStatistics(std::size_t capacity = 4096);
  void add(std::chrono::microseconds sample);
  [[nodiscard]] Percentiles calculate() const;
  [[nodiscard]] const std::vector<std::chrono::microseconds>& samples() const noexcept;

private:
  std::size_t capacity_;
  std::vector<std::chrono::microseconds> samples_;
  std::size_t next_{};
};

}
