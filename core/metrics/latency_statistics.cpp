#include "squatris/core/metrics/latency_statistics.hpp"

#include <algorithm>
#include <cmath>

namespace squatris::metrics {

LatencyStatistics::LatencyStatistics(std::size_t capacity) : capacity_(capacity) {
  samples_.reserve(capacity);
}

void LatencyStatistics::add(std::chrono::microseconds sample) {
  if (capacity_ == 0) {
    return;
  }
  if (samples_.size() < capacity_) {
    samples_.push_back(sample);
    return;
  }
  samples_[next_] = sample;
  next_ = (next_ + 1) % capacity_;
}

Percentiles LatencyStatistics::calculate() const {
  if (samples_.empty()) {
    return {};
  }
  auto ordered = samples_;
  std::sort(ordered.begin(), ordered.end());
  const auto percentile = [&ordered](double probability) {
    const auto position = probability * static_cast<double>(ordered.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const auto fraction = position - static_cast<double>(lower);
    const auto value = static_cast<double>(ordered[lower].count()) * (1.0 - fraction) +
                       static_cast<double>(ordered[upper].count()) * fraction;
    return value / 1000.0;
  };
  return {percentile(0.5), percentile(0.95), percentile(0.99), ordered.size()};
}

const std::vector<std::chrono::microseconds>& LatencyStatistics::samples() const noexcept {
  return samples_;
}

}
