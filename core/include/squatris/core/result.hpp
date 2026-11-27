#pragma once

#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace squatris {

enum class ErrorCode {
  none,
  invalid_argument,
  invalid_data,
  not_found,
  unavailable,
  timeout,
  end_of_stream,
  disconnected,
  initialization_failed,
  inference_failed,
  io_error,
  stopped
};

struct Status {
  ErrorCode code{ErrorCode::none};
  std::string message;

  [[nodiscard]] bool ok() const noexcept { return code == ErrorCode::none; }
  [[nodiscard]] static Status success() { return {}; }
};

template <typename T> class Result {
public:
  Result(T value) : value_(std::move(value)) {}
  Result(Status error) : value_(std::move(error)) {}

  [[nodiscard]] bool has_value() const noexcept { return std::holds_alternative<T>(value_); }
  [[nodiscard]] explicit operator bool() const noexcept { return has_value(); }
  [[nodiscard]] T& value() { return std::get<T>(value_); }
  [[nodiscard]] const T& value() const { return std::get<T>(value_); }
  [[nodiscard]] T&& take_value() { return std::move(std::get<T>(value_)); }
  [[nodiscard]] const Status& error() const { return std::get<Status>(value_); }

private:
  std::variant<T, Status> value_;
};

}
