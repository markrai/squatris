#include "squatris/core/crypto/sha256.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

#ifdef _WIN32
#include <windows.h>

#include <bcrypt.h>
#endif

namespace squatris::crypto {

Result<std::string> sha256_file(const std::filesystem::path& path) {
#ifdef _WIN32
  BCRYPT_ALG_HANDLE algorithm{};
  BCRYPT_HASH_HANDLE hash{};
  DWORD object_size{};
  DWORD result_size{};
  DWORD hash_size{};
  const auto close = [&] {
    if (hash != nullptr) {
      BCryptDestroyHash(hash);
    }
    if (algorithm != nullptr) {
      BCryptCloseAlgorithmProvider(algorithm, 0);
    }
  };
  if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0 ||
      BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&object_size),
                        sizeof(object_size), &result_size, 0) < 0 ||
      BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH, reinterpret_cast<PUCHAR>(&hash_size),
                        sizeof(hash_size), &result_size, 0) < 0) {
    close();
    return Status{ErrorCode::io_error, "Windows SHA-256 provider initialization failed"};
  }
  std::vector<std::uint8_t> object(object_size);
  std::vector<std::uint8_t> digest(hash_size);
  if (BCryptCreateHash(algorithm, &hash, object.data(), object_size, nullptr, 0, 0) < 0) {
    close();
    return Status{ErrorCode::io_error, "Windows SHA-256 hash creation failed"};
  }
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    close();
    return Status{ErrorCode::not_found, "File could not be opened for SHA-256: " + path.string()};
  }
  std::error_code size_error;
  const auto expected_size = std::filesystem::file_size(path, size_error);
  if (size_error) {
    close();
    return Status{ErrorCode::io_error,
                  "File size could not be read for SHA-256: " + path.string()};
  }
  std::array<char, 1 << 16> buffer{};
  std::uintmax_t total_bytes{};
  while (stream) {
    stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto bytes = stream.gcount();
    if (bytes > 0 && BCryptHashData(hash, reinterpret_cast<PUCHAR>(buffer.data()),
                                    static_cast<ULONG>(bytes), 0) < 0) {
      close();
      return Status{ErrorCode::io_error, "Windows SHA-256 update failed"};
    }
    total_bytes += static_cast<std::uintmax_t>(bytes);
  }
  const auto final_size = std::filesystem::file_size(path, size_error);
  if (stream.bad() || !stream.eof() || size_error || total_bytes != expected_size ||
      final_size != expected_size) {
    close();
    return Status{ErrorCode::io_error,
                  "File read was incomplete or changed during SHA-256: " + path.string()};
  }
  if (BCryptFinishHash(hash, digest.data(), hash_size, 0) < 0) {
    close();
    return Status{ErrorCode::io_error, "Windows SHA-256 finalization failed"};
  }
  close();
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const auto byte : digest) {
    output << std::setw(2) << static_cast<unsigned>(byte);
  }
  return output.str();
#else
  static_cast<void>(path);
  return Status{ErrorCode::unavailable, "SHA-256 is implemented for Windows in Phase 0A"};
#endif
}

}
