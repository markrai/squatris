# Development environment inventory

Inventory date: 2026-11-27, America/New_York.

The inspected machine is a temporary implementation PC, not TITAN.

| Item | Observed |
|---|---|
| OS | Windows build 26100 x64; legacy API label `Windows 10 Pro` |
| Computer | Lenovo 20KES0UL00 |
| CPU | Intel Core i7-8650U, 4 cores / 8 logical processors |
| RAM | Approximately 16 GB |
| GPU | Intel UHD Graphics 620, driver 31.0.101.2140 |
| NVIDIA / CUDA | No NVIDIA adapter, `nvidia-smi`, CUDA, cuDNN, or `nvcc` found |
| Visual Studio | Community 2026 18.0.2 |
| MSVC | 14.50.35717 and 14.44.35207 installed |
| Windows SDK | 10.0.26100.0 and 10.0.22621.0 |
| CMake | Visual Studio bundled 4.1.1-msvc1 |
| Ninja | Visual Studio bundled 1.12.1 |
| clang-format / clang-tidy | Visual Studio bundled x64 LLVM tools |
| Git | 2.51.0.windows.1 |
| Python | 3.13.7; not used as a production runtime |
| Package managers | WinGet 1.29.380; no global vcpkg, Chocolatey, or Scoop |
| Godot | Not installed; not required in Phase 0A |
| ONNX Runtime | No preinstalled native or Python runtime found |
| Camera devices | Integrated Camera present; Logitech VID_046D/PID_0990 reported with unknown device status |
| Camera formats | Media Foundation probe delivered camera 0 at 1280x720/30 FPS and 640x480/30 FPS in BGR8 buffers |

TITAN's Ryzen 9 7900X, RTX 4080 SUPER, NVIDIA driver, CUDA/cuDNN, and physical camera still require inventory and qualification. No global configuration was changed during this inspection.

## Temporary-PC smoke evidence

The official MMPose PoseTrack sample opened through FFmpeg at 1280x720/30 FPS. The pinned YOLOX and RTMPose models completed one warm-up and three measured end-to-end iterations through ONNX Runtime `CPUExecutionProvider`; all measured frames contained a pose and model hashes matched the manifest. Median estimator latency was 320.089 ms and measured throughput was 3.163 FPS. Three samples are smoke evidence only, not a performance qualification.

An explicit CUDA request against the CPU runtime failed during initialization with `Requested ONNX Runtime provider is unavailable: CUDAExecutionProvider` and exit code 1. No CPU fallback occurred.
