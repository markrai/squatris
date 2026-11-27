# Squatris

Squatris is a Windows-first motion-controlled falling-block game. This repository currently contains Phase 0A: a game-independent native pose foundation and the Squatris Pose Lab diagnostic application. It does not contain gameplay, gesture recognition, Godot integration, networking, or cloud inference.

The production path is C++20, OpenCV camera capture, RTMPose through ONNX Runtime, and an explicit CUDA execution provider. CPU inference, synthetic capture, prerecorded capture, and a deterministic mock estimator are available for development and hardware-independent testing.

## What is implemented

- Canonical backend-independent pose frames with explicit missing data and optional estimated depth.
- COCO-17 and HALPE-26 anatomical mapping with left/right identity preservation.
- OpenCV Media Foundation, prerecorded, and synthetic frame sources.
- A bounded latest-frame handoff that drops stale work instead of building a backlog.
- Strict ONNX Runtime CPU or CUDA provider selection.
- YOLOX-tiny person detection and RTMPose-m HALPE-26 inference.
- Pose Lab camera/skeleton/diagnostic visualization and keyboard controls.
- Reproducible JSON benchmarking with warm-up separation and latency percentiles.
- Deterministic GoogleTest coverage and Windows CI.

## Quick start on Windows

Run these commands from an x64 Native Tools PowerShell for Visual Studio:

```powershell
.\scripts\bootstrap-vcpkg.ps1
$env:VCPKG_ROOT=(Resolve-Path '.\.deps\vcpkg')
cmake --preset windows-debug
cmake --build --preset windows-debug
ctest --preset windows-debug
```

Run the hardware-independent Pose Lab:

```powershell
.\out\build\windows-debug\apps\pose-lab\SquatrisPoseLab.exe --synthetic --backend mock --width 1280 --height 720 --fps 60
```

Acquire verified model files and configure CPU or CUDA inference:

```powershell
.\scripts\acquire-models.ps1
.\scripts\acquire-onnxruntime.ps1 -Variant gpu
$env:SQUATRIS_ONNXRUNTIME_ROOT=(Resolve-Path '.\.deps\onnxruntime')
cmake --preset windows-cuda-release
cmake --build --preset windows-cuda-release
.\out\build\windows-cuda-release\apps\pose-lab\SquatrisPoseLab.exe --camera 0 --backend rtmpose --provider cuda
```

CUDA mode fails explicitly if `CUDAExecutionProvider` cannot be registered. Select `--provider cpu` deliberately when CPU behavior is desired.

See [BUILDING.md](docs/BUILDING.md), [ARCHITECTURE.md](docs/ARCHITECTURE.md), and [QUALIFICATION.md](docs/QUALIFICATION.md) before running on qualification hardware.

## Pose Lab controls

- `Q` or `Esc`: exit.
- `S`: toggle skeleton and landmark confidence overlay.
- `D`: toggle diagnostics.
- `M`: mirror only the preview presentation.

Mirroring never changes canonical anatomical left/right identity.

## Repository scope

Model weights, captured video, benchmark results, local dependencies, generated files, and build output are ignored by Git. No project license has been selected.

