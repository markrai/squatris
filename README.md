# Squatris

Squatris is a Windows-first motion-controlled falling-block game. This repository currently contains the Phase 0A.3 audited-CUDA, game-independent native pose foundation and the Squatris Pose Lab diagnostic application. Automated CUDA qualification passes on TITAN; representative footage and human camera observations remain incomplete as recorded in [PHASE-0A3-CUDA-EXECUTION.md](docs/PHASE-0A3-CUDA-EXECUTION.md). The repository does not contain gameplay, gesture recognition, player identity tracking, Godot integration, networking, or cloud inference.

The production path is C++20, OpenCV camera capture, RTMPose through ONNX Runtime, and an explicit CUDA execution provider. CPU inference, synthetic capture, prerecorded capture, and a deterministic mock estimator are available for development and hardware-independent testing.

## What is implemented

- Canonical backend-independent pose frames with explicit missing data and optional estimated depth.
- COCO-17 and HALPE-26 anatomical mapping with left/right identity preservation.
- OpenCV Media Foundation, prerecorded, and synthetic frame sources.
- A bounded latest-frame handoff that drops stale work instead of building a backlog.
- Frame-synchronized diagnostics, explicit pose freshness, and fail-closed tracking validity.
- Explicit CPU, strict CUDA, and hash-bound audited CUDA execution policies.
- YOLOX-tiny person detection and RTMPose-m HALPE-26 inference.
- Pose Lab camera/skeleton/diagnostic visualization and keyboard controls.
- Deterministic sequential and paced real-time JSON benchmarks with source/frame evidence, deadlines, report validation, warm-up separation, and latency percentiles.
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
.\scripts\acquire-cudnn.ps1
$env:SQUATRIS_ONNXRUNTIME_ROOT=(Resolve-Path '.\.deps\onnxruntime\gpu')
$env:PATH="$(Resolve-Path '.\.deps\cudnn\9.1.0.70\bin');$env:CUDA_PATH\bin;$env:PATH"
cmake --preset windows-cuda-release
cmake --build --preset windows-cuda-release
.\out\build\windows-cuda-release\apps\pose-lab\SquatrisPoseLab.exe --camera 0 --backend rtmpose --provider cuda-audited --policy-manifest config\rtmpose-cuda-policy.json --provider-evidence-dir qualification\pose-lab
```

CUDA mode fails explicitly if `CUDAExecutionProvider` cannot be registered. `cuda-strict` disables all CPU fallback; `cuda-audited` permits only the exact shape/index assignments in the pinned policy. Select `--provider cpu` deliberately when CPU behavior is desired.

Use `.deps\onnxruntime\gpu` for the GPU package or `.deps\onnxruntime\cpu` for the CPU package. See [BUILDING.md](docs/BUILDING.md), [ARCHITECTURE.md](docs/ARCHITECTURE.md), and [QUALIFICATION.md](docs/QUALIFICATION.md) before running on qualification hardware.

## Pose Lab controls

- `Q` or `Esc`: exit.
- `S`: toggle skeleton and landmark confidence overlay.
- `D`: toggle diagnostics.
- `M`: mirror only the preview presentation.
- `F`: switch between frame-synchronized diagnostics and low-latency preview.

Mirroring never changes canonical anatomical left/right identity. Low-latency mode suppresses skeletons when preview and pose sequences differ.

## Repository scope

Model weights, captured video, benchmark results, local dependencies, generated files, and build output are ignored by Git. No project license has been selected.
