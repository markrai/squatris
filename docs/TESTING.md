# Testing

Normal tests require no camera, NVIDIA GPU, CUDA, or model file.

## Local CI

Run every GitHub Windows CI job locally before pushing:

```powershell
.\ci.bat
```

Or from Git Bash on Windows:

```bash
./ci.sh
```

Both launchers invoke the same PowerShell 7 script as GitHub Actions. The runner initializes an x64 MSVC environment if necessary, verifies the pinned vcpkg checkout, bootstraps the repository-local copy if missing, and acquires the pinned ONNX Runtime CPU package only for the ONNX job. Existing dependency caches are reused.

Select a single job for a shorter iteration:

```powershell
.\ci.bat -Job build-test
.\ci.bat -Job clang-tidy
.\ci.bat -Job onnx-cpu-build-test
```

Each job uses a separate `out/build/ci-windows-*` directory, fresh CMake configuration and clean project compilation. The runner stops at the first failure and returns a nonzero exit code. Both CTest jobs fail if no tests are discovered. CUDA builds, camera tests and interactive GUI smoke tests remain separate.

Prerequisites: Windows x64, PowerShell 7 (`pwsh`), Git, Visual Studio with x64 C++ tools and Windows SDK, CMake 3.28+, Ninja, and Clang-Tidy for the analysis job. Visual Studio-bundled tools are discovered when absent from PATH. A supplied `VCPKG_ROOT` must already match the repository baseline. No global software or GPU settings are changed. The Bash launcher supports Git Bash/MSYS2/Cygwin on Windows, not native Linux or WSL, because the current CI uses MSVC and Windows APIs. Local compiler and SDK versions can differ from GitHub's windows-2025 image; the checks and dependency pins are shared.

## Individual tests

```powershell
ctest --preset windows-debug
```

The suite covers:

- Canonical coordinate normalization and bounds.
- COCO/HALPE mapping and anatomical left/right identity.
- Explicit missing landmarks, invalid confidence, and non-finite values.
- Joint angles, low-confidence inputs, and degenerate geometry.
- Equivalent projected pixel geometry across square, 16:9, and portrait images.
- Timestamp ordering.
- Latest-frame replacement, stale-frame timeout, and close behavior.
- Synthetic acquisition, mock inference, disconnection propagation, and idempotent shutdown.
- Single-use source lifecycle and worker-exception translation.
- Person detection versus usable pose, frame mismatch, freshness, capture stall, source failure, and deterministic inference-failure invalidation.
- Exact in-order sequential acquisition with source indices and timestamps.
- Deterministic, real-time, duration, and loop-identity benchmark execution plus schema-4 report validation.
- Rejection of corrupted frame ordering, phase counts, provider/policy identities, profile hashes, node assignments, aggregate statistics, terminal errors, and success invariants.
- Valid machine-readable benchmark reports for initialization failures.
- Pinned RTMPose preprocessing and SimCC reference fixtures.

The CPU ONNX-enabled preset compiles the complete native inference backend and runs the same hardware-independent suite without model weights:

```powershell
.\scripts\acquire-onnxruntime.ps1 -Variant cpu
$env:SQUATRIS_ONNXRUNTIME_ROOT=(Resolve-Path '.\.deps\onnxruntime\cpu')
cmake --preset windows-onnx-release
cmake --build --preset windows-onnx-release
ctest --preset windows-onnx-release
```

Hardware checks are opt-in, separately labeled, and excluded from normal CI. Configure them only on the target machine after acquiring the models and recording a qualification video:

```powershell
cmake --preset windows-cuda-release -DSQUATRIS_ENABLE_HARDWARE_TESTS=ON -DSQUATRIS_HARDWARE_VIDEO="$PWD\captures\qualification.mp4"
cmake --build --preset windows-cuda-release
ctest --preset windows-hardware
```

Live-camera behavior remains a visual/manual check:

```powershell
.\out\build\windows-cuda-release\apps\pose-lab\SquatrisPoseLab.exe --list-cameras
.\out\build\windows-cuda-release\apps\pose-lab\SquatrisPoseLab.exe --camera 0 --backend rtmpose --provider cuda-audited --policy-manifest config\rtmpose-cuda-policy.json --provider-evidence-dir qualification\pose-lab
```

A passing mock or CPU test does not imply that cameras, CUDA, TITAN performance, or physical poses work. Qualification evidence belongs in validated benchmark reports and the machine-readable procedure in `QUALIFICATION.md`.
