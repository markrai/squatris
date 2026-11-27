# Building

## Supported Phase 0A.1 host

The intended host is Windows 11 x64 with MSVC and the Windows SDK. CMake 3.28 or newer and Ninja are required. Visual Studio's bundled CMake and Ninja are sufficient when their directories are added to `PATH`, or when commands are invoked by absolute path. Run configure and build commands inside an x64 Visual Studio developer shell so MSVC's standard-library and SDK paths are available.

The project does not install drivers, CUDA, cuDNN, or global packages.

## CPU build and tests

Open an x64 Native Tools PowerShell for Visual Studio, then run:

```powershell
.\scripts\bootstrap-vcpkg.ps1
$env:VCPKG_ROOT=(Resolve-Path '.\.deps\vcpkg')
cmake --preset windows-debug
cmake --build --preset windows-debug
ctest --preset windows-debug
```

The release build is:

```powershell
cmake --preset windows-release
cmake --build --preset windows-release
ctest --preset windows-release
```

## ONNX Runtime CPU build

```powershell
.\scripts\acquire-onnxruntime.ps1 -Variant cpu
$env:SQUATRIS_ONNXRUNTIME_ROOT=(Resolve-Path '.\.deps\onnxruntime\cpu')
cmake --preset windows-onnx-release
cmake --build --preset windows-onnx-release
ctest --preset windows-onnx-release
```

Use `--provider cpu` at runtime.

CPU and GPU packages install into isolated variant directories and cannot overwrite one another.

## ONNX Runtime CUDA build

ONNX Runtime 1.20.1 GPU is pinned because its official compatibility matrix supports CUDA 12.x with cuDNN 9.x. Verify TITAN's installed CUDA and cuDNN before acquiring the runtime. Do not substitute cuDNN 8 or assume a newer ONNX Runtime package has the same CUDA requirement.

```powershell
.\scripts\acquire-onnxruntime.ps1 -Variant gpu
$env:SQUATRIS_ONNXRUNTIME_ROOT=(Resolve-Path '.\.deps\onnxruntime\gpu')
cmake --preset windows-cuda-release
cmake --build --preset windows-cuda-release
```

The GPU NuGet package is pinned by SHA-512. All native DLLs are copied beside built executables. CUDA and cuDNN DLL discovery still follows ONNX Runtime's documented Windows runtime rules.

## Models and Pose Lab

```powershell
.\scripts\acquire-models.ps1
.\out\build\windows-cuda-release\apps\pose-lab\SquatrisPoseLab.exe --list-cameras
.\out\build\windows-cuda-release\apps\pose-lab\SquatrisPoseLab.exe --camera 0 --width 1920 --height 1080 --fps 60 --backend rtmpose --provider cuda --detector models\detector.onnx --pose models\pose.onnx
```

Camera requests are preferences. Pose Lab reports the actual width, height, backend, encoding, configured capture rate, and measured delivery rate. Frame-synchronized diagnostics are the default. `--freshness-ms` controls pose and capture freshness, while `--max-inference-failures` bounds persistent failures.

## Static analysis and formatting

```powershell
clang-format -i (rg --files -g '*.cpp' -g '*.hpp')
cmake --preset windows-release -DCMAKE_CXX_CLANG_TIDY=clang-tidy -DSQUATRIS_WARNINGS_AS_ERRORS=ON
cmake --build --preset windows-release
```

The checked-in Clang-Tidy profile treats analyzer, bug-prone, concurrency, performance, and portability findings in project code as errors. Style-only transformations are handled by Clang-Format instead.

CPU ONNX-enabled compilation and hardware-independent tests run in CI without model files. CUDA and camera checks remain opt-in and are driven by `scripts/qualify-titan.ps1` on the target machine.
