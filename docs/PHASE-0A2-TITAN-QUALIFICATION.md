# Phase 0A.2 TITAN hardware qualification

## Status

`PHASE_0A2_PARTIAL`

Automated hardware-independent qualification and physical camera capability probing passed on TITAN. Strict CUDA RTMPose initialization is blocked before inference because the pinned detector graph contains nodes that ONNX Runtime 1.20.1 assigns to its default CPU execution provider while CPU fallback is explicitly disabled. No GPU inference, performance, sustained-run, live Pose Lab, or human-participation claim is made.

Qualification was performed on 2026-11-27. All repository changes remain uncommitted.

## Actual machine and dependency inventory

| Component | Observed value |
|---|---|
| Machine identity | `TITAN`; Micro-Star International `US Desktop Aegis ZS2` |
| OS | Windows 11 Home x64, version `10.0.26300`, build `26300` |
| CPU | AMD Ryzen 9 7900X, 12 cores, 24 logical processors, reported maximum 4701 MHz |
| Physical memory | 33,328,259,072 bytes, approximately 31.0 GiB |
| Discrete GPU | NVIDIA GeForce RTX 4080 SUPER, compute capability 8.9, 16,376 MiB VRAM |
| NVIDIA driver | `610.88`; Windows driver version `32.0.16.1088` |
| Driver CUDA compatibility | CUDA UMD 13.3 |
| CUDA toolkit | CUDA 12.8, `nvcc` 12.8.61, installed at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8` |
| cuDNN | Repository-local cuDNN 9.1.0.70 for CUDA 12; NVIDIA archive SHA-256 `3083b5bf20b0b2b29305de088e0b2cdbb78b5190655a27e8a7c596e47042eedf` |
| ONNX Runtime | Repository-local Microsoft GPU package 1.20.1, NuGet SHA-512 verified |
| Visual Studio | Visual Studio Community 2022 17.14.23 |
| MSVC | Toolset 14.44.35207; compiler 19.44.35222 |
| Windows SDKs | 10.0.22621.0 and 10.0.26100.0 |
| CMake | 4.2.1 |
| Ninja | 1.12.1 |
| vcpkg baseline | `9fa95bf687fa6ae7f2da0673a8067e5fc1515c27` |
| OpenCV | 4.14.0 with Media Foundation and FFmpeg |

ONNX Runtime's official compatibility matrix specifies CUDA 12.x and cuDNN 9.x for 1.20.x. The pinned CUDA provider DLL imports `cublasLt64_12.dll`, `cublas64_12.dll`, `cufft64_11.dll`, `cudart64_12.dll`, and `cudnn64_9.dll`. All are available through process-local paths during qualification. No driver, global PATH, global CUDA installation, or unrelated repository was modified.

The GPU compatibility source is the [ONNX Runtime CUDA execution-provider matrix](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html). The repository-local cuDNN archive and checksum come from NVIDIA's [Windows cuDNN redistributable](https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/) and its 9.1.0 manifest.

## Corrections completed

- Benchmark `usable_pose` classification now comes from `assess_tracking`, using the same freshness, failure, confidence, and landmark policy as live diagnostics.
- Pose availability, canonical usable-pose state, and full-body completeness are separate report fields.
- Shoulders, elbows, wrists, hips, knees, and ankles report left, right, and paired coverage independently.
- Tracking rejects missing, inverted, and future capture, inference-completion, and capture-observation timestamps.
- Benchmark schema 3 validates frame and source ordering, phase counts, per-frame and aggregate timing consistency, percentile sample counts, hashes, skipped-frame evidence, terminal errors, provider evidence, and success/failure invariants.
- A benchmark writes a valid pending failure report before model/session initialization and replaces it atomically with final evidence. Initialization failures therefore remain machine-readable. The external watchdog preserves that report if a native call blocks.
- Qualification invokes native programs in child processes with explicit external deadlines and process-tree termination. It does not terminate worker threads.
- Qualification records the actual CIM machine identity and compares it with the expected `TITAN` name instead of writing an assumed name.
- CUDA runs enable ONNX Runtime profiling for detector and pose sessions. A successful CUDA report requires CUDA node events from both profiles, profile hashes, and no CPU or unknown-provider node events.
- CPU detector preprocessing, detector postprocessing, pose preprocessing, pose postprocessing, and synchronous ONNX Runtime execution are separate timings. Reports explicitly state that CUDA graph execution is disabled.
- Detector preprocessing, inference, and postprocessing are independently timed in addition to the complete detector and estimator durations.
- Failed benchmark reports and six report-corruption classes are covered by automated tests.
- ONNX Runtime GPU acquisition follows the package's actual `buildTransitive/native/include` layout.
- cuDNN 9.1.0.70 can be acquired into `.deps` with an official SHA-256; only the qualification process PATH is changed.

## Build, tests, and model integrity

- Strict Windows CUDA Release configure: passed.
- Strict `/W4 /WX` Windows CUDA Release build: passed.
- Hardware-independent tests in the CUDA-enabled build: 44 of 44 passed.
- Deterministic and paced mock benchmark generation and schema validation: passed.
- Corrupted ordering, phase-count, timing-count, model-hash, success-invariant, and terminal-error reports: all rejected.
- Early initialization failure report generation and subsequent validation: passed.
- Detector SHA-256: `ceb11c07298f95c50d7c5abeb906d03340c85f23aa79e3e66966e7fb6c307250`.
- Pose SHA-256: `26f3a19e61304a600dfb82d1001d41d24343b89fc70a33ffc84657e0b0bf2ecf`.

The smoke input is the official MMPose `demo/resources/demo.mp4`, SHA-256 `dc788b413f384887dab9502197657c05393efd1d26825831e4da6fa75b4908c9`. It is suitable only for model-loading and inference smoke work. It is not full-body gameplay qualification footage.

## CUDA execution evidence and blocker

CUDA and cuDNN DLL loading succeeds, and ONNX Runtime begins creating the CUDA-aware detector session. Graph partitioning then prevents session initialization with:

```text
detector session initialization failed: This session contains graph nodes that are assigned to the default CPU EP, but fallback to CPU EP has been explicitly disabled by the user.
```

The failure report records:

- requested provider `CUDAExecutionProvider`;
- both verified model hashes;
- CPU fallback still disabled;
- `strict_provider_assignment: false`;
- no provider profiles accepted;
- no inference samples;
- detector-session initialization duration of approximately 1.31 seconds for this failed attempt;
- a typed `initialization_failed` terminal error at `estimator-initialization`.

This is not evidence of successful GPU execution. Enabling CPU fallback would allow a mixed-provider graph but would violate the current architecture and acceptance invariant, so it was not done silently or otherwise.

## Benchmark methodology and performance result

The implemented qualification path uses the same prerecorded input for deterministic sequential and paced real-time runs. It requires three repeated runs of each mode and a paced run of at least 300 seconds. Warm-up is separate from measurement. Reports include initialization and first-inference time, detector and pose timing stages, capture-delivery-to-completion latency, median, p95, p99, canonical useful-pose throughput, landmark coverage, skipped frames, and terminal errors. The external driver samples process CPU, working set, private memory, total GPU utilization, and total GPU memory use.

No GPU performance distribution is reported because strict session initialization failed before the first frame. Repeated steady-state and five-minute runs were correctly not started. The resource file contains only the short failed initialization attempt and ambient system GPU memory, so it must not be interpreted as RTMPose utilization or process VRAM.

The bottleneck is graph execution-provider compatibility, not measured RTX 4080 SUPER throughput. The 60 useful-pose-updates/s and future 80 ms response target remain unassessed.

## Camera capability evidence

Windows reports one active Logitech BRIO and one disconnected USB video device. OpenCV Media Foundation camera index 0 successfully delivered a sample at each probed mode:

- 1920x1080 at negotiated 60 FPS.
- 1920x1080 at negotiated 30 FPS.
- 1280x720 at negotiated 60 FPS.
- 1280x720 at negotiated 30 FPS.
- 640x480 at negotiated 60 FPS.
- 640x480 at negotiated 30 FPS.

The probe proves that frames were delivered at each requested/negotiated mode. It does not measure sustained delivered FPS. Live Pose Lab could not pass inference initialization, so responsive preview, frame synchronization, freshness, low-confidence behavior, frame skips, delivered FPS, resource use, and shutdown under CUDA remain unverified.

## Machine-readable evidence

- `qualification/phase0a2/automated/qualification-report.json`
- `qualification/phase0a2/automated/inventory.json`
- `qualification/phase0a2/automated/automated-tests.log`
- `qualification/phase0a2/automated/camera-enumeration.log`
- `qualification/phase0a2/automated/cuda-provider-smoke.json`
- `qualification/phase0a2/automated/cuda-provider-session.log`
- `qualification/phase0a2/automated/cuda-provider-session.resources.json`

Qualification output, dependencies, models, and build products are ignored by Git and remain local to TITAN.

## Required physical participation

Do not begin these observations until strict CUDA initialization and actual inference succeed. Run both the 1920x1080 60 FPS and 1280x720 60 FPS commands stored in `qualification-report.json`, then record participant, date/time, device, mode, delivered FPS, result, and notes for every item.

1. Stand naturally in a full-body view and verify stable shoulder, elbow, wrist, hip, knee, and ankle landmarks.
2. Perform several slow comfortable squats and record projected landmark continuity and confidence.
3. Perform several fast comfortable squats and record lag, skips, freshness, and recovery.
4. Move laterally left and right while remaining in view.
5. Swipe the left arm and right arm independently.
6. Combine lateral movement with left and right arm swipes.
7. Temporarily occlude each wrist and verify low-confidence or missing behavior and recovery.
8. Temporarily occlude each leg and verify hip, knee, and ankle behavior and recovery.
9. Leave the camera view and reenter it; verify fail-closed loss and clean reacquisition.
10. Have another person enter the camera view and record highest-confidence-person selection behavior.
11. Disconnect the camera and verify explicit failure, bounded shutdown, and no stale skeleton.
12. Keep the application running and interactive long enough to assess sustained preview responsiveness and latency accumulation.
13. Toggle frame-synchronized and low-latency preview; verify mismatched skeletons are suppressed.
14. Record delivered capture FPS rather than relying on requested or negotiated FPS.

These are landmark and tracking observations only. No gesture recognition or game action exists in this phase.

## Recommended next steps

1. Use ONNX Runtime verbose graph-partition diagnostics to identify the exact detector nodes assigned to CPU without enabling fallback in a qualification run.
2. Decide explicitly whether the architecture continues to require exclusive CUDA node assignment. If audited CPU shape/control nodes are acceptable, change the policy and report format only with approval; otherwise re-export or transform the same detector graph to eliminate those assignments and update its authoritative hash with approval.
3. Rerun the complete qualifier with a licensed, at least five-minute, full-body recording after strict provider initialization is resolved.
4. Review repeated-run variability and the five-minute resource trace before any optimization or model change.
5. Launch Pose Lab at 1080p60 and 720p60 and complete the human checklist.
6. Do not begin Phase 0B until CUDA inference and the physical observations are recorded.

Suggested commit message:

```text
fix: harden TITAN CUDA qualification evidence
```
