# Phase 0A.1 hardening review

## Requirement traceability

| Area | Implementation | Automated evidence | Remaining qualification |
|---|---|---|---|
| Geometry | Aspect-corrected projected 2D angle calculation | Square, 16:9, portrait, degenerate, and confidence tests | Human camera-viewpoint assessment |
| Synchronization | Exact inference-frame retention and synchronized default; mismatch suppression in low-latency mode | Tracking mismatch tests | Visual TITAN observation |
| Tracking validity | Detection and usable pose split; freshness, capture, source, landmark, and inference failure gates | Deterministic failure-injection tests | Physical disconnect and occlusion |
| Benchmarking | Sequential deterministic and paced real-time modes; hashes, IDs, timestamps, warm-up split, deadlines, bounded failures, validator | Four benchmark and report CTests | Sustained TITAN CUDA runs |
| Lifecycle | Single-use source contract, idempotent stop, exception translation, nonzero terminal exit | Restart, disconnection, exception, and shutdown tests | Media Foundation blocking-read behavior |
| Model correctness | Manifest verification, exact names/types/runtime dimensions/counts, RGB and SimCC fixtures | Math fixtures and real CPU RTMPose inference | Physical pose accuracy |
| Build and dependencies | Strict ONNX preset, CPU ONNX CI, staged idempotent acquisition, CPU/GPU isolation, C++ CUDA provider API | Strict local CPU/ONNX builds and script checks | Actual GPU package and CUDA load on TITAN |
| TITAN preparation | Repository-local machine-readable qualification script and transfer procedure | PowerShell parser validation | Entire TITAN procedure unexecuted |

## VENOM evidence

On 2026-11-27, the temporary VENOM machine completed:

- Strict `/W4 /WX` hardware-independent Debug compilation.
- 40 of 40 Debug CTests, including both benchmark modes, report validation, and unknown-provider rejection.
- Strict Release compilation with ONNX Runtime 1.20.1 CPU enabled.
- 40 of 40 ONNX-enabled Release CTests.
- A deterministic RTMPose CPU smoke run against the pinned MMPose sample video using verified detector and pose hashes.
- Three exact sequential source frames: one warm-up and two measured, zero skips, zero errors, person detected, and usable pose on all frames.
- Source-video SHA-256 `60fd3bc4a9f726c4704d3869471f89c61e992925727a09730047275066c63482`.
- Idempotent model and CPU ONNX Runtime acquisition checks.

This evidence establishes implementation correctness on the temporary CPU machine only. It is not NVIDIA, CUDA, TITAN performance, or physical pose evidence.

## Unresolved limitations

- TITAN hardware inventory and the repository qualification script have not been run on TITAN.
- `CUDAExecutionProvider`, CUDA 12, cuDNN 9, provider assignment, RTX performance, and strict GPU failure behavior remain unverified on the production machine.
- Live camera anatomy, projected knee-angle usefulness, occlusion recovery, physical disconnect, and sustained responsiveness require a human participant.
- OpenCV Media Foundation cannot be guaranteed to cancel a driver-blocked `read()`; shutdown waits for the native call to return.
- The pose model reports projected 2D landmarks and confidence-like scores, not calibrated probability or physical 3D joint geometry.
- The detector still selects the highest-confidence person. Identity tracking and gestures remain intentionally absent.

Phase 0B has not begun.
