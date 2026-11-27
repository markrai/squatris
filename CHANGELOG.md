# Changelog

## 0.1.0 - Unreleased

- Establish the Phase 0A native C++20 foundation.
- Add canonical pose, bounded latest-frame transport, synthetic and OpenCV frame sources.
- Add mock and RTMPose/YOLOX ONNX Runtime inference backends.
- Add Pose Lab diagnostics, acquisition and inference timing, prerecorded benchmarking, deterministic tests, and Windows CI.
- Add opt-in hardware qualification tests and explicit FFmpeg/Media Foundation OpenCV features.
- Harden projected 2D geometry for image aspect ratio and add cross-aspect regression coverage.
- Add synchronized Pose Lab rendering, explicit tracking validity, failure invalidation, and bounded inference failure handling.
- Split deterministic and real-time benchmarks, add source/frame evidence, report validation, deadlines, and strict provider rejection.
- Validate the pinned RTMPose tensor contract and model checksums before qualification execution.
- Isolate CPU/GPU runtime acquisition, add ONNX-enabled CI, and add the reproducible TITAN qualification workflow.
