# Architecture

## Dependency direction

`squatris_core` owns canonical data, validation, geometry, timing, and boundary interfaces. It has no OpenCV, ONNX Runtime, CUDA, or Godot dependency.

`squatris_backends` implements those interfaces. Camera adapters translate OpenCV images into owned BGR8 frames. Inference adapters translate model-specific tensors into canonical pose frames. Model details never escape this layer.

Pose Lab and benchmark executables depend on the interfaces and select concrete adapters at composition time. Godot remains absent until Phase 4A.

```text
Pose Lab / benchmark
        |
        v
core interfaces and canonical pose
        ^
        |
OpenCV capture / mock inference / RTMPose ONNX Runtime
```

## Canonical coordinates and anatomy

Image coordinates are normalized to `[0, 1]`, with `(0, 0)` at the top-left of the unmirrored source image, positive x to image-right, and positive y downward. A point outside the image or a non-finite point is unavailable rather than clamped into plausibility.

Geometry converts normalized coordinates back into aspect-corrected image space before measuring. Reported joint angles are projected 2D image angles. They are not physical 3D joint angles and can change with camera viewpoint, perspective, or out-of-plane motion.

Left and right always mean the subject's anatomical left and right. A mirrored preview transforms display x coordinates only; it does not rename landmarks. Estimated model depth uses `EstimatedDepth` and is never represented as measured physical depth.

Capture timestamps use `std::chrono::steady_clock` after a frame has been copied into its owned buffer at the capture boundary. Acquisition duration covers the blocking read or synthetic generation through completion of that copy. Inference completion timestamps use the same clock after postprocessing. They are suitable for durations and ordering, not wall-clock correlation. Frame sequence identifiers are monotonic within one source start.

Prerecorded and deterministic synthetic frames also retain a source-frame index and source-media timestamp. Those fields identify benchmark inputs; they do not replace the steady-clock timestamps used for latency.

Confidence values are model outputs normalized to `[0, 1]`. They are not calibrated probabilities and are not comparable between arbitrary models. Non-finite or out-of-range confidence is rejected. Missing landmarks contain no coordinates, depth, or confidence.

## Threads and ownership

Each live or prerecorded frame source owns one capture thread. The source allocates an immutable frame, publishes a shared owner into a one-slot latest-frame buffer, and releases its local owner. Consumers retain frames only for their current work.

Pose Lab owns one inference thread and the UI thread. The inference thread waits for a sequence newer than the last processed sequence. The UI thread independently requests the newest frame. When capture outruns a consumer, replacement is counted and stale frames are discarded.

Pose Lab starts in frame-synchronized diagnostic mode. The inference result retains shared ownership of the exact immutable frame that produced it, and the UI draws its skeleton only on that frame. Low-latency mode displays the latest capture, but suppresses the skeleton when the pose and display sequences differ. Diagnostics always show both sequence identifiers, match state, and pose age.

ONNX Runtime sessions belong exclusively to the estimator. Pose Lab calls an estimator from only its inference thread. The benchmark calls it serially from the main thread.

Synchronization occurs at the latest-frame buffer, the source statistics/error state, and Pose Lab's latest-inference snapshot. No communication queue is unbounded.

Tracking has three explicit states: unavailable, person detected, and usable pose. A usable pose requires fresh evidence and enough confident landmarks. Inference failure immediately clears prior evidence. Source failure, capture stall, missing pose output, insufficient confidence, and the configured freshness deadline all fail closed. This validity layer does not perform identity tracking or gesture recognition.

## Error propagation and shutdown

Boundary operations return `Status` or `Result<T>` with a typed error code and message. Backend exceptions are translated at the boundary. CUDA setup errors are returned and never trigger automatic CPU fallback.

Frame-source instances are deliberately single-use. A successful `start()` consumes the instance even if the worker later reaches end-of-stream or fails; callers create a new source to restart. This matches the one-way close semantics of the bounded latest-frame buffer. `stop()` remains idempotent. Capture-thread exceptions become terminal source errors instead of escaping the worker.

Shutdown order is:

1. Request inference cancellation.
2. Stop the frame source and close its latest-frame buffer.
3. Join inference and capture threads.
4. Destroy the estimator sessions.
5. Destroy UI windows.

Source stop operations are idempotent. Disconnection and end-of-stream remain inspectable through `terminal_error()`.

OpenCV Media Foundation `read()` can block inside the backend. Stop requests cannot forcibly cancel that native call; shutdown joins after it returns. Camera drivers that never return from a read remain an external limitation and must be exercised during TITAN physical qualification.

## Timing semantics

Frame acquisition is measured at the source boundary. Frame age is measured at inference entry. Person detection includes detector preprocessing, execution, and output selection. Pose preprocessing, ONNX execution, and pose postprocessing are timed separately. End-to-end processing covers one estimator call. ONNX Runtime `Run` is synchronous for the configured session, so the recorded pose inference duration includes completion visible to the calling CPU thread; it is not labeled as raw GPU kernel time.

Initialization, model loading, and warm-up are excluded from measured benchmark samples.

Deterministic benchmarks use a synchronous sequential source and cannot skip or loop frames. Real-time simulation uses paced capture and the one-slot latest-frame contract, so sequence gaps measure realistic skipped input. Both modes are deadline- and failure-bounded.

RTMPose session creation validates the pinned input/output names, tensor ranks, and element types. Each inference validates exact detector and HALPE-26 SimCC dimensions and element counts before reading tensor memory. CUDA sessions disable CPU execution-provider fallback.
