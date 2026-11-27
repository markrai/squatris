# Benchmarking

`SquatrisPoseBenchmark` runs the same estimator used by Pose Lab in one of two explicit modes.

- `deterministic` synchronously decodes every prerecorded or generated frame in exact order. It never loops, skips, or drops input.
- `realtime` paces input at the source rate and preserves one-slot latest-frame semantics. It measures sequence gaps, throughput, and latency while capture and inference compete in wall time.

## Commands

Hardware-independent deterministic smoke benchmark:

```powershell
.\out\build\windows-release\tools\benchmark\SquatrisPoseBenchmark.exe --mode deterministic --backend mock --warmup 20 --iterations 500 --output benchmarks\results\mock.json
```

CUDA benchmark with prerecorded footage:

```powershell
.\out\build\windows-cuda-release\tools\benchmark\SquatrisPoseBenchmark.exe --mode deterministic --video captures\qualification.mp4 --backend rtmpose --provider cuda --detector models\detector.onnx --pose models\pose.onnx --manifest models\manifest.json --warmup 100 --iterations 2000 --deadline-seconds 1800 --output benchmarks\results\titan-cuda.json
```

Real-time simulation of the same footage:

```powershell
.\out\build\windows-cuda-release\tools\benchmark\SquatrisPoseBenchmark.exe --mode realtime --video captures\qualification.mp4 --backend rtmpose --provider cuda --detector models\detector.onnx --pose models\pose.onnx --manifest models\manifest.json --warmup 100 --iterations 2000 --deadline-seconds 1800 --output benchmarks\results\titan-cuda-realtime.json
```

## Methodology

Model and session initialization occur before timing. Warm-up frames execute the complete pipeline and have their own statistics and frame evidence; they never enter the measured statistics. Both phases record frame acquisition, frame age at inference start, person detection, pose preprocessing, synchronous ONNX Runtime execution, pose postprocessing, and estimator end-to-end duration.

Median, p95, and p99 use linear interpolation over sorted samples. Every processed frame records its source sequence, source-frame index, source-media timestamp, person-detection state, and usable-pose state. Deterministic reports are invalid if any frame was skipped. Real-time reports retain sequence gaps as skipped frames. Errors and frames with no usable pose are counted separately. Throughput is successful pipeline iterations per phase wall time; it must not be interpreted as camera delivery rate.

Schema 2 output includes the source-video SHA-256, exact processed-frame evidence, provider, verified model hashes, build and dependency data, host data, execution mode, warm-up and measured statistics, skips, deadlines, throughput, and terminal errors. Unknown providers are rejected. Repeated acquisition or inference failures stop at `--max-consecutive-failures`; the whole execution stops at `--deadline-seconds`.

The deterministic mode does not loop. The input must contain at least `warmup + iterations` frames. A short or corrupt input produces a failed report instead of silently reusing footage.

Validate any saved report programmatically:

```powershell
.\out\build\windows-cuda-release\tools\benchmark\SquatrisPoseBenchmark.exe --validate-report benchmarks\results\titan-cuda.json
```

Validation checks the schema, provider, checksum evidence, phase counts, frame evidence, source hash, success consistency, and deterministic zero-skip invariant.

CPU mock results validate infrastructure only. They are not RTMPose performance evidence. CUDA results are valid only when the reported provider is `CUDAExecutionProvider`, `strict_provider_assignment` is true, model checksums match the authoritative manifest, report validation succeeds, and the report status is successful.

The 60-update/s and 80 ms response targets remain engineering targets. Phase 0A.1 does not claim they have been met.
