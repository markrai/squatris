# Benchmarking

`SquatrisPoseBenchmark` runs the same estimator used by Pose Lab against a prerecorded file or deterministic synthetic frames. Prerecorded mode is independent of a physical camera and does not pace to wall time.

## Commands

Hardware-independent smoke benchmark:

```powershell
.\out\build\windows-release\tools\benchmark\SquatrisPoseBenchmark.exe --backend mock --warmup 20 --iterations 500 --output benchmarks\results\mock.json
```

CUDA benchmark with prerecorded footage:

```powershell
.\out\build\windows-cuda-release\tools\benchmark\SquatrisPoseBenchmark.exe --video captures\qualification.mp4 --backend rtmpose --provider cuda --detector models\detector.onnx --pose models\pose.onnx --warmup 100 --iterations 2000 --output benchmarks\results\titan-cuda.json
```

## Methodology

Model/session initialization occurs before timing. Warm-up frames execute the complete pipeline but are excluded from percentile samples. Measured samples record frame acquisition, frame age at inference start, person detection, pose preprocessing, synchronous ONNX Runtime execution, pose postprocessing, and estimator end-to-end duration.

Median, p95, and p99 use linear interpolation over sorted samples. Source sequence gaps are reported as skipped frames. Errors and frames with no detected pose are counted separately. Throughput is useful pose pipeline iterations per measured wall time; it must not be interpreted as camera delivery rate.

The output includes input dimensions, provider, model hashes, build/dependency data, host data, configuration, frames processed/skipped, throughput, percentiles, and errors.

CPU mock results validate infrastructure only. They are not RTMPose performance evidence. CUDA results are valid only when the reported provider is `CUDAExecutionProvider` and the model checksums match the manifest.

The 60-update/s and 80 ms response targets remain engineering targets. Phase 0A does not claim they have been met.
