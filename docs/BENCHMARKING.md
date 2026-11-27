# Benchmarking

`SquatrisPoseBenchmark` runs the same estimator and execution policies used by Pose Lab.

- `deterministic` decodes every prerecorded or generated frame in exact order. It never loops, skips, or drops input.
- `realtime` paces input at the source rate and preserves one-slot latest-frame semantics. It measures sequence gaps, throughput, and latency while capture and inference compete in wall time.

The execution policies are `cpu`, `cuda-strict`, and `cuda-audited`. The legacy benchmark spelling `--provider cuda` maps only to `cuda-strict`; it never silently selects the audited policy. `cuda-audited` requires `config/rtmpose-cuda-policy.json`, runs detector and pose preflight sessions, rejects any assignment outside that pinned policy, and then creates separate unprofiled production sessions.

## Commands

Audited CUDA deterministic benchmark:

```powershell
.\out\build\windows-cuda-release\tools\benchmark\SquatrisPoseBenchmark.exe `
    --mode deterministic --video captures\qualification.mp4 `
    --backend rtmpose --provider cuda-audited `
    --detector models\detector.onnx --pose models\pose.onnx `
    --manifest models\manifest.json --policy-manifest config\rtmpose-cuda-policy.json `
    --warmup 100 --iterations 2000 --deadline-seconds 1800 `
    --output benchmarks\results\titan-cuda.json
```

Paced version of the same run:

```powershell
.\out\build\windows-cuda-release\tools\benchmark\SquatrisPoseBenchmark.exe `
    --mode realtime --video captures\qualification.mp4 `
    --backend rtmpose --provider cuda-audited `
    --detector models\detector.onnx --pose models\pose.onnx `
    --manifest models\manifest.json --policy-manifest config\rtmpose-cuda-policy.json `
    --warmup 100 --iterations 2000 --deadline-seconds 1800 `
    --output benchmarks\results\titan-cuda-realtime.json
```

Five-minute paced performance run with explicit looping:

```powershell
.\out\build\windows-cuda-release\tools\benchmark\SquatrisPoseBenchmark.exe `
    --mode realtime --video captures\qualification.mp4 --loop-video true `
    --duration-seconds 300 --warmup 100 --iterations 1 `
    --backend rtmpose --provider cuda-audited `
    --detector models\detector.onnx --pose models\pose.onnx `
    --manifest models\manifest.json --policy-manifest config\rtmpose-cuda-policy.json `
    --deadline-seconds 900 --output benchmarks\results\titan-cuda-sustained.json
```

Duration mode stops only after the measured steady-state interval reaches the requested duration. It is available only for prerecorded real-time mode. A looped frame records its loop identity, source index, and source timestamp; index and timestamp resets are valid only when the loop identity increments by one. Deterministic comparisons never loop.

Measure instrumentation overhead with otherwise identical audited runs. `--profile-production true` is limited to iteration mode and at most 100 total warm-up plus measured frames so trace output remains bounded:

```powershell
.\out\build\windows-cuda-release\tools\benchmark\SquatrisPoseBenchmark.exe `
    --mode deterministic --video captures\qualification.mp4 `
    --backend rtmpose --provider cuda-audited --profile-production true `
    --detector models\detector.onnx --pose models\pose.onnx `
    --manifest models\manifest.json --policy-manifest config\rtmpose-cuda-policy.json `
    --warmup 10 --iterations 50 --output benchmarks\results\titan-cuda-profiled.json
```

Compare CPU and audited CUDA numerically:

```powershell
.\out\build\windows-cuda-release\tools\benchmark\SquatrisPoseBenchmark.exe `
    --compare-cpu-report benchmarks\results\cpu.json `
    --compare-cuda-report benchmarks\results\cuda-audited.json `
    --parity-output benchmarks\results\cpu-cuda-parity.json
```

Validate a saved report and every referenced provider profile:

```powershell
.\out\build\windows-cuda-release\tools\benchmark\SquatrisPoseBenchmark.exe `
    --validate-report benchmarks\results\titan-cuda.json
```

## Evidence and validation

Schema 4 records the source SHA-256, benchmark executable path and SHA-256, model hashes, runtime and build identities, execution policy, provider evidence, full canonical landmark samples, selected person boxes, pose confidence, projected knee angles, loop identities, latency samples, recomputed aggregates, and terminal status. Reports are written to a temporary file in the destination directory and atomically replaced only after the stream closes successfully.

Model and session initialization and first-inference latency are separate from measured samples. Warm-up frames execute the full pipeline but never enter measured statistics. The unprofiled audited production sessions are created only after both preflight sessions pass the pinned policy. Profiling-enabled production runs are labeled explicitly and are not production-latency evidence.

Validation independently reads and hashes referenced ONNX Runtime profiles, reconstructs their node inventories, re-applies the pinned policy, recomputes median/p95/p99 from individual frame samples, and rejects modified provider, policy, profile, node, phase, ordering, loop, aggregate, or terminal-error evidence. Unknown providers and unexpected CPU nodes fail validation.

The CPU/CUDA parity command requires successful deterministic reports with the same source and model hashes. It compares person availability, bounding-box IoU, canonical landmark availability and coordinates, detection/pose/landmark confidences, anatomical left/right identities, projected knee angles, and full-body coverage. The current tolerances are IoU at least `0.995`, normalized coordinate difference at most `0.002`, confidence difference at most `0.005`, and projected knee-angle difference at most `2.0` degrees. Agreement is numerical parity, not proof of physical accuracy.

The 60-update/s and 80 ms response targets remain engineering targets. A short or looped clip is performance evidence only and cannot replace representative physical qualification.
