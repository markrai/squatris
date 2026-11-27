# Phase 0A.3 CUDA Execution and Qualification Recovery

Status: `PHASE_0A3_PARTIAL`

Date: 2026-11-27  
Qualification host: `TITAN`  
GPU: NVIDIA GeForce RTX 4080 SUPER  
ONNX Runtime: `1.20.1`  
Audited policy: `squatris-ort-1.20.1-cuda-audited-v1`

Automated audited CUDA execution is working and fail-closed. Both pinned model sessions execute their neural computation on CUDA, the six observed CPU assignments are small shape/index operations bound to exact identities, CPU/CUDA numerical parity passes, CPU and CUDA builds pass, three deterministic and paced repeats pass, and an executable-bound five-minute looped performance run passes. Phase 0A.3 remains partial because representative qualification footage was unavailable, no camera was present during the final Pose Lab check, and the required human physical observations have not occurred.

## Root cause

`cuda-strict` sets `session.disable_cpu_ep_fallback=1`. ONNX Runtime 1.20.1 rejects the detector during session creation because its optimizer deliberately assigns four shape/index nodes to the default CPU execution provider. This behavior comes from [`fallback_cpu_capability.cc`](https://github.com/microsoft/onnxruntime/blob/v1.20.1/onnxruntime/core/framework/fallback_cpu_capability.cc): ONNX Runtime identifies CPU shape subgraphs and CPU-preferred nodes when CPU execution is expected to cost less than accelerator launch and transfer overhead. The pinned pose graph has the same behavior for two shape nodes.

The preserved final strict failure is `qualification/phase0a3/strict/strict-failure.json`:

> detector session initialization failed: This session contains graph nodes that are assigned to the default CPU EP, but fallback to CPU EP has been explicitly disabled by the user.

This is not a missing CUDA dependency and it is not neural computation falling back. Strict behavior is unchanged and no strict-to-audited downgrade occurs.

## Investigation method

The diagnostic sessions intentionally enabled both CUDA and CPU providers, verbose ONNX Runtime logging, and [ONNX Runtime profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html). They were marked diagnostic and are not production qualification evidence. Detector and pose were investigated independently.

Static parsing established:

| Session | Model SHA-256 | Original nodes | IR | Opset | Control flow |
| --- | --- | ---: | ---: | ---: | --- |
| Detector | `ceb11c07298f95c50d7c5abeb906d03340c85f23aa79e3e66966e7fb6c307250` | 336 | 6 | 11 | None |
| Pose | `26f3a19e61304a600dfb82d1001d41d24343b89fc70a33ffc84657e0b0bf2ecf` | 247 | 6 | 11 | None |

The final policy preflight observed the exact optimized inventories: 258 CUDA plus 4 CPU detector nodes, and 185 CUDA plus 2 CPU pose nodes. Required CUDA witnesses include detector `Conv_31`, `Conv_287`, and `NonMaxSuppression_490`, and pose `Conv_0`, `Conv_205`, `MatMul_221`, and `MatMul_252`. The evidence JSON contains the complete CUDA identity/operator inventory rather than only a percentage.

## Complete CPU-node inventory

The durations below are from the final one-execution preflight. The earlier five-execution verbose diagnostic observed the same nodes and stable microsecond-scale durations. Domain is `ai.onnx` for every node.

| Session | Stable node | Op/index | Input → output | Approx. bytes in/out | Final duration | Classification and justification | Transfer evidence |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| Detector | `Gather_523` | Gather/328 | INT64[2] + INT64 scalar → INT64 scalar | 24/8 | 13 µs | Reads the candidate extent from `Shape_522` for the TopK bound. | CPU shape metadata; no explicit copy event observed. |
| Detector | `Less_526` | Less/329 | 2 INT64 scalars → BOOL scalar | 16/1 | 18 µs | Compares the requested TopK bound with candidate count. | CPU scalar metadata and initializer; no explicit copy event observed. |
| Detector | `Where_527` | Where/330 | BOOL + 2 INT64 scalars → INT64 scalar | 17/8 | 17 µs | Selects the bounded TopK count. | All operands are CPU scalar metadata; no explicit copy event observed. |
| Detector | `Reshape_528` | Reshape/331 | INT64 scalar + INT64[1] → INT64[1] | 16/8 | 10 µs | Produces the one-element count tensor consumed by CUDA TopK. | CPU tensor is a shape/count input to CUDA TopK. |
| Pose | `Slice_209` | Slice/207 | INT64[4] + 3 INT64[1] → INT64[2] | 56/16 | 19 µs | Slices spatial dimensions from `Shape_206` for SimCC reshaping. | CPU shape metadata; no explicit copy event observed. |
| Pose | `Concat_211` | Concat/208 | INT64[2] + INT64[1] → INT64[3] | 24/24 | 20 µs | Builds the shape tensor consumed by CUDA `Reshape_212`. | CPU tensor is a shape input to the CUDA Reshape path. |

No Conv, MatMul, activation, normalization, detector scoring, NMS, or pose-head neural computation was assigned to CPU. The final detector and pose profiles reported zero explicit memory-copy events. That means no copy event was exposed by this profiler; it does not prove that the runtime performed no implicit metadata transfer.

All six assignments are approved only under the pinned manifest. No observed CPU node was rejected as substantive neural computation. The audited implementation rejects every CPU node not matching an exact session, model hash, name, operator, and optimized node index. `cuda-strict` rejects all six because that policy permits no CPU assignment.

## Execution policies and enforcement

- `cpu`: intentional CPU execution. It requires `CPUExecutionProvider` and never claims CUDA evidence.
- `cuda-strict`: CUDA with CPU fallback disabled. Legacy benchmark `--provider cuda` maps to this policy only. The pinned detector continues to fail closed.
- `cuda-audited`: CUDA-required neural computation with only the exact six pinned CPU metadata nodes allowed.

The policy manifest is `config/rtmpose-cuda-policy.json`, SHA-256 `ccc6b13bc9176fe2cf1fb42911c201d8d30fcf8d0c63ff987337dab9f02c10ab`. It binds ONNX Runtime `1.20.1`, `ORT_ENABLE_ALL`, `ORT_SEQUENTIAL`, both model hashes, session names, original graph facts, exact optimized node counts, required CUDA neural nodes, and exact CPU nodes.

Audited estimator creation performs detector and pose preflight inference on dedicated profiling sessions before returning the estimator. Both profiles must independently reproduce the manifest. Only then are separate production sessions created with the same provider, runtime, optimization, execution-mode, device, and model configuration. Pose Lab uses this same creation path and atomically writes its policy evidence before starting capture. Missing CUDA dependencies, unavailable CUDA provider, model mismatch, modified policy or profile, missing CUDA witnesses, missing or additional CPU nodes, and unknown providers all fail initialization. A failed GPU session is never recreated as CPU.

Final unprofiled preflight artifacts:

| Session | Profile SHA-256 | CUDA/CPU/unknown nodes | Copy events |
| --- | --- | ---: | ---: |
| Detector | `8257d8d70f333f49c85aa74f0f7eb067cf7f20860b40e6561421fe99b372be94` | 258/4/0 | 0 |
| Pose | `7fafec3ebc4c69b3deace36b16646906e0c3969884590b094802cbdac0a89aa3` | 185/2/0 | 0 |

The benchmark validator independently reopens and hashes each profile, rebuilds every node aggregate, re-applies the manifest, and requires exact JSON equality. Corruption tests reject provider identity, policy identity/hash, profile hash, node assignment, and aggregate-statistic changes.

### Enforcement limitations

ONNX Runtime 1.20.1 does not expose a separate public static partition manifest through this integration. The enforcement combines exact optimized-node counts, stable profile node identities, static graph inspection, and the fact that both pinned graphs are straight-line graphs with no `If`, `Loop`, or `Scan`. Runtime counts and durations still describe observed preflight execution only.

Unprofiled production sessions are recreated after successful preflight so performance measurements contain no profiling instrumentation. Their assignment is expected to be deterministic for the identical runtime, model, and session configuration; ordinary unprofiled sessions cannot independently emit assignment evidence. The bounded profiling-enabled production run created separate production sessions and reproduced the same policy, which supports that assumption but does not turn a private runtime implementation detail into a formal API guarantee.

## CPU/CUDA numerical parity

The final comparison used the same four measured frames from the official five-frame MMPose demo, pinned models, deterministic order, a clean CPU ONNX Runtime package, and audited CUDA. Evidence is in `qualification/phase0a3/cpu-package/cpu-vs-cuda-parity.json`.

| Check | Tolerance | Observed |
| --- | ---: | ---: |
| Detection availability mismatches | 0 | 0 |
| Pose availability mismatches | 0 | 0 |
| Landmark availability mismatches | 0 | 0 |
| Full-body coverage mismatches | 0 | 0 |
| Minimum selected-box IoU | ≥ 0.995 | 0.999758 |
| Maximum normalized landmark delta | ≤ 0.002 | 0.001612 |
| Maximum confidence delta | ≤ 0.005 | 0.002446 |
| Maximum projected knee-angle delta | ≤ 2.0° | 1.415° |

Canonical subject-left and subject-right identities were compared without preview mirroring. The 2° angle tolerance accounts for amplification of sub-pixel coordinate differences by projected angle geometry. This proves numerical agreement for these frames only. It does not prove physical accuracy, and the available clip does not cover the requested standing, squat, lateral, arm, occlusion, and no-person cases.

## Performance and profiling overhead

Every final performance report is bound to benchmark executable SHA-256 `35c6d09302112618a42baa5c2f3e6fdbad6192596917394540eae8c33047e58c`, the runtime, model hashes, policy hash, and provider evidence.

Three unprofiled deterministic four-frame smoke repeats:

| Run | Median end-to-end | p95 | Throughput |
| --- | ---: | ---: | ---: |
| 1 | 12.877 ms | 13.121 ms | 74.894 FPS |
| 2 | 12.626 ms | 12.910 ms | 75.717 FPS |
| 3 | 12.760 ms | 12.988 ms | 74.908 FPS |

Three paced four-frame repeats had median end-to-end latency of 13.262, 13.392, and 13.087 ms. Their short-interval throughput figures are not used as a source-rate claim; the source is paced at 5 FPS.

The final clean CPU-package run measured 35.004 ms median end-to-end and 27.659 FPS over the same four measured frames. This is comparison evidence, not a long-run CPU performance claim.

Profiling overhead used identical one-warm-up/four-measured-frame runs:

| Mode | Median end-to-end | Throughput |
| --- | ---: | ---: |
| Audited production, profiling disabled | 12.823 ms | 71.409 FPS |
| Audited production, profiling enabled | 19.293 ms | 50.616 FPS |

The measured median overhead was 6.471 ms, or 50.46%. Profiled latency is not reported as production latency. Production profiling is CLI-gated to at most 100 total frames to bound trace size and memory.

## Five-minute sustained run

The final report is `qualification/phase0a3/sustained/five-minute-looped-final.json`; resource evidence is `five-minute-looped-final.resources.json`.

- Measured interval: 300.038 seconds.
- Frames: 1,501 measured, 0 skipped, 1,501 usable poses.
- Loop identities: 301 explicit loops over the five-frame clip.
- Paced throughput: 5.003 FPS.
- End-to-end latency: median 66.122 ms, p95 77.028 ms, p99 83.943 ms.
- Resource samples: 273; monitoring errors: 0.
- Process CPU: 1.475% average, normalized across 24 logical processors.
- Working set: 1,098.6 MiB maximum.
- GPU utilization: 31.44% average, 75% maximum.
- Device-wide GPU memory reported by `nvidia-smi`: 8,950.2 MiB average, 9,221 MiB maximum. This is not process-exclusive VRAM.

The large difference between the short deterministic median and sustained paced median is recorded rather than hidden. The sustained run includes a concurrent once-per-second `nvidia-smi` monitor and long-running paced capture. It met the measured-duration requirement with no deadline or monitoring error, but the repeated five-frame clip is not suitable content qualification footage.

## Infrastructure and architecture changes

- Added exact `cpu`, `cuda-strict`, and `cuda-audited` policy identities to the estimator interface, benchmark, and Pose Lab.
- Added `config/rtmpose-cuda-policy.json` and a backend provider-audit module with independent profile hashing, node aggregation, and policy validation.
- Moved reusable SHA-256 support into the dependency-free core.
- Added separate preflight and production sessions; production profiling is explicit and bounded.
- Added source loop identity, real duration mode, atomic report replacement, and measured-duration validation.
- Added full per-frame person, landmark, confidence, knee-angle, and coverage evidence plus CPU/CUDA parity comparison.
- Added executable hashing and recomputation of latency aggregates from samples.
- Added provider/policy/profile/node/aggregate corruption rejection.
- Hardened the watchdog with bounded process-tree termination, bounded `nvidia-smi`, explicit monitoring errors, atomic evidence, and mandatory `TITAN` host identity.
- Hardened CPU/GPU ONNX Runtime acquisition with package SHA-512 and per-artifact SHA-256/length checks. CPU and GPU header layouts are handled independently.
- Added MSVC-pinned CUDA Debug and Release presets and self-initializing Visual Studio environment discovery in qualification.
- Extracted provider policy analysis and cryptographic hashing from the benchmark. Command parsing, execution, schema generation, and general report validation remain in `tools/benchmark/main.cpp`; further extraction is reasonable when the schema changes again, but was not made speculative in this phase.

## Builds and tests

| Configuration | Result |
| --- | --- |
| CUDA Release, MSVC, warnings as errors | Built; 44/44 tests passed |
| CUDA Debug, MSVC, warnings as errors | Built; 44/44 tests passed |
| CPU ONNX Release from clean CPU package | Built; 44/44 tests passed |
| Intentional CPU pinned-model inference | Passed and report validated |
| `cuda-strict` | Failed closed as expected; failure report validated |
| `cuda-audited` detector and pose | Passed policy and real inference |
| Provider corruption tests | All corruptions rejected |

## Remaining blockers and physical observations

The only available video is the official MMPose five-frame demo, SHA-256 `dc788b413f384887dab9502197657c05393efd1d26825831e4da6fa75b4908c9`. It is suitable for loading, provider, parity, and looped performance smoke work only.

The final OpenCV Media Foundation enumeration reported no cameras, and Windows PnP returned no Camera/Image devices. A camera-index-0 Pose Lab attempt still completed the audited preflight and atomically wrote evidence with two verified sessions, zero verification errors, zero unexpected CPU nodes, and zero unknown-provider nodes before capture/UI startup failed: Media Foundation could not open the index and this OpenCV build did not provide `cvNamedWindow`. Pose Lab therefore could not be exercised against the Logitech BRIO during this final run. No human participant assessed anatomical mapping, squat continuity, lateral motion, arm motion, occlusion, tracking loss, reacquisition, disconnect behavior, delivered 60 FPS, or sustained responsiveness. No gesture recognition was added.

## Exact physical-test commands

Run from `C:\dev\project\squatris` after connecting the BRIO. First enumerate and confirm its actual index and negotiated modes:

```powershell
$env:PATH = "$(Join-Path $PWD '.deps\cudnn\9.1.0.70\bin');$(Join-Path $env:CUDA_PATH 'bin');$env:PATH"
.\out\build\windows-cuda-release\apps\pose-lab\SquatrisPoseLab.exe --list-cameras
```

For 1920×1080 at requested 60 FPS:

```powershell
.\out\build\windows-cuda-release\apps\pose-lab\SquatrisPoseLab.exe `
    --camera 0 --width 1920 --height 1080 --fps 60 `
    --backend rtmpose --provider cuda-audited `
    --detector models\detector.onnx --pose models\pose.onnx `
    --policy-manifest config\rtmpose-cuda-policy.json `
    --provider-evidence-dir qualification\phase0a3\pose-lab-1080p
```

For 1280×720 at requested 60 FPS:

```powershell
.\out\build\windows-cuda-release\apps\pose-lab\SquatrisPoseLab.exe `
    --camera 0 --width 1280 --height 720 --fps 60 `
    --backend rtmpose --provider cuda-audited `
    --detector models\detector.onnx --pose models\pose.onnx `
    --policy-manifest config\rtmpose-cuda-policy.json `
    --provider-evidence-dir qualification\phase0a3\pose-lab-720p
```

Use the enumerated BRIO index if it is not `0`. Record requested, negotiated, and delivered capture FPS separately. With diagnostics visible, assess subject-left/right under mirrored and unmirrored preview, standing, slow and fast squats, lateral motion, independent and combined arm motion, wrist and leg occlusion, leave/reenter, a second person entering, camera disconnection, tracking reacquisition, and latency accumulation. Press `F` to compare frame-synchronized and low-latency preview; a mismatched skeleton must remain suppressed.

## Suggested commit message

`phase 0a3: enforce audited CUDA execution and recover qualification`

Do not begin Phase 0B while this report is partial.
