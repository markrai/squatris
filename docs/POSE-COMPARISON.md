# Phase 0B.1: Pose model comparison

The existing RTMPose-m HALPE-26 backend is the reference. This phase introduces the comparison workload and evidence contract only. It does not add another inference backend, select a winner, change the CUDA policy or claim recognition accuracy.

## Workload

A schema-1 JSON manifest contains a `clips` array. Each entry has a unique lowercase identifier, a scenario, a path relative to the manifest directory (or an absolute path), the lowercase SHA-256 of the video bytes, the number of warm-up frames to consume from the start of the clip, and the number of subsequent measured frames. The file must contain at least that many frames. Up to 64 clips, 300 warm-up frames and 3600 measured frames per clip are supported. Each clip starts a new inference session. Never loop, seek, change the delivery rate, or drop frames for these comparisons.

The included `config/pose-workload-smoke.json` pins the existing five-frame MMPose demo. It is a wiring test, not representative physical evidence.

Record representative fixed-camera, full-body clips with enough lead-in for warm-up. Keep the camera, framing and lighting stable for stationary clips. Use separate recordings for normal and rapid squats, lateral steps, each arm swipe, combined movements, partial occlusion, an empty scene, re-entry, and a second person entering. The scenario names are standing, squat, lateral, swipe, combined, occlusion, empty, reentry, and multiple-people. Repeat conditions across viewpoints, lighting, distance and clothing as practical. Record both expected successes and difficult/negative cases. Store recordings and derived evidence under ignored captures/ or qualification/ directories; do not publish footage.

Example private workload creation (adjust the counts to the actual recording):

```powershell
$clip = (Resolve-Path 'captures\standing.mp4').Path
$workload = @{
    schema_version = 1
    clips = @(@{
        id = 'standing'
        scenario = 'standing'
        path = $clip
        sha256 = (Get-FileHash -LiteralPath $clip -Algorithm SHA256).Hash.ToLowerInvariant()
        warmup_frames = 30
        measured_frames = 300
    })
}
$workload | ConvertTo-Json -Depth 6 | Set-Content qualification\pose-workload.json -Encoding utf8
```

Freeze the manifest before comparing. The report bundle binds its exact bytes and every clip hash. Paths can be relative for portability, but changing manifest bytes requires a new run.

## Execute the baseline

Use the existing Release build with its pinned runtime, weights and CUDA policy. Run from repository root in PowerShell 7:

```powershell
pwsh -NoProfile -File .\scripts\run-pose-workload.ps1 -Workload config\pose-workload-smoke.json -OutputDirectory qualification\phase0b1\reference
```

The runner defaults to audited CUDA; use `-Provider cpu` with the ONNX CPU Release build for a CPU diagnostic. Strict CUDA is an explicit option and is expected to fail on the current pinned graphs. No silent fallback is introduced. Output must be a new directory. Each clip produces a native schema-4 report, provider evidence where applicable, stdout/stderr logs and a completed run.json manifest only after every clip succeeds. The native validator rechecks each report. The runner has an external process deadline in addition to the benchmark's internal deadline.

Repeat independently into a new candidate directory to exercise the harness with the same backend. A second run of the same model measures repeatability, not a different architecture:

```powershell
pwsh -NoProfile -File .\scripts\run-pose-workload.ps1 -Workload config\pose-workload-smoke.json -OutputDirectory qualification\phase0b1\repeat
pwsh -NoProfile -File .\scripts\compare-pose-workloads.ps1 -Workload config\pose-workload-smoke.json -ReferenceDirectory qualification\phase0b1\reference -CandidateDirectory qualification\phase0b1\repeat -Output qualification\phase0b1\comparison.json
```

The comparison uses the existing CUDA Release benchmark executable as its report validator. The reference and candidate must have been run on the same named machine and OpenCV version. Source SHA-256, image dimensions, native rate, sequential frame indices and media timestamps must agree across warm-up and measured phases. Individual run manifests bind the model, execution policy, executable, build, host and individual report hashes. A mismatch fails instead of creating a successful comparison. Profile artifacts must remain accessible at the paths required by the native validator. The final comparison is atomically published without replacing an existing result.

## Interpreting metrics

The comparator reports each clip and each backend separately. It includes raw canonical landmark availability for the 12 shoulder, elbow, wrist, hip, knee and ankle points, usable-pose and complete-body counts at the recorded backend policy, projected knee-angle availability, deterministic throughput, estimator median/p95/p99 latency, and standing-only frame-to-frame RMS displacement normalized by the image diagonal. Missing landmarks break displacement pairs. Sample counts are retained. This displacement measures observed movement, including real subject motion and camera noise; it is not a pure jitter measurement.

Confidence outputs are not calibrated probabilities and must not be compared numerically across models. Availability does not establish correct localization. A model with no output during an occlusion might be safer than a model that confidently tracks the wrong person. Deterministic throughput is not live capture FPS; per-frame JSON bookkeeping also contributes to benchmark wall time. No automatic model ranking is produced.

Source-byte and frame-identity checks ensure matching requested decoded frame identities through the common decoder; decoded pixel bytes are not independently hashed in this phase. Future adapters must use the same decode path or add explicit decoded-frame equivalence evidence. This runner currently invokes RTMPose only. Future adapters must implement canonical landmark mapping and appropriate model/runtime/provider validation before extending the report validator and runner. Do not accept unvalidated third-party JSON or relax audited CUDA enforcement to onboard a new model. Resource measurements, paced/sustained comparisons, annotated landmark localization accuracy, gesture recall, identity recovery and physical-angle ground truth are later evaluation work. Run equivalent idle/contended conditions and record environment differences; same host alone does not guarantee identical load.

## Verification

The shared CI setup runs scripts/tests/pose-comparison.tests.ps1 using small synthetic JSON contract fixtures without hardware, model weights or video codecs. It checks manifest hashing, duplicate IDs, rejection of changed input, frame identity, invalid run configuration and aspect-corrected displacement with missing landmarks. These are not native inference tests. The smoke run above is the end-to-end acceptance step on TITAN. Keep raw reports and provider evidence for independent revalidation.

The Phase 0A.4 idle performance run achieved approximately 59.97 useful pose updates per second over 300 seconds with 14.08 ms p95 estimator latency on simulated pacing. The user also reported satisfactory qualitative joint tracking in the live Pose Lab. This allows comparison infrastructure work to proceed; it does not establish measured webcam delivery, physical localization accuracy or bounded long-session inference memory.
