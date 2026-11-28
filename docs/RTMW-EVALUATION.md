# Phase 0B.2: RTMW-m candidate integration

RTMW-m at 256x192 is the initial candidate. OpenMMLab publishes a 133-keypoint Cocktail14 model at this resolution. The evaluation keeps the pinned YOLOX-tiny detector, crop, resolution, canonical skeleton, timestamp semantics and confidence gating unchanged. RTMW maps COCO-WholeBody body landmarks and big toes/heels to the same canonical names. Face and finger landmarks are not exposed. The existing HALPE-26 reference remains the default in Pose Lab and the benchmark.

References: [RTMW release](https://github.com/open-mmlab/mmpose/releases), [RTMlib model zoo](https://github.com/Tau-J/rtmlib), [COCO-WholeBody metadata](https://github.com/open-mmlab/mmpose/blob/main/configs/_base_/datasets/coco_wholebody.py). The candidate uses the shared RGB affine normalization and SimCC decoder provisionally; verify preprocessing and numerical output against the SDK export metadata and upstream reference on the actual artifact before qualification. Model weights and dataset terms require separate review.

## Pinned artifact

The official [RTMW-m ONNX SDK](https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip) was downloaded and hashed on TITAN. `models/rtmw-m-manifest.json` pins the archive and ONNX bytes separately, together with the unchanged reference detector. These are locally observed checksums, not independently authenticated upstream signatures. Preprocessing and output verification remain pending.

For a fresh installation, reuse the existing downloader without touching the baseline pose weights:

```powershell
pwsh -NoProfile -File .\scripts\acquire-models.ps1 -Manifest models\rtmw-m-manifest.json
```

If the verified ZIP has already been extracted under `qualification/rtmw-download`, reuse the local ONNX file instead of downloading again:

```powershell
$files = @(Get-ChildItem qualification\rtmw-download -Recurse -Filter *.onnx -File)
if ($files.Count -ne 1) { throw 'Expected exactly one ONNX file' }
pwsh -NoProfile -File .\scripts\prepare-rtmw-candidate.ps1 -PoseFile $files[0].FullName
```

The preparation script checks the repository pin, copies into a new ignored output directory, and copies the pinned manifest. It does not overwrite the baseline or accept a different caller-supplied checksum.

## CPU comparison smoke

Build `windows-onnx-release` with the pinned CPU runtime, or reuse a current passing `ci-windows-onnx-release` executable via `-Benchmark`. Run both backends on the same smoke workload. The examples assume the local preparation route above; after fresh acquisition use `models/rtmw-m.onnx` and `models/rtmw-m-manifest.json` instead.

```powershell
pwsh -NoProfile -File .\scripts\run-pose-workload.ps1 -Workload config\pose-workload-smoke.json -OutputDirectory qualification\phase0b2\reference-cpu -Provider cpu
pwsh -NoProfile -File .\scripts\run-pose-workload.ps1 -Workload config\pose-workload-smoke.json -OutputDirectory qualification\phase0b2\rtmw-cpu -Backend rtmw -Provider cpu -PoseModel qualification\rtmw-m-artifact\rtmw-m.onnx -ModelManifest qualification\rtmw-m-artifact\manifest.json
pwsh -NoProfile -File .\scripts\compare-pose-workloads.ps1 -Workload config\pose-workload-smoke.json -ReferenceDirectory qualification\phase0b2\reference-cpu -CandidateDirectory qualification\phase0b2\rtmw-cpu -Output qualification\phase0b2\comparison-cpu.json
```

The comparator currently requires the CUDA Release validator executable even for CPU reports; build it from the same revision. CPU is an intentional diagnostic, not silent fallback or a GPU-performance result. The same five-frame smoke is not tracking-accuracy evidence.

The downloaded RTMW-m export reports a symbolic landmark dimension (`-1`) in ONNX Runtime output metadata. For this candidate's SimCC outputs only, initialization accepts a symbolic dimension or the expected static value, while rejecting incompatible static dimensions. Each real inference still requires exact float32 tensor shapes `[1,133,384]` and `[1,133,512]` and corresponding element counts before any tensor data is read. The detector and input contracts are unchanged.

## GPU gate

Use the standalone [candidate CUDA profiler](CANDIDATE-CUDA-PROFILING.md) to collect observed execution evidence before writing a reviewed candidate policy. The diagnostic does not authorize production fallback.

The candidate-specific reviewed policy is `config/rtmw-m-cuda-policy.json`. It copies the existing detector inventory exactly, pins the RTMW model hash, expects 293 CUDA nodes and four exact CPU shape operations (297 observed optimized nodes), and requires the observed convolution and all eight matrix multiplication identities on CUDA. The four CPU nodes are Slice_341 (338), Concat_343 (339), Slice_361 (356), and Concat_362 (357). All observed operands and results are small INT64 vectors. The two reshape-metadata sequences are supported by the observed trace names and tensor shapes; exact original edge connectivity was not independently inspected. Their combined output size was 80 bytes and their combined cold-trace event duration was 128 microseconds, not a sustained timing measurement.

TITAN static inspection with ONNX 1.23.2 reported IR 6, opset 11, 405 original nodes, no top-level If/Loop/Scan, no graph-valued attributes, and no external graph initializers. The uploaded diagnostic trace and report hashes match the run manifest, and the 297 node records match the raw events. SimCC X and Y returned float32 [1,133,384] and [1,133,512]. This is evidence for the reviewed allowlist, not complete static provider coverage. Runtime artifact hashes are recorded as provenance; the existing auditor records actual DLL hashes but does not enforce these provenance hashes. The baseline policy is unchanged. No automatic policy generation or relaxation of the audit was introduced.

Run the paired CPU and audited CUDA smoke with the same CUDA Release executable (it supports explicit CPU execution). Build it first if needed. Use new directories:

```powershell
$benchmark = 'out\build\windows-cuda-release\tools\benchmark\SquatrisPoseBenchmark.exe'
$common = @{
    Workload = 'config\pose-workload-smoke.json'
    Backend = 'rtmw'
    PoseModel = 'qualification\rtmw-m-artifact\rtmw-m.onnx'
    ModelManifest = 'qualification\rtmw-m-artifact\manifest.json'
    Benchmark = $benchmark
}
pwsh -NoProfile -File .\scripts\run-pose-workload.ps1 -Workload $common.Workload -Backend $common.Backend -PoseModel $common.PoseModel -ModelManifest $common.ModelManifest -Benchmark $benchmark -Provider cpu -OutputDirectory qualification\phase0b2\rtmw-parity-cpu
pwsh -NoProfile -File .\scripts\run-pose-workload.ps1 -Workload $common.Workload -Backend $common.Backend -PoseModel $common.PoseModel -ModelManifest $common.ModelManifest -Benchmark $benchmark -Provider cuda-audited -PolicyManifest config\rtmw-m-cuda-policy.json -OutputDirectory qualification\phase0b2\rtmw-parity-cuda
& $benchmark --compare-cpu-report qualification\phase0b2\rtmw-parity-cpu\mmpose-demo.json --compare-cuda-report qualification\phase0b2\rtmw-parity-cuda\mmpose-demo.json --parity-output qualification\phase0b2\rtmw-parity.json
```

If the candidate fails the audit or parity, preserve the evidence and stop; do not widen the allowlist or tolerance merely to obtain a pass. The existing parity thresholds are unchanged. Successful preflight and parity are prerequisites to a separate GPU performance comparison; neither establishes physical accuracy or production readiness.

## Acceptance remaining

Validate tensor metadata and actual shape/element counts on TITAN, compare decoded coordinates and scores against the upstream SDK, capture CPU/CUDA numerical parity, and compare representative non-looped recordings with matching workload hashes. Run idle and contended paced workloads with resource evidence once the candidate policy is approved. Complete annotated localization, coverage, occlusion/re-entry and representative full-body camera checks before selecting a winner. The candidate is benchmark-only in this slice; no default live Pose Lab model change. CPU smoke and CI passed on TITAN; the new policy still requires actual audited CUDA preflight and parity.
