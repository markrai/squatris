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

RTMW strict CUDA is expected to fail if the common detector requires CPU shape operations. Do not bypass the existing audit. For audited CUDA, generate a separate candidate-specific policy after inspecting the exact ONNX graph and recording node assignments on TITAN with the pinned runtime. Preserve the existing detector inventory; justify every candidate CPU node by exact name, index, operator, shapes and purpose. Reject any substantive CPU neural execution. Validate actual graph control flow and profile coverage. The existing audited provider machinery requires exact model hashes, expected optimized-node inventory and approved shapes; a reference policy cannot authorize the candidate graph. Supply the reviewed file explicitly via `-PolicyManifest` with `-Provider cuda-audited`. No candidate policy is generated automatically in this PR.

## Acceptance remaining

Validate tensor metadata and actual shape/element counts on TITAN, compare decoded coordinates and scores against the upstream SDK, capture CPU/CUDA numerical parity, and compare representative non-looped recordings with matching workload hashes. Run idle and contended paced workloads with resource evidence once the candidate policy is approved. Complete annotated localization, coverage, occlusion/re-entry and representative full-body camera checks before selecting a winner. The candidate is benchmark-only in this slice; no default live Pose Lab model change. Windows CI and hardware execution are pending.
