# Phase 0B.2: RTMW-m candidate integration

RTMW-m at 256x192 is the initial candidate. OpenMMLab publishes a 133-keypoint Cocktail14 model at this resolution. The evaluation keeps the pinned YOLOX-tiny detector, crop, resolution, canonical skeleton, timestamp semantics and confidence gating unchanged. RTMW maps COCO-WholeBody body landmarks and big toes/heels to the same canonical names. Face and finger landmarks are not exposed. The existing HALPE-26 reference remains the default in Pose Lab and the benchmark.

References: [RTMW release](https://github.com/open-mmlab/mmpose/releases), [RTMlib model zoo](https://github.com/Tau-J/rtmlib), [COCO-WholeBody metadata](https://github.com/open-mmlab/mmpose/blob/main/configs/_base_/datasets/coco_wholebody.py). The candidate uses the shared RGB affine normalization and SimCC decoder provisionally; verify preprocessing and numerical output against the SDK export metadata and upstream reference on the actual artifact before qualification. Model weights and dataset terms require separate review.

## Local artifact pin

The upstream export is [RTMW-m ONNX SDK](https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-m-s_simcc-cocktail14_270e-256x192_20231122.zip). The binary could not be independently retrieved and hashed during implementation, so there is intentionally no fabricated checked-in candidate checksum. Obtain and inspect the official archive locally, retain its hash and SDK metadata, and confirm that its ONNX contract matches the candidate. Supply a separately reviewed expected ONNX SHA-256 explicitly; computing a digest from an untrusted download alone establishes reproducibility, not authenticity. Do not substitute a random re-export. The preparation script never replaces existing files or modifies the reference manifest.

```powershell
pwsh -NoProfile -File .\scripts\prepare-rtmw-candidate.ps1 -PoseFile 'C:\path\to\end2end.onnx' -ExpectedOnnxSha256 '<reviewed SHA-256>'
```

## CPU comparison smoke

Build `windows-onnx-release` with the pinned CPU runtime and run both backends on the same smoke workload:

```powershell
pwsh -NoProfile -File .\scripts\run-pose-workload.ps1 -Workload config\pose-workload-smoke.json -OutputDirectory qualification\phase0b2\reference-cpu -Provider cpu
pwsh -NoProfile -File .\scripts\run-pose-workload.ps1 -Workload config\pose-workload-smoke.json -OutputDirectory qualification\phase0b2\rtmw-cpu -Backend rtmw -Provider cpu -PoseModel qualification\rtmw-m-artifact\rtmw-m.onnx -ModelManifest qualification\rtmw-m-artifact\manifest.json
pwsh -NoProfile -File .\scripts\compare-pose-workloads.ps1 -Workload config\pose-workload-smoke.json -ReferenceDirectory qualification\phase0b2\reference-cpu -CandidateDirectory qualification\phase0b2\rtmw-cpu -Output qualification\phase0b2\comparison-cpu.json
```

The comparator currently requires the CUDA Release validator executable even for CPU reports; build it from the same revision. CPU is an intentional diagnostic, not silent fallback or a GPU-performance result. The same five-frame smoke is not tracking-accuracy evidence.

## GPU gate

RTMW strict CUDA is expected to fail if the common detector requires CPU shape operations. Do not bypass the existing audit. For audited CUDA, generate a separate candidate-specific policy after inspecting the exact ONNX graph and recording node assignments on TITAN with the pinned runtime. Preserve the existing detector inventory; justify every candidate CPU node by exact name, index, operator, shapes and purpose. Reject any substantive CPU neural execution. Validate actual graph control flow and profile coverage. The existing audited provider machinery requires exact model hashes, expected optimized-node inventory and approved shapes; a reference policy cannot authorize the candidate graph. Supply the reviewed file explicitly via `-PolicyManifest` with `-Provider cuda-audited`. No candidate policy is generated automatically in this PR.

## Acceptance remaining

Validate tensor metadata and actual shape/element counts on TITAN, compare decoded coordinates and scores against the upstream SDK, capture CPU/CUDA numerical parity, and compare representative non-looped recordings with matching workload hashes. Run idle and contended paced workloads with resource evidence once the candidate policy is approved. Complete annotated localization, coverage, occlusion/re-entry and representative full-body camera checks before selecting a winner. The candidate is benchmark-only in this slice; no default live Pose Lab model change. Windows CI and hardware execution are pending.
