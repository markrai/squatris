# Candidate CUDA profiling

This diagnostic utility inventories observed ONNX Runtime provider assignments before a candidate-specific production policy exists. It does not instantiate an IPoseEstimator, accept frames from a camera, generate a policy or authorize gameplay. The existing strict and audited execution paths remain unchanged.

## RTMW-m on TITAN

The CPU comparison successfully processed all four measured frames for each model using the same detector, executable, workload and runtime. RTMPose-m measured 31.05 FPS and 31.79 ms median estimator latency; RTMW-m measured 21.74 FPS and 42.84 ms. Both had four usable poses and three frames with complete core-body coverage under the configured tracking policy. This is smoke evidence, not sustained performance or accuracy qualification.

Build the current CUDA Release preset using the existing x64 Visual Studio developer shell, pinned vcpkg and ONNX Runtime GPU package:

```powershell
$env:VCPKG_ROOT = (Resolve-Path '.deps\vcpkg').Path
$env:SQUATRIS_ONNXRUNTIME_ROOT = (Resolve-Path '.deps\onnxruntime\gpu').Path
cmake --preset windows-cuda-release
cmake --build --preset windows-cuda-release
```

Use a new evidence directory. The previously prepared RTMW artifact is reused:

```powershell
pwsh -NoProfile -File .\scripts\profile-candidate-cuda.ps1 -Manifest models\rtmw-m-manifest.json -Model qualification\rtmw-m-artifact\rtmw-m.onnx -Width 192 -Height 256 -OutputDirectory qualification\phase0b2\rtmw-cuda-diagnostic
```

The wrapper validates the pinned model hash, sets process-local CUDA/cuDNN search paths, bounds the child process with a deadline, and records a completion manifest only after checking the report and raw trace identities. Unknown providers or no observed CUDA nodes cause a nonzero result even if diagnostic evidence was captured. The native tool takes explicit model path, lowercase SHA-256, input width and height, and a new output directory. It supports single-input RGB/BGR float32 NCHW models with three channels; other input contracts require a separate adapter. The tensor is all zeros, not a representative image.

## Evidence

- `candidate-profile.json`: declared input/output shapes, concrete returned output shapes/types/counts, actual loaded ORT module paths and hashes, device index, session settings, observed CUDA/CPU/unknown node counts, exact node names/operators/indices, recorded tensor shapes and byte fields, duration and copy-event evidence.
- `candidate_*.json`: raw ORT profile (exact path in the report).
- `diagnostic-run.json`: hash-bound completion manifest.

The profiler uses the same session option builder as production audited CUDA, with CPU fallback allowed only in this isolated diagnostic session. The session is destroyed after profiling; neither a production estimator nor a verified policy is returned. The report has verified=false, production_authorized=false, and static_assignment_complete=false. It is not accepted as native benchmark schema-4 evidence. No CPU node is automatically approved, even for shape operators.

## Review gate

Inspect the original graph using the pinned model bytes (including external data if any; the current manifests pin single ONNX files only). Establish graph operator set, IR version, node count, control flow and correspondence of observed optimized nodes. Review every CPU operation's exact name/index/operator, input/output metadata and purpose; reject substantive neural computation on CPU. Keep the existing detector policy unchanged and prepare a new candidate-specific reviewed pose inventory only after the evidence supports it. An observed execution inventory is not proof of complete static assignment. Missing or malformed profile evidence fails; byte fields are reported with the existing parser's semantics and a missing value is not proof of zero transfers.

Once a separate candidate policy is reviewed, run native audited CUDA preflight and CPU/CUDA parity on the pinned smoke workload. Then qualify sustained performance and representative tracking. The diagnostic's single synthetic run is not a throughput benchmark or an accuracy test.

Hardware-independent tests exercise diagnostic inventory parsing and rejection paths. Windows CI and actual GPU profiling must be verified locally. Do not commit raw weights, profiles or recordings.
