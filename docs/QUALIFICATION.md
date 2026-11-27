# Qualification

## Status classes

- Automated-test verified: deterministic behavior proven without physical hardware.
- Hardware verified: observed on the named machine with recorded commands and output.
- Unverified: implemented but not exercised on the required hardware.
- Blocked: a missing external dependency prevents the check.

## TITAN checklist

Record the following before changing any driver or GPU dependency:

```powershell
Get-ComputerInfo | Select-Object WindowsProductName,WindowsVersion,OsBuildNumber,OsArchitecture
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors
Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion
nvidia-smi
nvcc --version
Get-PnpDevice -Class Camera,Image | Select-Object Status,FriendlyName,InstanceId
cmake --version
ninja --version
```

Then verify:

1. ONNX Runtime reports `CUDAExecutionProvider` in Pose Lab and benchmark output.
2. An invalid/missing CUDA setup fails instead of reporting CPU.
3. `--list-cameras` reports each usable device and probed negotiated formats.
4. The chosen camera actually delivers the requested resolution and measured rate.
5. Preview mirroring does not swap anatomical arm labels.
6. Low-confidence or missing points do not create skeleton connections.
7. Camera disconnect produces an explicit diagnostic and shutdown remains clean.
8. Prerecorded CUDA benchmark completes with model hashes matching the manifest.
9. Benchmark JSON records median, p95, p99, skips, errors, and provider.
10. A sustained Pose Lab session remains responsive and does not accumulate latency.

The opt-in CTest qualification is registered with the `hardware-required` label and invoked with `ctest --preset windows-hardware`. It does not replace the live-camera observations above.

## Current evidence

The temporary Intel implementation PC verified Media Foundation enumeration and sample delivery for one camera, FFmpeg prerecorded input, the complete CPU RTMPose pipeline, and explicit rejection of an unavailable CUDA provider. It did not verify the TITAN inventory, `CUDAExecutionProvider`, RTX performance, preview anatomy with a person, physical knee-angle quality, disconnect behavior on a real device, or sustained-session responsiveness. Those items remain unverified rather than failed.

## Physical pose checks

Phase 0A checks only visibility and measurement quality: full-body framing, left/right correctness, confidence under ordinary motion, knee angles through standing and squatting, occlusion recovery, and tracking loss behavior. Gesture acceptance, duplicate hard-drop prevention, and intentional/unintentional activation rates belong to later phases.

Never mark the future 60 updates/s, 80 ms response, 99% intentional recognition, or duplicate-event targets as achieved from mock tests or a short visual smoke test.
