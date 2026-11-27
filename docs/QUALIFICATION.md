# Qualification

## Status classes

- Automated-test verified: deterministic behavior proven without physical hardware.
- Hardware verified: observed on the named machine with recorded commands and output.
- Unverified: implemented but not exercised on the required hardware.
- Blocked: a missing external dependency prevents the check.

## VENOM status

The temporary Intel implementation PC verifies the strict hardware-independent builds and tests, both benchmark modes, the ONNX-enabled CPU build, model-contract validation, manifest checksums, and deterministic CPU RTMPose inference. It does not verify NVIDIA loading, `CUDAExecutionProvider`, TITAN performance, or physical camera behavior. Those items remain unverified rather than failed.

## Transfer from VENOM to TITAN

All Phase 0A.1 changes must remain uncommitted for review. Transfer the working tree and its existing Git metadata without copying build outputs, downloaded dependencies, weights, or prior qualification evidence.

On VENOM, choose an empty transfer directory and copy the repository:

```powershell
$source = 'C:\dev\project\squatris'
$transfer = 'D:\transfer\squatris-phase0a1'
if (Test-Path -LiteralPath $transfer) { throw "Transfer directory must not already exist: $transfer" }
robocopy $source $transfer /E /XD out .deps qualification /XF detector.onnx pose.onnx
if ($LASTEXITCODE -gt 7) { throw "robocopy failed with exit code $LASTEXITCODE" }
Push-Location $source
$files = @(git ls-files --cached --others --exclude-standard)
$sourceManifest = @($files | ForEach-Object {
    [ordered]@{ path = $_; sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_).Hash }
})
Pop-Location
$sourceManifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath 'D:\transfer\squatris-phase0a1-files.json' -Encoding utf8
```

Move both the copied directory and `squatris-phase0a1-files.json` to TITAN. On TITAN, place the directory at `C:\dev\project\squatris`, then verify every reviewable file before running anything:

```powershell
$repository = 'C:\dev\project\squatris'
$expected = Get-Content -LiteralPath 'C:\transfer\squatris-phase0a1-files.json' -Raw | ConvertFrom-Json
Push-Location $repository
$actual = @($expected | ForEach-Object {
    [ordered]@{ path = $_.path; sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.path).Hash }
})
$difference = Compare-Object ($expected | ConvertTo-Json -Compress) ($actual | ConvertTo-Json -Compress)
if ($difference) { throw 'Transferred source hashes do not match VENOM' }
git status --short
Pop-Location
```

The status output must show the Phase 0A.1 working-tree changes. Do not commit them before review.

## Reproducible TITAN procedure

Prerequisites are an existing Visual Studio C++ environment, Git, CMake, Ninja, an NVIDIA driver compatible with CUDA 12, cuDNN 9 discoverable by Windows, a prerecorded qualification video, and a physical camera for the manual observations. The procedure does not install drivers, CUDA, cuDNN, or change global system configuration.

From a PowerShell development shell on TITAN:

```powershell
Set-Location C:\dev\project\squatris
.\scripts\qualify-titan.ps1 `
    -Video 'C:\qualification-inputs\squatris-full-body.mp4' `
    -OutputDirectory 'C:\qualification-results\squatris-phase0a1' `
    -CameraIndex 0 `
    -WarmupIterations 100 `
    -MeasuredIterations 2000
if ($LASTEXITCODE -ne 0) { throw 'Automated TITAN qualification failed' }
```

The script records OS, CPU, GPU, driver, camera, tool, CUDA, and cuDNN inventory; verifies pinned vcpkg, ONNX Runtime GPU, and model artifacts; performs a strict release build; runs hardware-independent tests; creates an actual strict CUDA session; verifies provider assignment and model hashes; runs and validates a deterministic prerecorded benchmark; probes camera indices and negotiated modes; and writes `qualification-report.json` plus step logs. Its dependency changes are confined to the repository `.deps`, `models`, `out`, and selected report directories.

The script requires `nvidia-smi` to advertise CUDA 12 or newer driver compatibility. The definitive cuDNN 9 compatibility check is successful CUDA session creation and inference with CPU fallback disabled.

## Physical camera participation

Automated enumeration cannot establish visual correctness. After the script succeeds, a person must run the exact command stored in `qualification-report.json`, normally:

```powershell
.\out\build\windows-cuda-release\apps\pose-lab\SquatrisPoseLab.exe --camera 0 --backend rtmpose --provider cuda --detector models\detector.onnx --pose models\pose.onnx
```

Record these observations separately from automated results:

1. Full-body framing and actual negotiated live capture rate.
2. Anatomical left and right under mirrored and unmirrored preview.
3. Projected 2D knee-angle behavior through standing and squatting.
4. Occlusion, low-confidence behavior, and recovery.
5. Physical camera disconnect and shutdown behavior.
6. Sustained responsiveness and whether latency accumulates.

Frame-synchronized mode is the default. Press `F` to compare low-latency preview behavior. In low-latency mode a mismatched skeleton must be suppressed, and diagnostics must report the preview and pose sequences as mismatched.

## Acceptance checklist

1. `qualification-report.json` has `passed: true` and every automated step passed.
2. CUDA benchmark reports contain `CUDAExecutionProvider`, `strict_provider_assignment: true`, and `checksums_verified: true`.
3. Report validation succeeds and deterministic `frames_skipped` is zero.
4. Prerecorded frame identifiers and media timestamps match the selected source order.
5. Camera enumeration records at least one device and its probed negotiated modes.
6. All physical observations are attached with the human participant, time, device, and outcome.

The future 60 updates/s, 80 ms response, recognition, and duplicate-event targets cannot be marked achieved from mock tests or a short visual smoke test. Player identity and gesture behavior are outside Phase 0A.1.
