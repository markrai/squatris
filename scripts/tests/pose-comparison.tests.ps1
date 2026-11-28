$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot '..\pose-comparison.ps1')

function Expect-Rejected {
    param([scriptblock]$Action, [string]$Expected)
    $caught = $null
    try { & $Action | Out-Null } catch { $caught = $_.Exception.Message }
    if ($null -eq $caught -or $caught -notmatch $Expected) {
        throw "Expected rejection matching '$Expected'; actual: $caught"
    }
}

$directory = Join-Path ([IO.Path]::GetTempPath()) "squatris-workload-$([Guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Path $directory | Out-Null
try {
    $clipPath = Join-Path $directory 'test.mp4'
    [IO.File]::WriteAllText($clipPath, 'hash fixture, not video')
    $manifestPath = Join-Path $directory 'workload.json'
    $manifest = [ordered]@{
        schema_version = 1
        clips = @([ordered]@{
            id = 'stationary'
            scenario = 'standing'
            path = 'test.mp4'
            sha256 = Get-PoseComparisonHash $clipPath
            warmup_frames = 1
            measured_frames = 3
        })
    }
    Write-PoseComparisonJson $manifest $manifestPath
    $workload = Read-PoseWorkload $manifestPath
    $clip = $workload.clips[0]
    if ($clip.path -ne $clipPath) { throw 'Relative video resolution failed' }
    Expect-Rejected { Write-PoseComparisonJson $manifest $manifestPath } 'exist|already'
    $names = @('left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle')
    function New-TestFrame {
        param([int]$Index, [double]$X, [bool]$Available)
        return [ordered]@{
            sequence = $Index + 1
            source_frame_index = $Index
            source_loop_index = $null
            source_media_timestamp_us = $Index * 16667
            projected_knee_angles_degrees = @{ left = 170; right = 171 }
            landmarks = @(foreach ($name in $names) {
                [ordered]@{ name = $name; available = $Available; x = $X; y = 0.5 }
            })
        }
    }
    $hash = 'a' * 64
    $report = [ordered]@{
        schema_version = 4
        status = @{ success = $true }
        configuration = @{
            mode = 'deterministic'
            loop_video = $false
            simulated_source_fps = $null
            duration_seconds = $null
            production_profiling_enabled = $false
            recreate_audited_sessions_for_experiment = $false
            warmup_iterations = 1
            measured_iterations = 3
        }
        input = @{ kind = 'video'; source_sha256 = $clip.sha256; width = 800; height = 600; native_media_fps = 60 }
        build = @{ configuration = 'Release'; opencv = 'fixture'; onnxruntime = 'fixture'; executable = @{ sha256 = $hash } }
        hardware = @{ computer_name = 'fixture' }
        model = @{
            backend = 'fixture'
            execution_policy = 'cpu'
            execution_provider = 'CPUExecutionProvider'
            detector_sha256 = $hash
            pose_sha256 = $hash
            checksums_verified = $true
            provider_evidence = @{ verified = $true }
        }
        results = @{ frames_skipped = 0 }
        warmup = @{ frames_processed = 1; processed_frames = @((New-TestFrame 0 0.5 $true)) }
        measured = @{
            frames_processed = 3
            processed_frames = @((New-TestFrame 1 0.5 $true), (New-TestFrame 2 0.51 $true), (New-TestFrame 3 0.52 $false))
            pose_available_frames = 2
            usable_pose_frames = 2
            full_body_complete_frames = 2
            throughput_fps = 60
            timings = @{ estimator_end_to_end = @{ median_ms = 10; p95_ms = 11; p99_ms = 12 } }
        }
    }
    $candidate = $report | ConvertTo-Json -Depth 30 | ConvertFrom-Json -AsHashtable
    $result = Compare-PoseClip $report $candidate $clip
    $metric = $result.reference.standing_frame_to_frame_displacement.left_knee
    if ($metric.consecutive_pairs -ne 1 -or [Math]::Abs($metric.rms_image_diagonal_fraction - 0.008) -gt 0.000001) {
        throw 'Displacement is not aspect corrected or bridges a missing landmark'
    }
    if ($result.reference.landmark_availability.left_knee.available_frames -ne 2) {
        throw 'Landmark availability mismatch'
    }
    $candidate.measured.processed_frames[0].source_media_timestamp_us++
    Expect-Rejected { Compare-PoseClip $report $candidate $clip } 'Input identity mismatch'
    $candidate.measured.processed_frames[0].source_media_timestamp_us--
    $candidate.measured.processed_frames[0].source_frame_index = 3
    Expect-Rejected { Compare-PoseClip $report $candidate $clip } 'Invalid frame identity'
    $candidate.measured.processed_frames[0].source_frame_index = 1
    $candidate.configuration.loop_video = $true
    Expect-Rejected { Compare-PoseClip $report $candidate $clip } 'Not an unprofiled sequential workload'
    $candidate.configuration.loop_video = $false
    $candidate.status.success = $false
    Expect-Rejected { Compare-PoseClip $report $candidate $clip } 'Unsuccessful'
    $candidate.status.success = $true
    $candidate.input.source_sha256 = 'b' * 64
    Expect-Rejected { Compare-PoseClip $report $candidate $clip } 'Source hash mismatch'
    $duplicatePath = Join-Path $directory 'duplicate.json'
    $manifest.clips = @($manifest.clips[0], $manifest.clips[0])
    Write-PoseComparisonJson $manifest $duplicatePath
    Expect-Rejected { Read-PoseWorkload $duplicatePath } 'Duplicate clip id'
    [IO.File]::AppendAllText($clipPath, 'changed')
    Expect-Rejected { Read-PoseWorkload $manifestPath } 'Missing or changed clip'
} finally {
    Remove-Item -LiteralPath $directory -Recurse -Force
}
Write-Host 'PASS: pose comparison contract fixtures'
