function Get-PoseComparisonHash {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Read-PoseComparisonJson {
    param([string]$Path)
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -AsHashtable -Depth 100
}

function Write-PoseComparisonJson {
    param([object]$Value, [string]$Path)
    $temporary = "$Path.$([Guid]::NewGuid().ToString('N')).tmp"
    try {
        $Value | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $temporary -Encoding utf8
        [IO.File]::Move($temporary, $Path, $false)
    } finally {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    }
}

function Assert-PoseComparison {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) {
        throw $Message
    }
}

function Read-PoseWorkload {
    param([string]$Path)
    $path = (Resolve-Path -LiteralPath $Path).Path
    $hash = Get-PoseComparisonHash $path
    $manifest = Read-PoseComparisonJson $path
    Assert-PoseComparison ($manifest.schema_version -eq 1) 'Unsupported workload schema'
    Assert-PoseComparison ($manifest.clips -is [array] -and $manifest.clips.Count -gt 0 -and
        $manifest.clips.Count -le 64) 'Workload requires 1 to 64 clips'
    $ids = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $clips = foreach ($clip in $manifest.clips) {
        Assert-PoseComparison ($clip.id -is [string] -and $clip.id -cmatch '^[a-z0-9][a-z0-9_-]{0,63}$') 'Invalid clip id'
        Assert-PoseComparison ($ids.Add($clip.id)) "Duplicate clip id: $($clip.id)"
        Assert-PoseComparison ($clip.scenario -cin @('smoke', 'standing', 'squat', 'lateral',
            'swipe', 'combined', 'occlusion', 'empty', 'reentry', 'multiple-people')) "Unknown scenario: $($clip.scenario)"
        Assert-PoseComparison ($clip.sha256 -is [string] -and $clip.sha256 -cmatch '^[0-9a-f]{64}$') "Invalid clip hash: $($clip.id)"
        Assert-PoseComparison (($clip.warmup_frames -is [long] -or $clip.warmup_frames -is [int]) -and
            $clip.warmup_frames -ge 0 -and $clip.warmup_frames -le 300) "Invalid warm-up count: $($clip.id)"
        Assert-PoseComparison (($clip.measured_frames -is [long] -or $clip.measured_frames -is [int]) -and
            $clip.measured_frames -ge 1 -and $clip.measured_frames -le 3600) "Invalid measured count: $($clip.id)"
        Assert-PoseComparison ($clip.path -is [string] -and -not [string]::IsNullOrWhiteSpace($clip.path)) "Missing clip path: $($clip.id)"
        $video = [IO.Path]::GetFullPath($clip.path, (Split-Path -Parent $path))
        Assert-PoseComparison ((Test-Path -LiteralPath $video -PathType Leaf) -and
            (Get-PoseComparisonHash $video) -ceq $clip.sha256) "Missing or changed clip: $($clip.id)"
        [pscustomobject]@{
            id = $clip.id
            scenario = $clip.scenario
            sha256 = $clip.sha256
            path = $video
            warmup_frames = $clip.warmup_frames
            measured_frames = $clip.measured_frames
        }
    }
    Assert-PoseComparison ((Get-PoseComparisonHash $path) -ceq $hash) 'Workload manifest changed while being read'
    return [pscustomobject]@{ path = $path; sha256 = $hash; clips = @($clips) }
}

function Get-PoseReportIdentity {
    param([object]$Report)
    foreach ($hash in @($Report.build.executable.sha256, $Report.model.detector_sha256, $Report.model.pose_sha256)) {
        Assert-PoseComparison ($hash -is [string] -and $hash -cmatch '^[0-9a-f]{64}$') 'Missing model or executable hash'
    }
    return [ordered]@{
        backend = $Report.model.backend
        execution_policy = $Report.model.execution_policy
        execution_provider = $Report.model.execution_provider
        detector_sha256 = $Report.model.detector_sha256
        pose_sha256 = $Report.model.pose_sha256
        executable_sha256 = $Report.build.executable.sha256
        build = $Report.build.configuration
        opencv = $Report.build.opencv
        onnxruntime = $Report.build.onnxruntime
        hardware = $Report.hardware
    }
}

function Assert-PoseWorkloadReport {
    param([object]$Report, [object]$Clip)
    Assert-PoseComparison ($Report.schema_version -eq 4 -and $Report.status.success -eq $true) "Unsuccessful or unsupported report: $($Clip.id)"
    $config = $Report.configuration
    Assert-PoseComparison ($config.mode -ceq 'deterministic' -and $config.loop_video -eq $false -and
        $null -eq $config.simulated_source_fps -and $null -eq $config.duration_seconds -and
        $config.production_profiling_enabled -eq $false -and
        $config.recreate_audited_sessions_for_experiment -eq $false) "Not an unprofiled sequential workload: $($Clip.id)"
    Assert-PoseComparison ($Report.input.kind -ceq 'video' -and $Report.input.source_sha256 -ceq $Clip.sha256) "Source hash mismatch: $($Clip.id)"
    Assert-PoseComparison ($Report.build.configuration -ceq 'Release' -and
        $Report.model.checksums_verified -eq $true -and $Report.model.provider_evidence.verified -eq $true) "Unverified build or model: $($Clip.id)"
    Assert-PoseComparison ($config.warmup_iterations -eq $Clip.warmup_frames -and
        $config.measured_iterations -eq $Clip.measured_frames -and
        $Report.results.frames_skipped -eq 0) "Workload count mismatch: $($Clip.id)"
    $index = 0
    $previousTimestamp = -1
    foreach ($phase in @('warmup', 'measured')) {
        $expected = if ($phase -eq 'warmup') { $Clip.warmup_frames } else { $Clip.measured_frames }
        $records = $Report[$phase].processed_frames
        Assert-PoseComparison ($records -is [array] -and $records.Count -eq $expected -and
            $Report[$phase].frames_processed -eq $expected) "Incomplete $phase phase: $($Clip.id)"
        foreach ($frame in $records) {
            Assert-PoseComparison ($null -ne $frame.source_frame_index -and $frame.source_frame_index -eq $index -and
                $frame.sequence -eq ($index + 1) -and $null -eq $frame.source_loop_index -and
                $null -ne $frame.source_media_timestamp_us -and
                $frame.source_media_timestamp_us -ge 0 -and
                $frame.source_media_timestamp_us -ge $previousTimestamp) "Invalid frame identity at $index in $($Clip.id)"
            $previousTimestamp = $frame.source_media_timestamp_us
            $index++
        }
    }
}

function Get-ValidatedPoseReport {
    param([string]$Benchmark, [string]$Path, [object]$Clip)
    $hash = Get-PoseComparisonHash $Path
    & $Benchmark --validate-report $Path | Out-Host
    Assert-PoseComparison ($LASTEXITCODE -eq 0) "Native report validation failed: $Path"
    $report = Read-PoseComparisonJson $Path
    Assert-PoseComparison ((Get-PoseComparisonHash $Path) -ceq $hash) "Report changed during validation: $Path"
    Assert-PoseWorkloadReport $report $Clip
    return [pscustomobject]@{ report = $report; sha256 = $hash }
}

function Get-PoseRunIdentity {
    param([object]$Report)
    return (Get-PoseReportIdentity $Report | ConvertTo-Json -Depth 20 -Compress)
}

function Get-PoseClipMetrics {
    param([object]$Report, [string]$Scenario)
    $measured = $Report.measured
    $count = $measured.frames_processed
    $availability = [ordered]@{}
    $jitter = [ordered]@{}
    $names = @('left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle')
    $width = [double]$Report.input.width
    $height = [double]$Report.input.height
    Assert-PoseComparison ($width -gt 0 -and $height -gt 0) 'Invalid image dimensions'
    $diagonal = [Math]::Sqrt($width * $width + $height * $height)
    foreach ($name in $names) {
        $present = 0
        $pairs = 0
        $squared = 0.0
        $previous = $null
        foreach ($frame in $measured.processed_frames) {
            $landmark = @($frame.landmarks | Where-Object name -ceq $name)
            Assert-PoseComparison ($landmark.Count -eq 1) "Missing or duplicate landmark: $name"
            $current = $landmark[0]
            if ($current.available) {
                $present++
                if ($null -ne $previous -and $Scenario -ceq 'standing') {
                    $dx = ($current.x - $previous.x) * $width / $diagonal
                    $dy = ($current.y - $previous.y) * $height / $diagonal
                    $squared += $dx * $dx + $dy * $dy
                    $pairs++
                }
                $previous = $current
            } else {
                $previous = $null
            }
        }
        $availability[$name] = [ordered]@{ available_frames = $present; fraction = $present / $count }
        if ($Scenario -ceq 'standing') {
            $jitter[$name] = [ordered]@{
                consecutive_pairs = $pairs
                rms_image_diagonal_fraction = if ($pairs -gt 0) { [Math]::Sqrt($squared / $pairs) } else { $null }
            }
        }
    }
    return [ordered]@{
        frames = $count
        pose_available_frames = $measured.pose_available_frames
        usable_pose_frames_at_backend_policy = $measured.usable_pose_frames
        full_body_complete_frames_at_backend_policy = $measured.full_body_complete_frames
        landmark_availability = $availability
        standing_frame_to_frame_displacement = if ($Scenario -ceq 'standing') { $jitter } else { $null }
        deterministic_throughput_fps = $measured.throughput_fps
        estimator_latency = $measured.timings.estimator_end_to_end
        projected_knee_angle_available_frames = [ordered]@{
            left = @($measured.processed_frames | Where-Object { $null -ne $_.projected_knee_angles_degrees.left }).Count
            right = @($measured.processed_frames | Where-Object { $null -ne $_.projected_knee_angles_degrees.right }).Count
        }
    }
}

function Compare-PoseClip {
    param([object]$Reference, [object]$Candidate, [object]$Clip)
    Assert-PoseWorkloadReport $Reference $Clip
    Assert-PoseWorkloadReport $Candidate $Clip
    foreach ($key in @('width', 'height', 'native_media_fps')) {
        Assert-PoseComparison ($Reference.input[$key] -eq $Candidate.input[$key]) "Input $key mismatch: $($Clip.id)"
    }
    Assert-PoseComparison ($Reference.build.opencv -ceq $Candidate.build.opencv) "Decoder library version mismatch: $($Clip.id)"
    Assert-PoseComparison ($Reference.hardware.computer_name -ceq $Candidate.hardware.computer_name) "Host mismatch: $($Clip.id)"
    foreach ($phase in @('warmup', 'measured')) {
        for ($i = 0; $i -lt $Reference[$phase].processed_frames.Count; $i++) {
            $a = $Reference[$phase].processed_frames[$i]
            $b = $Candidate[$phase].processed_frames[$i]
            Assert-PoseComparison ($a.source_frame_index -eq $b.source_frame_index -and
                $a.source_media_timestamp_us -eq $b.source_media_timestamp_us) "Input identity mismatch: $($Clip.id) $phase frame $i"
        }
    }
    return [ordered]@{
        id = $Clip.id
        scenario = $Clip.scenario
        source_sha256 = $Clip.sha256
        reference = Get-PoseClipMetrics $Reference $Clip.scenario
        candidate = Get-PoseClipMetrics $Candidate $Clip.scenario
    }
}
