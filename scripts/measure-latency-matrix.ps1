param(
    [string]$Video = 'qualification\inputs\mmpose-demo.mp4',
    [string]$OutputDirectory = 'qualification\phase0a4\latency-matrix',
    [int]$ShortSeconds = 20,
    [int]$SustainedSeconds = 120,
    [switch]$MonitoredOnly
)

$ErrorActionPreference = 'Stop'
if ($ShortSeconds -lt 20 -or $SustainedSeconds -lt 120) {
    throw 'Short and sustained experiments require at least 20 and 120 seconds respectively'
}

$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$videoPath = [IO.Path]::GetFullPath((Join-Path $repositoryRoot $Video))
$outputRoot = [IO.Path]::GetFullPath((Join-Path $repositoryRoot $OutputDirectory))
$benchmark = Join-Path $repositoryRoot 'out\build\windows-cuda-release\tools\benchmark\SquatrisPoseBenchmark.exe'
$modelBenchmark = Join-Path $repositoryRoot 'out\build\windows-cuda-release\tools\model-benchmark\SquatrisModelBenchmark.exe'
$detector = Join-Path $repositoryRoot 'models\detector.onnx'
$pose = Join-Path $repositoryRoot 'models\pose.onnx'
$manifest = Join-Path $repositoryRoot 'models\manifest.json'
$policy = Join-Path $repositoryRoot 'config\rtmpose-cuda-policy.json'
foreach ($path in @($videoPath, $benchmark, $modelBenchmark, $detector, $pose, $manifest, $policy)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required experiment artifact is missing: $path"
    }
}
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$cudnnBin = Join-Path $repositoryRoot '.deps\cudnn\9.1.0.70\bin'
$cudaBin = if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH 'bin' } else { $null }
$env:PATH = @($cudnnBin, $cudaBin, $env:PATH) -join ';'

function Quote-Argument {
    param([string]$Value)
    if ($Value -notmatch '[\s"]') { return $Value }
    return '"' + $Value.Replace('"', '\"') + '"'
}

function Write-JsonAtomically {
    param([object]$Value, [string]$Path, [int]$Depth = 12)
    $temporary = "$Path.$([Guid]::NewGuid().ToString('N')).tmp"
    try {
        $Value | ConvertTo-Json -Depth $Depth | Set-Content -LiteralPath $temporary -Encoding utf8
        [IO.File]::Move($temporary, $Path, $true)
    } finally {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-NvidiaQuery {
    param([string[]]$Arguments, [string]$Prefix)
    $stdoutPath = Join-Path $outputRoot ".$Prefix-$([Guid]::NewGuid().ToString('N')).stdout"
    $stderrPath = Join-Path $outputRoot ".$Prefix-$([Guid]::NewGuid().ToString('N')).stderr"
    try {
        $process = Start-Process -FilePath 'nvidia-smi' -ArgumentList (($Arguments | ForEach-Object {
            Quote-Argument $_
        }) -join ' ') -PassThru -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath -WindowStyle Hidden
        if (-not $process.WaitForExit(2000)) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            [void]$process.WaitForExit(2000)
            throw 'nvidia-smi exceeded its two-second sampling deadline'
        }
        if ($process.ExitCode -ne 0) {
            $errorText = if (Test-Path -LiteralPath $stderrPath) {
                Get-Content -LiteralPath $stderrPath -Raw
            } else { '' }
            throw "nvidia-smi failed with exit code $($process.ExitCode): $errorText"
        }
        return @(Get-Content -LiteralPath $stdoutPath)
    } finally {
        Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
    }
}

function Convert-Number {
    param([string]$Value)
    $number = 0.0
    if ([double]::TryParse($Value.Trim(), [Globalization.NumberStyles]::Float,
            [Globalization.CultureInfo]::InvariantCulture, [ref]$number)) {
        return $number
    }
    return $null
}

function Get-ResourceSample {
    param([Diagnostics.Process]$Process, [Diagnostics.Stopwatch]$Watch,
        [double]$PreviousCpuMilliseconds, [double]$PreviousElapsedMilliseconds)
    $Process.Refresh()
    $cpuMilliseconds = $Process.TotalProcessorTime.TotalMilliseconds
    $cpuWindow = $Watch.Elapsed.TotalMilliseconds - $PreviousElapsedMilliseconds
    $cpuPercent = if ($cpuWindow -gt 0) {
        100.0 * ($cpuMilliseconds - $PreviousCpuMilliseconds) /
            ($cpuWindow * [Environment]::ProcessorCount)
    } else { 0.0 }
    $deviceLines = @(Invoke-NvidiaQuery @(
        '--query-gpu=clocks.current.graphics,clocks.current.sm,clocks.current.memory,temperature.gpu,utilization.gpu,pstate,power.draw,memory.used',
        '--format=csv,noheader,nounits'
    ) 'device')
    if ($deviceLines.Count -eq 0) { throw 'nvidia-smi returned no device sample' }
    $device = $deviceLines[0].Split(',') | ForEach-Object { $_.Trim() }
    if ($device.Count -ne 8) { throw 'nvidia-smi returned an unexpected device sample' }
    $processGpuMemory = $null
    $processGpuMemoryAvailable = $false
    try {
        $computeLines = @(Invoke-NvidiaQuery @(
            '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'
        ) 'compute')
        foreach ($line in $computeLines) {
            $fields = $line.Split(',') | ForEach-Object { $_.Trim() }
            if ($fields.Count -eq 2 -and $fields[0] -eq "$($Process.Id)") {
                $processGpuMemory = Convert-Number $fields[1]
                $processGpuMemoryAvailable = $null -ne $processGpuMemory
                break
            }
        }
    } catch {
        $processGpuMemoryAvailable = $false
    }
    return [ordered]@{
        sample = [ordered]@{
            elapsed_seconds = $Watch.Elapsed.TotalSeconds
            process_cpu_percent = $cpuPercent
            process_thread_count = $Process.Threads.Count
            working_set_bytes = $Process.WorkingSet64
            private_memory_bytes = $Process.PrivateMemorySize64
            gpu_graphics_clock_mhz = Convert-Number $device[0]
            gpu_sm_clock_mhz = Convert-Number $device[1]
            gpu_memory_clock_mhz = Convert-Number $device[2]
            gpu_temperature_c = Convert-Number $device[3]
            gpu_utilization_percent = Convert-Number $device[4]
            gpu_power_state = $device[5]
            gpu_power_draw_watts = Convert-Number $device[6]
            device_gpu_memory_used_mib = Convert-Number $device[7]
            process_gpu_memory_available = $processGpuMemoryAvailable
            process_gpu_memory_used_mib = $processGpuMemory
        }
        cpu_milliseconds = $cpuMilliseconds
        elapsed_milliseconds = $Watch.Elapsed.TotalMilliseconds
    }
}

function Invoke-ExperimentProcess {
    param([string]$Name, [string]$Program, [string[]]$Arguments,
        [bool]$MonitorResources, [int]$TimeoutSeconds)
    $stdoutPath = Join-Path $outputRoot "$Name.stdout.log"
    $stderrPath = Join-Path $outputRoot "$Name.stderr.log"
    $resourcePath = Join-Path $outputRoot "$Name.resources.json"
    $argumentLine = ($Arguments | ForEach-Object { Quote-Argument $_ }) -join ' '
    $process = Start-Process -FilePath $Program -ArgumentList $argumentLine -PassThru `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -WindowStyle Hidden
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $samples = [Collections.Generic.List[object]]::new()
    $errors = [Collections.Generic.List[string]]::new()
    $lastSample = -1000.0
    $cpuMilliseconds = 0.0
    $cpuElapsed = 0.0
    while (-not $process.WaitForExit(250)) {
        if ($watch.Elapsed.TotalSeconds -ge $TimeoutSeconds) {
            $killer = Start-Process -FilePath 'taskkill.exe' -ArgumentList @(
                '/PID', "$($process.Id)", '/T', '/F'
            ) -PassThru -WindowStyle Hidden
            if (-not $killer.WaitForExit(5000)) {
                Stop-Process -Id $killer.Id -Force -ErrorAction SilentlyContinue
            }
            [void]$process.WaitForExit(5000)
            throw "$Name exceeded its $TimeoutSeconds second deadline"
        }
        if ($MonitorResources -and $watch.Elapsed.TotalMilliseconds - $lastSample -ge 1000.0) {
            $lastSample = $watch.Elapsed.TotalMilliseconds
            try {
                $resource = Get-ResourceSample $process $watch $cpuMilliseconds $cpuElapsed
                $samples.Add($resource.sample)
                $cpuMilliseconds = $resource.cpu_milliseconds
                $cpuElapsed = $resource.elapsed_milliseconds
            } catch {
                $errors.Add($_.Exception.Message)
            }
        }
    }
    $watch.Stop()
    [ordered]@{
        schema_version = 1
        monitoring_enabled = $MonitorResources
        elapsed_seconds = $watch.Elapsed.TotalSeconds
        process_id = $process.Id
        exit_code = $process.ExitCode
        monitoring_errors = $errors
        samples = $samples
    } | ForEach-Object { Write-JsonAtomically $_ $resourcePath 10 }
    if ($process.ExitCode -ne 0) {
        $errorText = if (Test-Path -LiteralPath $stderrPath) {
            Get-Content -LiteralPath $stderrPath -Raw
        } else { '' }
        throw "$Name failed with exit code $($process.ExitCode): $errorText"
    }
    return $resourcePath
}

$summaries = [Collections.Generic.List[object]]::new()
function Invoke-FullPipelineCase {
    param([string]$Name, [string]$Mode, [int]$Warmup, [int]$Iterations,
        [int]$DurationSeconds, [double]$Rate, [bool]$Monitor)
    $reportPath = Join-Path $outputRoot "$Name.json"
    $arguments = @(
        '--mode', $Mode, '--video', $videoPath, '--backend', 'rtmpose',
        '--provider', 'cuda-audited', '--warmup', "$Warmup", '--iterations', "$Iterations",
        '--loop-video', 'true', '--detector', $detector, '--pose', $pose,
        '--manifest', $manifest, '--policy-manifest', $policy,
        '--deadline-seconds', "$($DurationSeconds + 600)", '--output', $reportPath
    )
    if ($DurationSeconds -gt 0) { $arguments += @('--duration-seconds', "$DurationSeconds") }
    if ($Rate -gt 0) { $arguments += @('--simulated-source-fps', "$Rate") }
    $resources = Invoke-ExperimentProcess $Name $benchmark $arguments $Monitor ($DurationSeconds + 660)
    $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
    if (-not $report.status.success) { throw "$Name produced an unsuccessful report" }
    $summaries.Add([ordered]@{
        name = $Name
        workload = 'complete-image-to-pose'
        mode = $Mode
        monitoring_enabled = $Monitor
        simulated_source_fps = if ($Rate -gt 0) { $Rate } else { $null }
        measured_seconds = $report.measured.elapsed_seconds
        frames = $report.measured.frames_processed
        useful_pose_fps = $report.measured.useful_pose_throughput_fps
        skipped_frames = $report.results.frames_skipped
        estimator_median_ms = $report.measured.timings.estimator_end_to_end.median_ms
        estimator_p95_ms = $report.measured.timings.estimator_end_to_end.p95_ms
        estimator_p99_ms = $report.measured.timings.estimator_end_to_end.p99_ms
        detector_preprocessing_median_ms = $report.measured.timings.detector_preprocessing.median_ms
        detector_inference_median_ms = $report.measured.timings.detector_inference.median_ms
        detector_postprocessing_median_ms = $report.measured.timings.detector_postprocessing.median_ms
        pose_preprocessing_median_ms = $report.measured.timings.pose_preprocessing.median_ms
        pose_inference_median_ms = $report.measured.timings.pose_inference.median_ms
        pose_postprocessing_median_ms = $report.measured.timings.pose_postprocessing.median_ms
        first_inference_ms = $report.runtime.first_inference_ms
        report = $reportPath
        resources = $resources
    })
}

function Invoke-ModelOnlyCase {
    param([string]$Name, [bool]$Monitor)
    $reportPath = Join-Path $outputRoot "$Name.json"
    $arguments = @(
        '--detector', $detector, '--pose', $pose, '--policy-manifest', $policy,
        '--warmup', '100', '--iterations', '2000', '--output', $reportPath
    )
    $resources = Invoke-ExperimentProcess $Name $modelBenchmark $arguments $Monitor 900
    $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
    $summaries.Add([ordered]@{
        name = $Name
        workload = $report.workload
        monitoring_enabled = $Monitor
        measured_seconds = $report.results.elapsed_seconds
        frames = $report.configuration.iterations
        useful_pose_fps = $null
        skipped_frames = $null
        estimator_median_ms = $report.results.combined_model_execution.median_ms
        estimator_p95_ms = $report.results.combined_model_execution.p95_ms
        estimator_p99_ms = $report.results.combined_model_execution.p99_ms
        detector_inference_median_ms = $report.results.detector_inference.median_ms
        pose_inference_median_ms = $report.results.pose_inference.median_ms
        report = $reportPath
        resources = $resources
    })
}

if (-not $MonitoredOnly) {
    Invoke-FullPipelineCase 'continuous-cold' 'deterministic' 0 200 0 0 $false
    Invoke-FullPipelineCase 'continuous-warmed' 'deterministic' 100 1000 0 0 $false
}
foreach ($rate in @(5, 15, 30, 60)) {
    if (-not $MonitoredOnly) {
        Invoke-FullPipelineCase "paced-$rate-monitor-off" 'realtime' 20 1 $ShortSeconds $rate $false
    }
    Invoke-FullPipelineCase "paced-$rate-monitor-on" 'realtime' 20 1 $ShortSeconds $rate $true
}
Invoke-FullPipelineCase 'paced-60-sustained-monitor-on' 'realtime' 100 1 $SustainedSeconds 60 $true
if (-not $MonitoredOnly) {
    Invoke-ModelOnlyCase 'synthetic-tensors-monitor-off' $false
}
Invoke-ModelOnlyCase 'synthetic-tensors-monitor-on' $true

$matrix = [ordered]@{
    schema_version = 1
    generated_utc = [DateTime]::UtcNow.ToString('o')
    host = $env:COMPUTERNAME
    source = [ordered]@{
        path = $videoPath
        sha256 = (Get-FileHash -LiteralPath $videoPath -Algorithm SHA256).Hash.ToLowerInvariant()
        simulated_pacing_is_not_physical_observation = $true
    }
    experiments = $summaries
}
$matrixPath = Join-Path $outputRoot 'matrix-summary.json'
Write-JsonAtomically $matrix $matrixPath 10
Write-Output $matrixPath
