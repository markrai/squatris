param(
    [Parameter(Mandatory)][string]$Workload,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [ValidateSet('cpu', 'cuda-audited', 'cuda-strict')][string]$Provider = 'cuda-audited',
    [ValidateSet('rtmpose', 'rtmw')][string]$Backend = 'rtmpose',
    [string]$PoseModel,
    [string]$ModelManifest,
    [string]$PolicyManifest,
    [string]$Benchmark,
    [ValidateRange(30, 3600)][int]$DeadlineSeconds = 600
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
Set-StrictMode -Version Latest
. "$PSScriptRoot\pose-comparison.ps1"

$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if ($Backend -eq 'rtmw' -and (-not $PoseModel -or -not $ModelManifest -or
    ($Provider -eq 'cuda-audited' -and -not $PolicyManifest))) {
    throw 'RTMW requires explicit -PoseModel, -ModelManifest and, for audited CUDA, -PolicyManifest'
}
$posePath = if ($PoseModel) { (Resolve-Path -LiteralPath $PoseModel).Path } else { Join-Path $root 'models\pose.onnx' }
$manifestPath = if ($ModelManifest) { (Resolve-Path -LiteralPath $ModelManifest).Path } else { Join-Path $root 'models\manifest.json' }
$policyPath = if ($PolicyManifest) { (Resolve-Path -LiteralPath $PolicyManifest).Path } else { Join-Path $root 'config\rtmpose-cuda-policy.json' }
$work = Read-PoseWorkload $Workload
$output = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $output) { throw "Output directory already exists: $output" }
$preset = if ($Provider -eq 'cpu') { 'windows-onnx-release' } else { 'windows-cuda-release' }
$benchmark = if ($Benchmark) { (Resolve-Path -LiteralPath $Benchmark).Path } else { Join-Path $root "out\build\$preset\tools\benchmark\SquatrisPoseBenchmark.exe" }
if (-not (Test-Path -LiteralPath $benchmark -PathType Leaf)) { throw "Build $preset first" }
if ($Provider -ne 'cpu') {
    $cudaBin = if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH 'bin' } else { '' }
    $env:PATH = @((Join-Path $root '.deps\cudnn\9.1.0.70\bin'), $cudaBin, $env:PATH) -join ';'
}
$executableHash = Get-PoseComparisonHash $benchmark
New-Item -ItemType Directory -Path $output | Out-Null
$identity = $null
$reports = [Collections.Generic.List[object]]::new()
foreach ($clip in $work.clips) {
    $reportPath = Join-Path $output "$($clip.id).json"
    Assert-PoseComparison ((Get-PoseComparisonHash $clip.path) -ceq $clip.sha256) "Clip changed before run: $($clip.id)"
    Write-Host "=== $($clip.id) ($($clip.scenario)) ==="
    $arguments = @('--mode', 'deterministic', '--video', $clip.path,
        '--backend', $Backend, '--provider', $Provider,
        '--warmup', "$($clip.warmup_frames)", '--iterations', "$($clip.measured_frames)",
        '--loop-video', 'false', '--detector', (Join-Path $root 'models\detector.onnx'),
        '--pose', $posePath,
        '--manifest', $manifestPath,
        '--policy-manifest', $policyPath,
        '--deadline-seconds', "$DeadlineSeconds", '--output', $reportPath)
    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $benchmark
    $startInfo.WorkingDirectory = $root
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    foreach ($argument in $arguments) { $startInfo.ArgumentList.Add($argument) }
    $process = [Diagnostics.Process]::Start($startInfo)
    $stdoutFile = [IO.File]::Create((Join-Path $output "$($clip.id).stdout.log"))
    $stderrFile = [IO.File]::Create((Join-Path $output "$($clip.id).stderr.log"))
    try {
        $stdout = $process.StandardOutput.BaseStream.CopyToAsync($stdoutFile)
        $stderr = $process.StandardError.BaseStream.CopyToAsync($stderrFile)
        if (-not $process.WaitForExit(($DeadlineSeconds + 60) * 1000)) {
            $process.Kill($true)
            if (-not $process.WaitForExit(10000)) { throw "Could not terminate benchmark: $($clip.id)" }
            throw "External benchmark deadline exceeded: $($clip.id)"
        }
        if (-not [Threading.Tasks.Task]::WaitAll([Threading.Tasks.Task[]]@($stdout, $stderr), 10000)) {
            throw "Benchmark log drain timed out: $($clip.id)"
        }
        Assert-PoseComparison ($process.ExitCode -eq 0) "Benchmark failed: $($clip.id); inspect logs in $output"
    } finally {
        if (-not $process.HasExited) { $process.Kill($true) }
        $stdoutFile.Dispose()
        $stderrFile.Dispose()
        $process.Dispose()
    }
    $validated = Get-ValidatedPoseReport $benchmark $reportPath $clip
    $report = $validated.report
    $expectedBackend = if ($Backend -eq 'rtmw') { 'rtmw-m-coco133-256x192' } else { 'rtmpose-m-halpe26-256x192' }
    Assert-PoseComparison ($report.build.executable.sha256 -ceq $executableHash -and
        $report.model.backend -ceq $expectedBackend -and
        $report.model.execution_policy -ceq $Provider) "Unexpected model execution identity: $($clip.id)"
    $currentIdentity = Get-PoseRunIdentity $report
    if ($null -eq $identity) { $identity = $currentIdentity }
    Assert-PoseComparison ($identity -ceq $currentIdentity) "Model/build/host identity changed: $($clip.id)"
    Assert-PoseComparison ((Get-PoseComparisonHash $clip.path) -ceq $clip.sha256) "Clip changed during run: $($clip.id)"
    $reports.Add([ordered]@{ id = $clip.id; report_sha256 = $validated.sha256 })
}
Assert-PoseComparison ((Get-PoseComparisonHash $work.path) -ceq $work.sha256 -and
    (Get-PoseComparisonHash $benchmark) -ceq $executableHash) 'Workload or executable changed during run'
Write-PoseComparisonJson ([ordered]@{
    schema_version = 1
    workload_sha256 = $work.sha256
    model_identity = ($identity | ConvertFrom-Json -AsHashtable)
    reports = $reports.ToArray()
}) (Join-Path $output 'run.json')
Write-Host "Workload complete: $output"
