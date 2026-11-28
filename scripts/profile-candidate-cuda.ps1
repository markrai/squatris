param(
    [Parameter(Mandatory)][string]$Manifest,
    [Parameter(Mandatory)][string]$Model,
    [ValidateSet('pose_estimator', 'person_detector')][string]$Role = 'pose_estimator',
    [Parameter(Mandatory)][ValidateRange(1, 2048)][int]$Width,
    [Parameter(Mandatory)][ValidateRange(1, 2048)][int]$Height,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [ValidateRange(10, 600)][int]$DeadlineSeconds = 120
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. "$PSScriptRoot\pose-comparison.ps1"

$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$manifestPath = (Resolve-Path -LiteralPath $Manifest).Path
$modelPath = (Resolve-Path -LiteralPath $Model).Path
$manifestHash = Get-PoseComparisonHash $manifestPath
$pin = Read-PoseComparisonJson $manifestPath
Assert-PoseComparison ($pin.schema_version -eq 1) 'Unsupported model manifest schema'
$artifacts = @($pin.artifacts | Where-Object role -ceq $Role)
Assert-PoseComparison ($artifacts.Count -eq 1) 'Expected exactly one matching artifact in manifest'
$expectedHash = $artifacts[0].onnx_sha256
Assert-PoseComparison ($expectedHash -cmatch '^[0-9a-f]{64}$') 'Invalid pinned model checksum'
Assert-PoseComparison ((Get-PoseComparisonHash $modelPath) -ceq $expectedHash) 'Model checksum mismatch'
$output = [IO.Path]::GetFullPath($OutputDirectory)
Assert-PoseComparison (-not (Test-Path -LiteralPath $output)) 'Output directory must not exist'
$profiler = Join-Path $root 'out\build\windows-cuda-release\tools\candidate-profiler\SquatrisCandidateProfiler.exe'
Assert-PoseComparison (Test-Path -LiteralPath $profiler -PathType Leaf) 'Build windows-cuda-release first'
$profilerHash = Get-PoseComparisonHash $profiler

$cudaBin = if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH 'bin' } else { '' }
$env:PATH = @((Join-Path $root '.deps\cudnn\9.1.0.70\bin'), $cudaBin, $env:PATH) -join ';'
$startInfo = [Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $profiler
$startInfo.WorkingDirectory = $root
$startInfo.UseShellExecute = $false
foreach ($argument in @('--model', $modelPath, '--sha256', $expectedHash,
    '--width', "$Width", '--height', "$Height", '--output-directory', $output)) {
    $startInfo.ArgumentList.Add($argument)
}
$process = [Diagnostics.Process]::Start($startInfo)
try {
    if (-not $process.WaitForExit($DeadlineSeconds * 1000)) {
        $process.Kill($true)
        if (-not $process.WaitForExit(10000)) { throw 'Could not terminate diagnostic profiler' }
        throw 'Candidate profiler exceeded external deadline'
    }
    Assert-PoseComparison ($process.ExitCode -eq 0) 'Candidate profiler failed'
} finally {
    $process.Dispose()
}
$reportPath = Join-Path $output 'candidate-profile.json'
$report = Read-PoseComparisonJson $reportPath
Assert-PoseComparison ($report.kind -ceq 'candidate-cuda-diagnostic' -and
    $report.production_authorized -eq $false -and $report.verified -eq $false -and
    $report.model.sha256 -ceq $expectedHash -and
    $report.executable.sha256 -ceq $profilerHash) 'Unexpected diagnostic report identity'
Assert-PoseComparison ((Get-PoseComparisonHash $manifestPath) -ceq $manifestHash -and
    (Get-PoseComparisonHash $profiler) -ceq $profilerHash -and
    (Get-PoseComparisonHash $modelPath) -ceq $expectedHash) 'Diagnostic artifact changed'
Assert-PoseComparison ((Get-PoseComparisonHash $report.evidence.path) -ceq $report.evidence.sha256) 'Raw profile hash mismatch'
Write-PoseComparisonJson ([ordered]@{
    schema_version = 1
    kind = 'candidate-cuda-diagnostic-manifest'
    manifest_sha256 = $manifestHash
    model_id = $artifacts[0].id
    model_sha256 = $expectedHash
    report_sha256 = Get-PoseComparisonHash $reportPath
    profile_sha256 = $report.evidence.sha256
    production_authorized = $false
}) (Join-Path $output 'diagnostic-run.json')
if (-not $report.cuda_observed -or $report.observed_node_counts.unknown -ne 0) {
    throw 'Unexpected provider observations; review diagnostic evidence before continuing'
}
Write-Host "Candidate diagnostic complete (not approved): $output"
