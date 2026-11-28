param(
    [Parameter(Mandatory)][string]$Workload,
    [Parameter(Mandatory)][string]$ReferenceDirectory,
    [Parameter(Mandatory)][string]$CandidateDirectory,
    [Parameter(Mandatory)][string]$Output,
    [string]$Benchmark
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
Set-StrictMode -Version Latest
. "$PSScriptRoot\pose-comparison.ps1"

$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$validator = if ($Benchmark) { (Resolve-Path -LiteralPath $Benchmark).Path } else { Join-Path $root 'out\build\windows-cuda-release\tools\benchmark\SquatrisPoseBenchmark.exe' }
if (-not (Test-Path -LiteralPath $validator -PathType Leaf)) { throw 'Build windows-cuda-release first' }
$validatorHash = Get-PoseComparisonHash $validator
$work = Read-PoseWorkload $Workload
$referenceRoot = (Resolve-Path -LiteralPath $ReferenceDirectory).Path
$candidateRoot = (Resolve-Path -LiteralPath $CandidateDirectory).Path
Assert-PoseComparison ($referenceRoot -ne $candidateRoot) 'Supply two independent run directories'
$referenceRun = Read-PoseComparisonJson (Join-Path $referenceRoot 'run.json')
$candidateRun = Read-PoseComparisonJson (Join-Path $candidateRoot 'run.json')
foreach ($run in @($referenceRun, $candidateRun)) {
    Assert-PoseComparison ($run.schema_version -eq 1 -and $run.workload_sha256 -ceq $work.sha256 -and
        $run.reports.Count -eq $work.clips.Count) 'Incomplete or mismatched run manifest'
}
$results = foreach ($clip in $work.clips) {
    $reports = foreach ($side in @(
        @{ root = $referenceRoot; run = $referenceRun },
        @{ root = $candidateRoot; run = $candidateRun }
    )) {
        $entry = @($side.run.reports | Where-Object id -ceq $clip.id)
        Assert-PoseComparison ($entry.Count -eq 1) "Missing or duplicate run entry: $($clip.id)"
        $validated = Get-ValidatedPoseReport $validator (Join-Path $side.root "$($clip.id).json") $clip
        Assert-PoseComparison ($entry[0].report_sha256 -ceq $validated.sha256) "Report hash mismatch: $($clip.id)"
        Assert-PoseComparison ((Get-PoseRunIdentity $validated.report) -ceq
            ($side.run.model_identity | ConvertTo-Json -Compress -Depth 20)) "Model identity mismatch: $($clip.id)"
        $validated
    }
    Compare-PoseClip $reports[0].report $reports[1].report $clip
}
Assert-PoseComparison ((Get-PoseComparisonHash $work.path) -ceq $work.sha256 -and
    (Get-PoseComparisonHash $validator) -ceq $validatorHash) 'Workload or validator changed during comparison'
foreach ($clip in $work.clips) {
    Assert-PoseComparison ((Get-PoseComparisonHash $clip.path) -ceq $clip.sha256) "Clip changed: $($clip.id)"
}
$destination = [IO.Path]::GetFullPath($Output)
Write-PoseComparisonJson ([ordered]@{
    schema_version = 1
    workload_sha256 = $work.sha256
    validator_sha256 = $validatorHash
    reference = $referenceRun.model_identity
    candidate = $candidateRun.model_identity
    reference_run_manifest = $referenceRun
    candidate_run_manifest = $candidateRun
    frame_identity_scope = 'source file SHA-256, sequential indices, media timestamps, dimensions, OpenCV version; decoded pixels are not independently hashed'
    physical_accuracy_claim = $false
    confidence_scores_cross_model_comparable = $false
    ranking = $null
    clips = @($results)
}) $destination
Write-Host "Comparison written: $destination"
