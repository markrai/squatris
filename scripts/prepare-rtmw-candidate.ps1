param(
    [Parameter(Mandatory)][string]$PoseFile,
    [ValidatePattern('^[0-9a-fA-F]{64}$')][string]$ExpectedOnnxSha256,
    [string]$OutputDirectory = 'qualification\rtmw-m-artifact'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. "$PSScriptRoot\pose-comparison.ps1"

$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$manifestPath = Join-Path $root 'models\rtmw-m-manifest.json'
$manifest = Read-PoseComparisonJson $manifestPath
$poses = @($manifest.artifacts | Where-Object role -ceq 'pose_estimator')
Assert-PoseComparison ($manifest.schema_version -eq 1 -and $poses.Count -eq 1 -and
    $poses[0].id -ceq 'rtmw-m-coco133-256x192' -and
    $poses[0].onnx_file -ceq 'rtmw-m.onnx' -and
    $poses[0].onnx_sha256 -cmatch '^[0-9a-f]{64}$') 'Invalid candidate manifest'
$hash = $poses[0].onnx_sha256
if ($ExpectedOnnxSha256 -and $ExpectedOnnxSha256.ToLowerInvariant() -cne $hash) {
    throw 'Supplied checksum does not match the repository pin'
}
$source = (Resolve-Path -LiteralPath $PoseFile).Path
Assert-PoseComparison ((Get-PoseComparisonHash $source) -ceq $hash) 'Candidate ONNX checksum mismatch'
$output = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $output) { throw "Output directory already exists: $output" }
New-Item -ItemType Directory -Path $output | Out-Null
$target = Join-Path $output 'rtmw-m.onnx'
Copy-Item -LiteralPath $source -Destination $target
Assert-PoseComparison ((Get-PoseComparisonHash $target) -ceq $hash) 'Copied candidate checksum mismatch'
Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $output 'manifest.json')
Write-Host "Verified pinned candidate: $target"
Write-Host "Candidate manifest: $(Join-Path $output 'manifest.json')"
