param(
    [Parameter(Mandatory)][string]$PoseFile,
    [Parameter(Mandatory)][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$ExpectedOnnxSha256,
    [string]$OutputDirectory = 'qualification\rtmw-m-artifact'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. "$PSScriptRoot\pose-comparison.ps1"

$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$source = (Resolve-Path -LiteralPath $PoseFile).Path
$hash = $ExpectedOnnxSha256.ToLowerInvariant()
Assert-PoseComparison ((Get-PoseComparisonHash $source) -ceq $hash) 'Candidate ONNX checksum mismatch'
Assert-PoseComparison ($hash -cne '26f3a19e61304a600dfb82d1001d41d24343b89fc70a33ffc84657e0b0bf2ecf') 'Candidate cannot use reference pose weights'
$baseline = Read-PoseComparisonJson (Join-Path $root 'models\manifest.json')
$detectors = @($baseline.artifacts | Where-Object role -ceq 'person_detector')
Assert-PoseComparison ($detectors.Count -eq 1) 'Expected a single reference detector'
$output = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $output) { throw "Output directory already exists: $output" }
New-Item -ItemType Directory -Path $output | Out-Null
$target = Join-Path $output 'rtmw-m.onnx'
Copy-Item -LiteralPath $source -Destination $target
Assert-PoseComparison ((Get-PoseComparisonHash $target) -ceq $hash) 'Copied candidate checksum mismatch'
Write-PoseComparisonJson ([ordered]@{
    schema_version = 1
    artifacts = @(
        $detectors[0],
        [ordered]@{
            id = 'rtmw-m-coco133-256x192'
            role = 'pose_estimator'
            upstream = 'OpenMMLab RTMW-m Cocktail14 ONNX SDK export'
            upstream_url = 'https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-m-s_simcc-cocktail14_270e-256x192_20231122.zip'
            onnx_sha256 = $hash
            onnx_file = 'rtmw-m.onnx'
            checksum_authority = 'explicit operator-supplied local artifact pin; not upstream-authenticated'
            input = 'RGB float32 NCHW 1x3x256x192 affine crop'
            output = 'float32 SimCC x [1,133,384], y [1,133,512]'
            redistribution = 'Not approved for redistribution; review trained-weight and dataset terms'
        }
    )
}) (Join-Path $output 'manifest.json')
Write-Host "Pinned local candidate: $target"
Write-Host "Candidate manifest: $(Join-Path $output 'manifest.json')"
