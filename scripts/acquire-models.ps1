param(
    [string]$Destination = (Join-Path $PSScriptRoot '..\models')
)

$ErrorActionPreference = 'Stop'
$destinationPath = [IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null
$artifacts = @(
    @{
        Name = 'detector'
        Url = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip'
        ArchiveHash = '36E09CA555916253FA1B5D51BEC01E48E98E1F0AF966C51CE3E844D8C8FC4CFC'
        OnnxHash = 'CEB11C07298F95C50D7C5ABEB906D03340C85F23AA79E3E66966E7FB6C307250'
    },
    @{
        Name = 'pose'
        Url = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip'
        ArchiveHash = '55B81170E236040B59FC792AD0A8315301AC4C079A3BDB1095D838AAD3088D18'
        OnnxHash = '26F3A19E61304A600DFB82D1001D41D24343B89FC70A33FFC84657E0B0BF2ECF'
    }
)

foreach ($artifact in $artifacts) {
    $archive = Join-Path $destinationPath "$($artifact.Name).zip"
    $extract = Join-Path $destinationPath ".$($artifact.Name)-extract"
    Invoke-WebRequest -Uri $artifact.Url -OutFile $archive
    $archiveHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash
    if ($archiveHash -ne $artifact.ArchiveHash) {
        throw "Checksum mismatch for $archive"
    }
    if (Test-Path $extract) {
        Remove-Item -LiteralPath $extract -Recurse -Force
    }
    Expand-Archive -LiteralPath $archive -DestinationPath $extract
    $onnxFiles = @(Get-ChildItem -LiteralPath $extract -Recurse -Filter '*.onnx' -File)
    if ($onnxFiles.Count -ne 1) {
        throw "Expected exactly one ONNX file in $archive"
    }
    $onnxHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $onnxFiles[0].FullName).Hash
    if ($onnxHash -ne $artifact.OnnxHash) {
        throw "ONNX checksum mismatch for $archive"
    }
    Copy-Item -LiteralPath $onnxFiles[0].FullName -Destination (Join-Path $destinationPath "$($artifact.Name).onnx") -Force
    Remove-Item -LiteralPath $extract -Recurse -Force
    Remove-Item -LiteralPath $archive -Force
}

Write-Output "Verified models acquired in $destinationPath"

