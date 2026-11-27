param(
    [string]$Destination = (Join-Path $PSScriptRoot '..\models'),
    [string]$Manifest = (Join-Path $PSScriptRoot '..\models\manifest.json')
)

$ErrorActionPreference = 'Stop'

function Assert-SafeDirectory {
    param([string]$Path)
    $fullPath = [IO.Path]::GetFullPath($Path)
    $rootPath = [IO.Path]::GetPathRoot($fullPath)
    if ($fullPath.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar) -eq
        $rootPath.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)) {
        throw "Refusing to use a filesystem root as a destination: $fullPath"
    }
    if (Test-Path -LiteralPath $fullPath -PathType Leaf) {
        throw "Destination is a file: $fullPath"
    }
    return $fullPath
}

$destinationPath = Assert-SafeDirectory $Destination
$manifestPath = [IO.Path]::GetFullPath($Manifest)
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Model manifest does not exist: $manifestPath"
}
$modelManifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($modelManifest.schema_version -ne 1 -or -not $modelManifest.artifacts) {
    throw "Unsupported or empty model manifest: $manifestPath"
}
New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null

foreach ($artifact in $modelManifest.artifacts) {
    if (-not $artifact.id -or -not $artifact.url -or -not $artifact.archive_sha256 -or
        -not $artifact.onnx_sha256 -or -not $artifact.onnx_file) {
        throw "Model manifest artifact is incomplete"
    }
    $target = Join-Path $destinationPath ([IO.Path]::GetFileName([string]$artifact.onnx_file))
    if (Test-Path -LiteralPath $target -PathType Leaf) {
        $existingHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash
        if ($existingHash -ieq [string]$artifact.onnx_sha256) {
            Write-Output "Verified existing model: $target"
            continue
        }
    }

    $operationId = [Guid]::NewGuid().ToString('N')
    $archive = Join-Path $destinationPath ".$($artifact.id)-$operationId.zip"
    $extract = Join-Path $destinationPath ".$($artifact.id)-$operationId"
    try {
        Invoke-WebRequest -Uri ([string]$artifact.url) -OutFile $archive
        $archiveHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash
        if ($archiveHash -ine [string]$artifact.archive_sha256) {
            throw "Archive checksum mismatch for $($artifact.id)"
        }
        New-Item -ItemType Directory -Path $extract | Out-Null
        Expand-Archive -LiteralPath $archive -DestinationPath $extract
        $onnxFiles = @(Get-ChildItem -LiteralPath $extract -Recurse -Filter '*.onnx' -File)
        if ($onnxFiles.Count -ne 1) {
            throw "Expected exactly one ONNX file for $($artifact.id)"
        }
        $onnxHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $onnxFiles[0].FullName).Hash
        if ($onnxHash -ine [string]$artifact.onnx_sha256) {
            throw "ONNX checksum mismatch for $($artifact.id)"
        }
        Copy-Item -LiteralPath $onnxFiles[0].FullName -Destination $target -Force
    } finally {
        if (Test-Path -LiteralPath $extract) {
            Remove-Item -LiteralPath $extract -Recurse -Force
        }
        if (Test-Path -LiteralPath $archive) {
            Remove-Item -LiteralPath $archive -Force
        }
    }
    Write-Output "Acquired and verified model: $target"
}

Write-Output "All manifest models are verified in $destinationPath"
