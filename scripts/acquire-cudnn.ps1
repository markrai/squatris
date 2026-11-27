param(
    [string]$Destination = (Join-Path $PSScriptRoot '..\.deps\cudnn\9.1.0.70')
)

$ErrorActionPreference = 'Stop'
$version = '9.1.0.70'
$url = 'https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.1.0.70_cuda12-archive.zip'
$expectedSha256 = '3083b5bf20b0b2b29305de088e0b2cdbb78b5190655a27e8a7c596e47042eedf'
$runtimeRoot = [IO.Path]::GetFullPath($Destination)
$driveRoot = [IO.Path]::GetPathRoot($runtimeRoot)
if ($runtimeRoot.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar) -eq
    $driveRoot.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)) {
    throw "Refusing to use a filesystem root as a destination: $runtimeRoot"
}
if (Test-Path -LiteralPath $runtimeRoot -PathType Leaf) {
    throw "Destination is a file: $runtimeRoot"
}

$marker = Join-Path $runtimeRoot 'squatris-cudnn.json'
$primaryDll = Join-Path $runtimeRoot 'bin\cudnn64_9.dll'
if (Test-Path -LiteralPath $runtimeRoot) {
    if (-not (Test-Path -LiteralPath $marker -PathType Leaf) -or
        -not (Test-Path -LiteralPath $primaryDll -PathType Leaf)) {
        throw "Existing cuDNN destination is not a complete managed installation: $runtimeRoot"
    }
    $installed = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
    if ($installed.version -eq $version -and $installed.sha256 -eq $expectedSha256) {
        Write-Output "Verified existing cuDNN $version runtime: $runtimeRoot"
        Write-Output "Prepend to process PATH: $(Join-Path $runtimeRoot 'bin')"
        return
    }
    throw "Existing cuDNN destination does not match the pinned runtime: $runtimeRoot"
}

$parent = Split-Path -Parent $runtimeRoot
New-Item -ItemType Directory -Path $parent -Force | Out-Null
$operationId = [Guid]::NewGuid().ToString('N')
$archive = Join-Path $parent ".cudnn-$version-$operationId.zip"
$extract = Join-Path $parent ".cudnn-$version-$operationId-extract"
$stage = Join-Path $parent ".cudnn-$version-$operationId-stage"
try {
    Invoke-WebRequest -Uri $url -OutFile $archive
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash
    if ($actualHash -ine $expectedSha256) {
        throw "cuDNN archive checksum mismatch"
    }
    New-Item -ItemType Directory -Path $extract | Out-Null
    Expand-Archive -LiteralPath $archive -DestinationPath $extract
    $dlls = @(Get-ChildItem -LiteralPath $extract -Filter 'cudnn*.dll' -File -Recurse)
    if ($dlls.Count -lt 2 -or -not ($dlls.Name -contains 'cudnn64_9.dll')) {
        throw "cuDNN archive does not contain the expected Windows runtime DLLs"
    }
    $stageBin = Join-Path $stage 'bin'
    New-Item -ItemType Directory -Path $stageBin -Force | Out-Null
    foreach ($dll in $dlls) {
        Copy-Item -LiteralPath $dll.FullName -Destination $stageBin
    }
    $license = Get-ChildItem -LiteralPath $extract -File -Recurse |
        Where-Object { $_.Name -match '^LICENSE' } | Select-Object -First 1
    if ($license) {
        Copy-Item -LiteralPath $license.FullName -Destination $stage
    }
    [ordered]@{
        version = $version
        cuda_major = 12
        source = $url
        sha256 = $expectedSha256
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $stage 'squatris-cudnn.json') -Encoding utf8
    Move-Item -LiteralPath $stage -Destination $runtimeRoot
} finally {
    if (Test-Path -LiteralPath $stage) {
        Remove-Item -LiteralPath $stage -Recurse -Force
    }
    if (Test-Path -LiteralPath $extract) {
        Remove-Item -LiteralPath $extract -Recurse -Force
    }
    if (Test-Path -LiteralPath $archive) {
        Remove-Item -LiteralPath $archive -Force
    }
}

Write-Output "Installed cuDNN $version runtime in $runtimeRoot"
Write-Output "Prepend to process PATH: $(Join-Path $runtimeRoot 'bin')"
