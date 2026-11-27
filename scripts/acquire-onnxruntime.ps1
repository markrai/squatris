param(
    [ValidateSet('cpu', 'gpu')]
    [string]$Variant = 'cpu',
    [string]$Destination
)

$ErrorActionPreference = 'Stop'
$version = '1.20.1'
if (-not $Destination) {
    $Destination = Join-Path $PSScriptRoot "..\.deps\onnxruntime\$Variant"
}
$package = if ($Variant -eq 'gpu') { 'microsoft.ml.onnxruntime.gpu.windows' } else { 'microsoft.ml.onnxruntime' }
$expectedSha512 = if ($Variant -eq 'gpu') {
    'ZqqJuLvyfGmOs2s/Ov0inj2C97Y9De+NH1KnAE+GtM6ANPiarTQ4qkTF4yFyOI56ZoWusDEKiznz3ctdvLcTJw=='
} else {
    '/IZ7tO4xeqphNGAHn6dU+Ate5kmugmsQKwJpl2qJBAq4rtl2j2hQWlUn+KoxYcwS4ToIw4KzXVdzHuhMaw1HjQ=='
}

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

function Get-RequiredArtifacts {
    param([string]$RequestedVariant)
    $artifacts = @(
        'include\onnxruntime_c_api.h',
        'include\onnxruntime_cxx_api.h',
        'include\onnxruntime_cxx_inline.h',
        'lib\onnxruntime.dll',
        'lib\onnxruntime.lib'
    )
    if ($RequestedVariant -eq 'gpu') {
        $artifacts += @(
            'lib\onnxruntime_providers_shared.dll',
            'lib\onnxruntime_providers_cuda.dll'
        )
    }
    return $artifacts
}

function Get-ArtifactInventory {
    param([string]$Root, [string]$RequestedVariant)
    return @(Get-RequiredArtifacts $RequestedVariant | ForEach-Object {
        $path = Join-Path $Root $_
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Required ONNX Runtime artifact is missing: $path"
        }
        $item = Get-Item -LiteralPath $path
        [ordered]@{
            relative_path = $_
            length = $item.Length
            sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    })
}

function Test-ArtifactInventory {
    param([string]$Root, [string]$RequestedVariant, [object[]]$Expected)
    if (-not $Expected) {
        return $false
    }
    $actual = @(Get-ArtifactInventory $Root $RequestedVariant)
    if ($actual.Count -ne $Expected.Count) {
        return $false
    }
    for ($index = 0; $index -lt $actual.Count; ++$index) {
        if ($actual[$index].relative_path -ne $Expected[$index].relative_path -or
            $actual[$index].length -ne $Expected[$index].length -or
            $actual[$index].sha256 -ne $Expected[$index].sha256) {
            return $false
        }
    }
    return $true
}

function Write-JsonAtomically {
    param([object]$Value, [string]$Path)
    $temporary = "$Path.$([Guid]::NewGuid().ToString('N')).tmp"
    try {
        $Value | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding utf8
        [IO.File]::Move($temporary, $Path, $true)
    } finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
}

$runtimeRoot = Assert-SafeDirectory $Destination
$marker = Join-Path $runtimeRoot 'squatris-runtime.json'
if (Test-Path -LiteralPath $runtimeRoot) {
    if (-not (Test-Path -LiteralPath $marker -PathType Leaf)) {
        throw "Existing runtime destination is not managed by this script: $runtimeRoot"
    }
    $installed = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
    $identityMatches = $installed.version -eq $version -and $installed.variant -eq $Variant -and
        $installed.package -eq $package -and $installed.sha512 -eq $expectedSha512
    if ($identityMatches -and (Test-ArtifactInventory $runtimeRoot $Variant $installed.artifacts)) {
        Write-Output "Verified existing ONNX Runtime $version $Variant installation: $runtimeRoot"
        Write-Output "`$env:SQUATRIS_ONNXRUNTIME_ROOT='$runtimeRoot'"
        return
    }
    if ($identityMatches -and -not $installed.artifacts) {
        $upgraded = [ordered]@{
            version = $version
            variant = $Variant
            package = $package
            sha512 = $expectedSha512
            artifacts = @(Get-ArtifactInventory $runtimeRoot $Variant)
        }
        Write-JsonAtomically $upgraded $marker
        Write-Output "Recorded and verified per-artifact integrity for existing ONNX Runtime $version $Variant installation: $runtimeRoot"
        Write-Output "`$env:SQUATRIS_ONNXRUNTIME_ROOT='$runtimeRoot'"
        return
    }
    throw "Existing runtime destination does not match requested version and variant: $runtimeRoot"
}

$parent = Split-Path -Parent $runtimeRoot
New-Item -ItemType Directory -Path $parent -Force | Out-Null
$operationId = [Guid]::NewGuid().ToString('N')
$archive = Join-Path $parent ".$package-$version-$operationId.nupkg"
$extract = Join-Path $parent ".$package-$version-$operationId-extract"
$stage = Join-Path $parent ".$package-$version-$operationId-stage"
$url = "https://api.nuget.org/v3-flatcontainer/$package/$version/$package.$version.nupkg"

try {
    Invoke-WebRequest -Uri $url -OutFile $archive
    $sha512 = [Security.Cryptography.SHA512]::Create()
    $stream = [IO.File]::OpenRead($archive)
    try {
        $actualSha512 = [Convert]::ToBase64String($sha512.ComputeHash($stream))
    } finally {
        $stream.Dispose()
        $sha512.Dispose()
    }
    if ($actualSha512 -ne $expectedSha512) {
        throw "ONNX Runtime package checksum mismatch for $package $version"
    }
    New-Item -ItemType Directory -Path $extract | Out-Null
    Expand-Archive -LiteralPath $archive -DestinationPath $extract
    $includeSource = @(
        (Join-Path $extract 'buildTransitive\native\include'),
        (Join-Path $extract 'build\native\include')
    ) | Where-Object {
        Test-Path -LiteralPath (Join-Path $_ 'onnxruntime_cxx_api.h') -PathType Leaf
    } | Select-Object -First 1
    $librarySource = Join-Path $extract 'runtimes\win-x64\native'
    if (-not $includeSource -or
        -not (Test-Path -LiteralPath (Join-Path $librarySource 'onnxruntime.dll'))) {
        throw "ONNX Runtime package layout is not the pinned Windows x64 layout"
    }
    New-Item -ItemType Directory -Path (Join-Path $stage 'include') -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $stage 'lib') -Force | Out-Null
    Copy-Item -Path (Join-Path $includeSource '*') -Destination (Join-Path $stage 'include') -Force
    Copy-Item -Path (Join-Path $librarySource '*') -Destination (Join-Path $stage 'lib') -Force
    Copy-Item -LiteralPath (Join-Path $extract 'LICENSE') -Destination $stage -Force
    Copy-Item -LiteralPath (Join-Path $extract 'ThirdPartyNotices.txt') -Destination $stage -Force
    $installation = [ordered]@{
        version = $version
        variant = $Variant
        package = $package
        sha512 = $expectedSha512
        artifacts = @(Get-ArtifactInventory $stage $Variant)
    }
    Write-JsonAtomically $installation (Join-Path $stage 'squatris-runtime.json')
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

Write-Output "Installed ONNX Runtime $version $Variant in $runtimeRoot"
Write-Output "`$env:SQUATRIS_ONNXRUNTIME_ROOT='$runtimeRoot'"
