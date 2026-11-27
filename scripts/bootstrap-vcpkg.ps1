param(
    [string]$Destination = (Join-Path $PSScriptRoot '..\.deps\vcpkg')
)

$ErrorActionPreference = 'Stop'
$baseline = '9fa95bf687fa6ae7f2da0673a8067e5fc1515c27'
$resolved = [IO.Path]::GetFullPath($Destination)
$rootPath = [IO.Path]::GetPathRoot($resolved)
if ($resolved.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar) -eq
    $rootPath.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)) {
    throw "Refusing to use a filesystem root as a vcpkg destination: $resolved"
}
if (Test-Path -LiteralPath $resolved -PathType Leaf) {
    throw "vcpkg destination is a file: $resolved"
}

function Invoke-Native {
    param(
        [string]$Program,
        [string[]]$Arguments
    )
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Native command failed with exit code $LASTEXITCODE`: $Program $($Arguments -join ' ')"
    }
}

if (-not (Test-Path -LiteralPath $resolved)) {
    $parent = Split-Path -Parent $resolved
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Invoke-Native git @('clone', 'https://github.com/microsoft/vcpkg.git', $resolved)
} elseif (-not (Test-Path -LiteralPath (Join-Path $resolved '.git'))) {
    throw "Existing vcpkg destination is not a Git repository: $resolved"
}

Invoke-Native git @('-C', $resolved, 'fetch', '--no-tags', 'origin', $baseline)
$remote = (& git -C $resolved remote get-url origin).Trim()
if ($LASTEXITCODE -ne 0 -or $remote -notmatch 'microsoft/vcpkg(?:\.git)?$') {
    throw "Existing repository is not the expected Microsoft vcpkg remote: $resolved"
}
Invoke-Native git @('-C', $resolved, 'checkout', '--detach', $baseline)
$head = (& git -C $resolved rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $head -ne $baseline) {
    throw "vcpkg checkout did not resolve to the pinned baseline"
}
Invoke-Native (Join-Path $resolved 'bootstrap-vcpkg.bat') @('-disableMetrics')
Write-Output "Verified vcpkg baseline $baseline in $resolved"
Write-Output "`$env:VCPKG_ROOT='$resolved'"
