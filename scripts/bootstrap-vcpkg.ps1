param(
    [string]$Destination = (Join-Path $PSScriptRoot '..\.deps\vcpkg')
)

$ErrorActionPreference = 'Stop'
$baseline = '9fa95bf687fa6ae7f2da0673a8067e5fc1515c27'
$resolved = [IO.Path]::GetFullPath($Destination)
if (-not (Test-Path $resolved)) {
    git clone https://github.com/microsoft/vcpkg.git $resolved
}
git -C $resolved fetch origin $baseline
git -C $resolved checkout --detach $baseline
& (Join-Path $resolved 'bootstrap-vcpkg.bat') -disableMetrics
Write-Output "Set VCPKG_ROOT=$resolved for this shell with:"
Write-Output "`$env:VCPKG_ROOT='$resolved'"

