param(
    [ValidateSet('cpu', 'gpu')]
    [string]$Variant = 'cpu',
    [string]$Destination = (Join-Path $PSScriptRoot '..\.deps\onnxruntime')
)

$ErrorActionPreference = 'Stop'
$version = '1.20.1'
$package = if ($Variant -eq 'gpu') { 'microsoft.ml.onnxruntime.gpu.windows' } else { 'microsoft.ml.onnxruntime' }
$expectedSha512 = if ($Variant -eq 'gpu') {
    'ZqqJuLvyfGmOs2s/Ov0inj2C97Y9De+NH1KnAE+GtM6ANPiarTQ4qkTF4yFyOI56ZoWusDEKiznz3ctdvLcTJw=='
} else {
    '/IZ7tO4xeqphNGAHn6dU+Ate5kmugmsQKwJpl2qJBAq4rtl2j2hQWlUn+KoxYcwS4ToIw4KzXVdzHuhMaw1HjQ=='
}
$url = "https://api.nuget.org/v3-flatcontainer/$package/$version/$package.$version.nupkg"
$root = [IO.Path]::GetFullPath($Destination)
$archive = "$root.nupkg"
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
    throw "ONNX Runtime package checksum mismatch"
}
$extract = "$root-extract"
if (Test-Path $root) {
    Remove-Item -LiteralPath $root -Recurse -Force
}
if (Test-Path $extract) {
    Remove-Item -LiteralPath $extract -Recurse -Force
}
Expand-Archive -LiteralPath $archive -DestinationPath $extract
New-Item -ItemType Directory -Path (Join-Path $root 'include') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $root 'lib') -Force | Out-Null
Copy-Item -Path (Join-Path $extract 'build\native\include\*') -Destination (Join-Path $root 'include') -Force
Copy-Item -Path (Join-Path $extract 'runtimes\win-x64\native\*') -Destination (Join-Path $root 'lib') -Force
Copy-Item -LiteralPath (Join-Path $extract 'LICENSE') -Destination $root -Force
Copy-Item -LiteralPath (Join-Path $extract 'ThirdPartyNotices.txt') -Destination $root -Force
Remove-Item -LiteralPath $extract -Recurse -Force
Remove-Item -LiteralPath $archive -Force
Write-Output "Set SQUATRIS_ONNXRUNTIME_ROOT=$root for this shell with:"
Write-Output "`$env:SQUATRIS_ONNXRUNTIME_ROOT='$root'"
