param(
    [ValidateSet('all', 'build-test', 'clang-tidy', 'onnx-cpu-build-test')]
    [string]$Job = 'all'
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    Write-Host "> $Program $($Arguments -join ' ')"
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE"
    }
}

function Initialize-Compiler {
    $installation = $env:VSINSTALLDIR
    if ($env:VSCMD_ARG_TGT_ARCH -notin @('x64', 'amd64') -or
        -not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
        if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
            throw 'Visual Studio installer (vswhere.exe) was not found'
        }
        $installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -ne 0 -or -not $installation) {
            throw 'Visual Studio with x64 C++ tools was not found'
        }
        $module = Join-Path $installation 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll'
        Import-Module $module -ErrorAction Stop
        Enter-VsDevShell -VsInstallPath $installation -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64'
    }
    if ($env:VSCMD_ARG_TGT_ARCH -notin @('x64', 'amd64')) {
        throw 'The Visual Studio developer environment is not targeting x64'
    }
    if ($installation) {
        $bundledTools = @{
            'cmake.exe' = 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
            'ninja.exe' = 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja'
            'clang-tidy.exe' = 'VC\Tools\Llvm\x64\bin'
        }
        foreach ($tool in $bundledTools.Keys) {
            if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
                $directory = Join-Path $installation $bundledTools[$tool]
                if (Test-Path -LiteralPath (Join-Path $directory $tool) -PathType Leaf) {
                    $env:PATH = "$directory;$env:PATH"
                }
            }
        }
    }
    foreach ($tool in @('cl.exe', 'git.exe', 'cmake.exe', 'ctest.exe', 'ninja.exe')) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            throw "Required tool not found: $tool"
        }
    }
}

function Initialize-Vcpkg {
    $configuration = Get-Content -LiteralPath 'vcpkg-configuration.json' -Raw | ConvertFrom-Json
    $baseline = $configuration.'default-registry'.baseline
    if (-not $env:VCPKG_ROOT) {
        $env:VCPKG_ROOT = Join-Path $PWD.Path '.deps\vcpkg'
    }
    $env:VCPKG_ROOT = [IO.Path]::GetFullPath($env:VCPKG_ROOT)
    $executable = Join-Path $env:VCPKG_ROOT 'vcpkg.exe'
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        $localRoot = Join-Path $PWD.Path '.deps\vcpkg'
        if ($env:VCPKG_ROOT -ne $localRoot) {
            throw "Supplied VCPKG_ROOT does not contain vcpkg.exe: $env:VCPKG_ROOT"
        }
        & "$PSScriptRoot\bootstrap-vcpkg.ps1" -Destination $localRoot
    }
    $head = & git.exe -C $env:VCPKG_ROOT rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $head -ne $baseline) {
        throw "VCPKG_ROOT must be at the pinned baseline $baseline"
    }
    Write-Host "vcpkg baseline: $head"
}

function Invoke-CiJob {
    param([string]$Name, [string]$Preset, [bool]$RunTests)
    Write-Host ""
    Write-Host "=== $Name ==="
    Invoke-Checked cmake.exe @('--fresh', '--preset', $Preset,
        '-DBUILD_TESTING=ON', '-DSQUATRIS_BUILD_TESTS=ON',
        '-DSQUATRIS_BUILD_APPS=ON', '-DSQUATRIS_BUILD_BENCHMARKS=ON',
        '-DSQUATRIS_ENABLE_HARDWARE_TESTS=OFF', '-DSQUATRIS_ENABLE_GUI_SMOKE_TEST=OFF')
    Invoke-Checked cmake.exe @('--build', '--preset', $Preset, '--clean-first')
    if ($RunTests) {
        Invoke-Checked ctest.exe @('--preset', $Preset)
    }
    Write-Host "PASS: $Name"
}

if (-not $IsWindows) {
    throw 'Squatris CI requires Windows x64 and MSVC'
}

Push-Location (Join-Path $PSScriptRoot '..')
try {
    Initialize-Compiler
    if ($Job -in @('all', 'clang-tidy') -and
        -not (Get-Command clang-tidy.exe -ErrorAction SilentlyContinue)) {
        throw 'Clang-Tidy is required for the analysis job'
    }
    Initialize-Vcpkg
    Invoke-Checked cmake.exe @('--version')
    Invoke-Checked ninja.exe @('--version')
    if ($Job -in @('all', 'clang-tidy')) {
        Invoke-Checked clang-tidy.exe @('--version')
    }

    if ($Job -in @('all', 'build-test')) {
        Invoke-CiJob 'build-test' 'ci-windows-release' $true
    }
    if ($Job -in @('all', 'clang-tidy')) {
        Invoke-CiJob 'clang-tidy' 'ci-windows-clang-tidy' $false
    }
    if ($Job -in @('all', 'onnx-cpu-build-test')) {
        & "$PSScriptRoot\acquire-onnxruntime.ps1" -Variant cpu
        $env:SQUATRIS_ONNXRUNTIME_ROOT = (Resolve-Path '.deps\onnxruntime\cpu').Path
        Invoke-CiJob 'onnx-cpu-build-test' 'ci-windows-onnx-release' $true
    }
    Write-Host ""
    Write-Host "CI PASS: $Job"
} finally {
    Pop-Location
}
