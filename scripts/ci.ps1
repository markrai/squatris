param(
    [ValidateSet('all', 'build-test', 'clang-tidy', 'onnx-cpu-build-test')]
    [string]$Job = 'all',
    [string]$VcpkgRoot,
    [switch]$Quiet
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
. "$PSScriptRoot\ci-diagnostics.ps1"

$failures = [System.Collections.Generic.List[object]]::new()
$results = [System.Collections.Generic.List[object]]::new()

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
    param([string]$SelectedRoot)
    $configuration = Get-Content -LiteralPath 'vcpkg-configuration.json' -Raw | ConvertFrom-Json
    $baseline = $configuration.'default-registry'.baseline
    $localRoot = [IO.Path]::GetFullPath((Join-Path $PWD.Path '.deps\vcpkg'))
    $env:VCPKG_ROOT = $SelectedRoot
    Write-Host "vcpkg root: $SelectedRoot"
    $executable = Join-Path $SelectedRoot 'vcpkg.exe'
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        if ($SelectedRoot -ne $localRoot) {
            throw "Explicit vcpkg root does not contain vcpkg.exe: $SelectedRoot"
        }
        & "$PSScriptRoot\bootstrap-vcpkg.ps1" -Destination $localRoot
    }
    if (-not (Test-Path -LiteralPath (Join-Path $SelectedRoot '.git'))) {
        throw "vcpkg root is not a Git checkout: $SelectedRoot"
    }
    $head = & git.exe -C $SelectedRoot rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $head -ne $baseline) {
        throw "vcpkg checkout at $SelectedRoot must be at the pinned baseline $baseline (found: $head)"
    }
    Write-Host "vcpkg baseline: $head"
}

function Invoke-CiStep {
    param([string]$Name, [scriptblock]$Action)
    $log = Join-Path $reportDirectory "$Name.log"
    $writer = [IO.StreamWriter]::new($log, $false, [Text.UTF8Encoding]::new($false))
    $failure = $null
    Write-Host "  $Name"
    try {
        & $Action *>&1 | ForEach-Object {
            $line = $_.ToString()
            $writer.WriteLine($line)
            if (-not $Quiet) {
                Write-Host $line
            }
        }
    } catch {
        $failure = $_
        $writer.WriteLine($_.ToString())
    } finally {
        $writer.Dispose()
    }
    if ($null -ne $failure) {
        $diagnostics = @(Get-CiDiagnostics -Lines @(Get-Content -LiteralPath $log) -RepositoryRoot $repositoryRoot)
        $failures.Add([pscustomobject]@{
            Step = $Name
            Log = $log
            Message = $failure.Exception.Message
            Diagnostics = $diagnostics
        })
        throw "Failed step: $Name"
    }
}

function Invoke-CiJob {
    param([string]$Name, [string]$Preset, [bool]$RunTests)
    Write-Host ""
    Write-Host "=== $Name ==="
    try {
        if ($Name -eq 'clang-tidy') {
            Invoke-CiStep "$Name-prerequisites" {
                if (-not (Get-Command clang-tidy.exe -ErrorAction SilentlyContinue)) {
                    throw 'Clang-Tidy is required for the analysis job'
                }
                Invoke-Checked clang-tidy.exe @('--version')
            }
        }
        if ($Name -eq 'onnx-cpu-build-test') {
            Invoke-CiStep "$Name-prerequisites" {
                & "$PSScriptRoot\acquire-onnxruntime.ps1" -Variant cpu
                $env:SQUATRIS_ONNXRUNTIME_ROOT = (Resolve-Path '.deps\onnxruntime\cpu').Path
            }
        }
        Invoke-CiStep "$Name-configure" {
            Invoke-Checked cmake.exe @('--fresh', '--preset', $Preset,
                '-DBUILD_TESTING=ON', '-DSQUATRIS_BUILD_TESTS=ON',
                '-DSQUATRIS_BUILD_APPS=ON', '-DSQUATRIS_BUILD_BENCHMARKS=ON',
                '-DSQUATRIS_ENABLE_HARDWARE_TESTS=OFF', '-DSQUATRIS_ENABLE_GUI_SMOKE_TEST=OFF')
        }
        Invoke-CiStep "$Name-build" {
            Invoke-Checked cmake.exe @('--build', '--preset', $Preset, '--clean-first', '--', '-k', '0')
        }
        if ($RunTests) {
            Invoke-CiStep "$Name-test" {
                Invoke-Checked ctest.exe @('--preset', $Preset)
            }
        }
        $results.Add([pscustomobject]@{ Job = $Name; Status = 'PASS' })
        Write-Host "PASS: $Name"
    } catch {
        $results.Add([pscustomobject]@{ Job = $Name; Status = 'FAIL' })
        Write-Host "FAIL: $Name" -ForegroundColor Red
    }
}

if (-not $IsWindows) {
    throw 'Squatris CI requires Windows x64 and MSVC'
}

Push-Location (Join-Path $PSScriptRoot '..')
try {
    $repositoryRoot = $PWD.Path
    $runId = "$(Get-Date -Format 'yyyyMMdd-HHmmss')-$([Guid]::NewGuid().ToString('N').Substring(0, 8))"
    $reportDirectory = Join-Path $repositoryRoot "out\ci-reports\$runId"
    New-Item -ItemType Directory -Path $reportDirectory -Force | Out-Null
    Write-Host "CI logs: $reportDirectory"
    $setupPassed = $false
    try {
        Invoke-CiStep 'setup' {
            $selectedRoot = if ([string]::IsNullOrWhiteSpace($VcpkgRoot)) {
                [IO.Path]::GetFullPath((Join-Path $repositoryRoot '.deps\vcpkg'))
            } else {
                [IO.Path]::GetFullPath($VcpkgRoot, $repositoryRoot)
            }
            Initialize-Compiler
            Initialize-Vcpkg -SelectedRoot $selectedRoot
            Invoke-Checked cmake.exe @('--version')
            Invoke-Checked ninja.exe @('--version')
            & "$PSScriptRoot\tests\ci-diagnostics.tests.ps1"
            & "$PSScriptRoot\tests\pose-comparison.tests.ps1"
        }
        $setupPassed = $true
    } catch {
        Write-Host 'FAIL: setup' -ForegroundColor Red
    }

    if ($setupPassed) {
        if ($Job -in @('all', 'build-test')) {
            Invoke-CiJob 'build-test' 'ci-windows-release' $true
        }
        if ($Job -in @('all', 'clang-tidy')) {
            Invoke-CiJob 'clang-tidy' 'ci-windows-clang-tidy' $false
        }
        if ($Job -in @('all', 'onnx-cpu-build-test')) {
            Invoke-CiJob 'onnx-cpu-build-test' 'ci-windows-onnx-release' $true
        }
    } else {
        foreach ($name in @('build-test', 'clang-tidy', 'onnx-cpu-build-test')) {
            if ($Job -in @('all', $name)) {
                $results.Add([pscustomobject]@{ Job = $name; Status = 'SKIPPED (setup failed)' })
            }
        }
    }

    $summary = [System.Collections.Generic.List[string]]::new()
    $summary.Add('')
    $summary.Add('================ CI SUMMARY ================')
    foreach ($result in $results) {
        $summary.Add(('{0,-22} {1}' -f $result.Job, $result.Status))
    }
    foreach ($failure in $failures) {
        $summary.Add('')
        $summary.Add("--- $($failure.Step) ---")
        $summary.Add($failure.Message)
        foreach ($diagnostic in $failure.Diagnostics) {
            $summary.Add("  $diagnostic")
        }
        $summary.Add("Full log: $($failure.Log)")
    }
    $failed = -not $setupPassed -or $failures.Count -gt 0 -or
        @($results | Where-Object Status -ne 'PASS').Count -gt 0
    $summary.Add('')
    $summary.Add($(if ($failed) { 'CI FAIL' } else { 'CI PASS' }))
    $summaryPath = Join-Path $reportDirectory 'summary.txt'
    $summary.Add("Summary: $summaryPath")
    $summary | Set-Content -LiteralPath $summaryPath -Encoding utf8
    $summary | ForEach-Object { Write-Host $_ }
} finally {
    Pop-Location
}
if ($failed) {
    exit 1
}
exit 0
