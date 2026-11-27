param(
    [Parameter(Mandatory = $true)]
    [string]$Video,
    [string]$OutputDirectory,
    [int]$CameraIndex = 0,
    [int]$WarmupIterations = 100,
    [int]$MeasuredIterations = 2000,
    [int]$BenchmarkRepeats = 3,
    [int]$SustainedSeconds = 300
)

$ErrorActionPreference = 'Stop'
if ($WarmupIterations -lt 1 -or $MeasuredIterations -lt 1 -or $BenchmarkRepeats -lt 3 -or
    $SustainedSeconds -lt 300) {
    throw 'Warm-up, measured iterations, at least three repeats, and a five-minute sustained run are required'
}

$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$videoPath = [IO.Path]::GetFullPath($Video)
if (-not (Test-Path -LiteralPath $videoPath -PathType Leaf)) {
    throw "Qualification video does not exist: $videoPath"
}
if (-not $OutputDirectory) {
    $timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $OutputDirectory = Join-Path $repositoryRoot "qualification\titan-$timestamp"
}
$reportRoot = [IO.Path]::GetFullPath($OutputDirectory)
$driveRoot = [IO.Path]::GetPathRoot($reportRoot)
if ($reportRoot.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar) -eq
    $driveRoot.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)) {
    throw 'Refusing to use a filesystem root for qualification output'
}
New-Item -ItemType Directory -Path $reportRoot -Force | Out-Null

$steps = [Collections.Generic.List[object]]::new()
$qualificationErrors = [Collections.Generic.List[string]]::new()
$benchmarkReports = [Collections.Generic.List[string]]::new()
$resourceReports = [Collections.Generic.List[string]]::new()
$startedAt = Get-Date

function Add-Step {
    param([string]$Name, [bool]$Passed, [string]$Evidence, [string]$Message)
    $steps.Add([ordered]@{ name = $Name; passed = $Passed; evidence = $Evidence; message = $Message })
}

function Quote-ProcessArgument {
    param([string]$Value)
    if ($Value -notmatch '[\s"]') {
        return $Value
    }
    return '"' + $Value.Replace('"', '\"') + '"'
}

function Write-JsonAtomically {
    param([object]$Value, [string]$Path, [int]$Depth = 14)
    $temporary = "$Path.$([Guid]::NewGuid().ToString('N')).tmp"
    try {
        $Value | ConvertTo-Json -Depth $Depth | Set-Content -LiteralPath $temporary -Encoding utf8
        [IO.File]::Move($temporary, $Path, $true)
    } finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
}

function Get-NvidiaResourceSample {
    param([string]$WorkingDirectory)
    $identifier = [Guid]::NewGuid().ToString('N')
    $stdoutPath = Join-Path $WorkingDirectory ".nvidia-smi-$identifier.stdout"
    $stderrPath = Join-Path $WorkingDirectory ".nvidia-smi-$identifier.stderr"
    $process = $null
    try {
        $process = Start-Process -FilePath 'nvidia-smi' -ArgumentList @(
            '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'
        ) -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath `
            -WindowStyle Hidden
        if (-not $process.WaitForExit(2000)) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            if (-not $process.WaitForExit(2000)) {
                throw 'nvidia-smi did not terminate after its resource-monitor deadline'
            }
            throw 'nvidia-smi exceeded its two-second resource-monitor deadline'
        }
        if ($process.ExitCode -ne 0) {
            $errorText = if (Test-Path -LiteralPath $stderrPath) {
                Get-Content -LiteralPath $stderrPath -Raw
            } else { '' }
            throw "nvidia-smi resource query failed with exit code $($process.ExitCode): $errorText"
        }
        $values = @(Get-Content -LiteralPath $stdoutPath)
        if ($values.Count -eq 0 -or $values[0] -notmatch '^\s*(\d+)\s*,\s*(\d+)') {
            throw 'nvidia-smi resource query returned an unexpected result'
        }
        return [ordered]@{ utilization_percent = [int]$Matches[1]; memory_used_mib = [int]$Matches[2] }
    } finally {
        Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-Logged {
    param(
        [string]$Name,
        [string]$Program,
        [string[]]$Arguments,
        [int]$TimeoutSeconds = 600,
        [switch]$MonitorResources,
        [switch]$ExpectFailure
    )
    $logPath = Join-Path $reportRoot "$Name.log"
    $stdoutPath = Join-Path $reportRoot "$Name.stdout.log"
    $stderrPath = Join-Path $reportRoot "$Name.stderr.log"
    $resourcePath = Join-Path $reportRoot "$Name.resources.json"
    $argumentLine = ($Arguments | ForEach-Object { Quote-ProcessArgument $_ }) -join ' '
    $process = Start-Process -FilePath $Program -ArgumentList $argumentLine -PassThru `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -WindowStyle Hidden
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $samples = [Collections.Generic.List[object]]::new()
    $lastSampleMilliseconds = -1000.0
    $lastCpuMilliseconds = 0.0
    $lastCpuSampleMilliseconds = 0.0
    $timedOut = $false
    $terminationFailed = $false
    $monitoringErrors = [Collections.Generic.List[string]]::new()
    $gpuMonitoringAvailable = $true
    while (-not $process.WaitForExit(250)) {
        if ($watch.Elapsed.TotalSeconds -ge $TimeoutSeconds) {
            $timedOut = $true
            try {
                $killer = Start-Process -FilePath 'taskkill.exe' -ArgumentList @(
                    '/PID', "$($process.Id)", '/T', '/F'
                ) -PassThru -WindowStyle Hidden
                if (-not $killer.WaitForExit(5000)) {
                    Stop-Process -Id $killer.Id -Force -ErrorAction SilentlyContinue
                    $terminationFailed = $true
                    $monitoringErrors.Add('taskkill exceeded its five-second deadline')
                }
            } catch {
                $terminationFailed = $true
                $monitoringErrors.Add("taskkill failed: $($_.Exception.Message)")
            }
            if (-not $process.WaitForExit(5000)) {
                $terminationFailed = $true
                $monitoringErrors.Add('child process remained alive five seconds after termination attempt')
            }
            break
        }
        if ($MonitorResources -and
            $watch.Elapsed.TotalMilliseconds - $lastSampleMilliseconds -ge 1000.0) {
            $lastSampleMilliseconds = $watch.Elapsed.TotalMilliseconds
            try {
                $process.Refresh()
                $cpuMilliseconds = $process.TotalProcessorTime.TotalMilliseconds
                $elapsedCpuWindow = $watch.Elapsed.TotalMilliseconds - $lastCpuSampleMilliseconds
                $cpuPercent = if ($elapsedCpuWindow -gt 0.0) {
                    100.0 * ($cpuMilliseconds - $lastCpuMilliseconds) /
                        ($elapsedCpuWindow * [Environment]::ProcessorCount)
                } else { 0.0 }
                $lastCpuMilliseconds = $cpuMilliseconds
                $lastCpuSampleMilliseconds = $watch.Elapsed.TotalMilliseconds
                $gpuUtilization = $null
                $gpuMemoryMib = $null
                if ($gpuMonitoringAvailable) {
                    try {
                        $gpuSample = Get-NvidiaResourceSample $reportRoot
                        $gpuUtilization = $gpuSample.utilization_percent
                        $gpuMemoryMib = $gpuSample.memory_used_mib
                    } catch {
                        $gpuMonitoringAvailable = $false
                        $monitoringErrors.Add($_.Exception.Message)
                    }
                }
                $samples.Add([ordered]@{
                    elapsed_seconds = $watch.Elapsed.TotalSeconds
                    process_cpu_percent = $cpuPercent
                    working_set_bytes = $process.WorkingSet64
                    private_memory_bytes = $process.PrivateMemorySize64
                    gpu_utilization_percent = $gpuUtilization
                    gpu_memory_used_mib = $gpuMemoryMib
                })
            } catch {
                $monitoringErrors.Add("process resource sampling failed: $($_.Exception.Message)")
            }
        }
    }
    $watch.Stop()
    $stdout = if (Test-Path -LiteralPath $stdoutPath) {
        Get-Content -LiteralPath $stdoutPath -Raw
    } else { '' }
    $stderr = if (Test-Path -LiteralPath $stderrPath) {
        Get-Content -LiteralPath $stderrPath -Raw
    } else { '' }
    @($stdout, $stderr) | Set-Content -LiteralPath $logPath -Encoding utf8
    if ($stdout) { Write-Host $stdout }
    if ($stderr) { Write-Host $stderr }
    if ($MonitorResources) {
        [ordered]@{
            schema_version = 1
            process_id = $process.Id
            elapsed_seconds = $watch.Elapsed.TotalSeconds
            timed_out = $timedOut
            termination_failed = $terminationFailed
            monitoring_errors = $monitoringErrors
            samples = $samples
        } | ForEach-Object { Write-JsonAtomically $_ $resourcePath 8 }
        $resourceReports.Add($resourcePath)
    }
    if ($timedOut) {
        $terminationMessage = if ($terminationFailed) { ' and termination was not confirmed' } else { '' }
        Add-Step $Name $false $logPath `
            "external watchdog reached $TimeoutSeconds seconds$terminationMessage"
        throw "$Name exceeded its external deadline of $TimeoutSeconds seconds$terminationMessage"
    }
    if ($ExpectFailure) {
        if ($process.ExitCode -eq 0) {
            Add-Step $Name $false $logPath "$Program unexpectedly succeeded"
            throw "$Name unexpectedly succeeded"
        }
        Add-Step $Name $true $logPath "failed closed with exit code $($process.ExitCode) as expected"
        return [ordered]@{ log = $logPath; resources = if ($MonitorResources) { $resourcePath } else { $null } }
    }
    if ($process.ExitCode -ne 0) {
        Add-Step $Name $false $logPath "$Program exited with code $($process.ExitCode)"
        throw "$Name failed with exit code $($process.ExitCode)"
    }
    Add-Step $Name $true $logPath 'completed'
    return [ordered]@{ log = $logPath; resources = if ($MonitorResources) { $resourcePath } else { $null } }
}

function Command-Version {
    param([string]$Program, [string[]]$Arguments)
    $command = Get-Command $Program -ErrorAction SilentlyContinue
    if (-not $command) {
        return $null
    }
    $identifier = [Guid]::NewGuid().ToString('N')
    $stdoutPath = Join-Path $reportRoot ".version-$identifier.stdout"
    $stderrPath = Join-Path $reportRoot ".version-$identifier.stderr"
    try {
        $argumentLine = ($Arguments | ForEach-Object { Quote-ProcessArgument $_ }) -join ' '
        $process = Start-Process -FilePath $command.Source -ArgumentList $argumentLine -PassThru `
            -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath `
            -WindowStyle Hidden
        if (-not $process.WaitForExit(10000)) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            $process.WaitForExit(2000) | Out-Null
            throw "$Program version query exceeded ten seconds"
        }
        $stdout = if (Test-Path -LiteralPath $stdoutPath) {
            Get-Content -LiteralPath $stdoutPath -Raw
        } else { '' }
        $stderr = if (Test-Path -LiteralPath $stderrPath) {
            Get-Content -LiteralPath $stderrPath -Raw
        } else { '' }
        return @($stdout, $stderr) -join "`n"
    } finally {
        Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
    }
}

function Enable-MsvcEnvironment {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
        throw 'vswhere.exe was not found'
    }
    $installationPath = (& $vswhere -latest -products '*' `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath |
        Select-Object -First 1)
    if (-not $installationPath) {
        throw 'A Visual Studio installation with the x64 C++ toolchain was not found'
    }
    $vcvars = Join-Path $installationPath 'VC\Auxiliary\Build\vcvars64.bat'
    $environmentLines = & $env:ComSpec /d /s /c "call `"$vcvars`" >nul && set"
    if ($LASTEXITCODE -ne 0) {
        throw 'Visual Studio x64 environment initialization failed'
    }
    foreach ($line in $environmentLines) {
        $separator = $line.IndexOf('=')
        if ($separator -gt 0) {
            [Environment]::SetEnvironmentVariable($line.Substring(0, $separator),
                $line.Substring($separator + 1), 'Process')
        }
    }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw 'cl.exe is unavailable after Visual Studio environment initialization'
    }
}

function Invoke-DependencyScript {
    param([string]$Name, [string]$Path, [object[]]$Arguments = @())
    $logPath = Join-Path $reportRoot "$Name.log"
    try {
        & $Path @Arguments 2>&1 | Tee-Object -FilePath $logPath | Out-Host
        Add-Step $Name $true $logPath 'repository-local dependency verified'
    } catch {
        Add-Step $Name $false $logPath $_.Exception.Message
        throw
    }
}

$inventory = [ordered]@{}
$inventoryPath = Join-Path $reportRoot 'inventory.json'
try {
    Enable-MsvcEnvironment
    $computer = Get-CimInstance Win32_ComputerSystem
    $operatingSystem = Get-CimInstance Win32_OperatingSystem
    $processors = @(Get-CimInstance Win32_Processor)
    $videoControllers = @(Get-CimInstance Win32_VideoController)
    $inventory.computer = $computer | Select-Object Manufacturer, Model, Name, TotalPhysicalMemory
    $inventory.os = $operatingSystem | Select-Object Caption, Version, BuildNumber, OSArchitecture
    $inventory.processors = @($processors | Select-Object Name, Manufacturer, NumberOfCores,
        NumberOfLogicalProcessors, MaxClockSpeed)
    $inventory.video_controllers = @($videoControllers | Select-Object Name, DriverVersion, AdapterRAM)
    $inventory.cameras = @(Get-PnpDevice -Class Camera, Image -ErrorAction SilentlyContinue |
        Select-Object Status, FriendlyName, InstanceId)
    $inventory.cmake = Command-Version cmake @('--version')
    $inventory.ninja = Command-Version ninja @('--version')
    $inventory.compiler = Command-Version cl @('/Bv')
    $inventory.windows_sdk_versions = @(Get-ChildItem 'C:\Program Files (x86)\Windows Kits\10\Include' `
        -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name)
    $inventory.nvidia_smi = Command-Version nvidia-smi @()
    $inventory.nvidia_gpu_query = Command-Version nvidia-smi @(
        '--query-gpu=name,uuid,driver_version,memory.total,compute_cap,pci.bus_id',
        '--format=csv,noheader')
    $inventory.nvcc = Command-Version nvcc @('--version')
    $inventory.cuda_path = $env:CUDA_PATH
    $inventory.cuda_runtime_dlls = @('cublasLt64_12.dll', 'cublas64_12.dll', 'cufft64_11.dll',
        'cudart64_12.dll') | ForEach-Object {
            [ordered]@{ name = $_; paths = @(where.exe $_ 2>$null) }
        }
    $inventory.expected_cuda_major = 12
    $inventory.expected_cudnn_major = 9
    $inventory.onnxruntime_version = '1.20.1'
    $inventory.actual_machine_name = $computer.Name
    $inventory.expected_machine_name = 'TITAN'
    $inventory.machine_name_matches_expected = $computer.Name -ieq 'TITAN'
    Write-JsonAtomically $inventory $inventoryPath 10
    if (-not $inventory.nvidia_smi) {
        throw 'TITAN qualification requires an NVIDIA driver visible to nvidia-smi'
    }
    if (-not $inventory.machine_name_matches_expected) {
        throw "TITAN qualification cannot run on host '$($computer.Name)'"
    }
    Add-Step 'hardware-inventory' $true $inventoryPath 'actual Windows machine inventory recorded'
    $cudaVersionMatch = [regex]::Match([string]$inventory.nvidia_smi,
        '(?:CUDA|CUDA UMD) Version:\s*(\d+)')
    if (-not $cudaVersionMatch.Success -or [int]$cudaVersionMatch.Groups[1].Value -lt 12) {
        throw 'The NVIDIA driver does not advertise CUDA 12 compatibility'
    }
    Add-Step 'cuda-driver-compatibility' $true $inventoryPath `
        'driver advertises CUDA 12 or newer compatibility'
} catch {
    Add-Step 'hardware-inventory' $false $inventoryPath $_.Exception.Message
    $qualificationErrors.Add($_.Exception.Message)
}

$buildReady = $false
Push-Location $repositoryRoot
try {
    try {
        Invoke-DependencyScript 'dependency-vcpkg' (Join-Path $PSScriptRoot 'bootstrap-vcpkg.ps1')
        $env:VCPKG_ROOT = Join-Path $repositoryRoot '.deps\vcpkg'
        $ninja = Get-ChildItem -LiteralPath (Join-Path $env:VCPKG_ROOT 'downloads\tools') `
            -Filter 'ninja.exe' -File -Recurse -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if (-not $ninja) {
            throw 'Ninja was not found in the repository-local vcpkg tool cache'
        }
        $env:PATH = @($ninja.DirectoryName, $env:PATH) -join ';'
        $inventory.ninja = Command-Version ninja @('--version')
        Invoke-DependencyScript 'dependency-onnxruntime-cpu' `
            (Join-Path $PSScriptRoot 'acquire-onnxruntime.ps1') @('cpu')
        $env:SQUATRIS_ONNXRUNTIME_ROOT = Join-Path $repositoryRoot '.deps\onnxruntime\cpu'
        $env:PATH = @((Join-Path $env:SQUATRIS_ONNXRUNTIME_ROOT 'lib'), $env:PATH) -join ';'
        Invoke-Logged 'configure-onnx-cpu-release' cmake @(
            '--preset', 'windows-onnx-release', '-DSQUATRIS_WARNINGS_AS_ERRORS=ON'
        ) 900 | Out-Null
        Invoke-Logged 'build-onnx-cpu-release' cmake @(
            '--build', '--preset', 'windows-onnx-release'
        ) 1800 | Out-Null
        Invoke-Logged 'onnx-cpu-tests' ctest @(
            '--test-dir', (Join-Path $repositoryRoot 'out\build\windows-onnx-release'),
            '--output-on-failure', '-LE', 'hardware-required'
        ) 1800 | Out-Null

        Invoke-DependencyScript 'dependency-onnxruntime-gpu' `
            (Join-Path $PSScriptRoot 'acquire-onnxruntime.ps1') @('gpu')
        $env:SQUATRIS_ONNXRUNTIME_ROOT = Join-Path $repositoryRoot '.deps\onnxruntime\gpu'
        Invoke-DependencyScript 'dependency-cudnn' (Join-Path $PSScriptRoot 'acquire-cudnn.ps1')
        $cudnnRoot = Join-Path $repositoryRoot '.deps\cudnn\9.1.0.70'
        $cudnnBin = Join-Path $cudnnRoot 'bin'
        $cudaBin = if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH 'bin' } else { $null }
        $env:PATH = (@($cudnnBin, $cudaBin, $env:PATH) | Where-Object { $_ }) -join ';'
        $inventory.cudnn = Get-Content -LiteralPath (Join-Path $cudnnRoot 'squatris-cudnn.json') `
            -Raw | ConvertFrom-Json
        $inventory.cudnn_runtime_dlls = @(Get-ChildItem -LiteralPath $cudnnBin -Filter 'cudnn*.dll' |
            Select-Object Name, FullName, @{Name='Version';Expression={$_.VersionInfo.FileVersion}})
        Write-JsonAtomically $inventory $inventoryPath 10
        Invoke-DependencyScript 'dependency-models' (Join-Path $PSScriptRoot 'acquire-models.ps1')

        Invoke-Logged 'configure-strict-debug' cmake @(
            '--preset', 'windows-cuda-debug', '-DSQUATRIS_WARNINGS_AS_ERRORS=ON'
        ) 900 | Out-Null
        Invoke-Logged 'build-strict-debug' cmake @(
            '--build', '--preset', 'windows-cuda-debug'
        ) 1800 | Out-Null
        Invoke-Logged 'configure-strict-release' cmake @(
            '--preset', 'windows-cuda-release', '-DSQUATRIS_WARNINGS_AS_ERRORS=ON'
        ) 900 | Out-Null
        Invoke-Logged 'build-strict-release' cmake @(
            '--build', '--preset', 'windows-cuda-release'
        ) 1800 | Out-Null
        Invoke-Logged 'automated-tests' ctest @(
            '--test-dir', (Join-Path $repositoryRoot 'out\build\windows-cuda-release'),
            '--output-on-failure', '-LE', 'hardware-required'
        ) 1800 | Out-Null
        Invoke-Logged 'automated-tests-debug' ctest @(
            '--test-dir', (Join-Path $repositoryRoot 'out\build\windows-cuda-debug'),
            '--output-on-failure', '-LE', 'hardware-required'
        ) 1800 | Out-Null
        $buildReady = $true
    } catch {
        $qualificationErrors.Add($_.Exception.Message)
    }

    $benchmark = Join-Path $repositoryRoot `
        'out\build\windows-cuda-release\tools\benchmark\SquatrisPoseBenchmark.exe'
    $poseLab = Join-Path $repositoryRoot `
        'out\build\windows-cuda-release\apps\pose-lab\SquatrisPoseLab.exe'
    $detector = Join-Path $repositoryRoot 'models\detector.onnx'
    $pose = Join-Path $repositoryRoot 'models\pose.onnx'
    $manifest = Join-Path $repositoryRoot 'models\manifest.json'
    $policyManifest = Join-Path $repositoryRoot 'config\rtmpose-cuda-policy.json'

    if ($buildReady) {
        try {
            $cameraResult = Invoke-Logged 'camera-enumeration' $poseLab @('--list-cameras') 180
            $cameraText = Get-Content -LiteralPath $cameraResult.log -Raw
            if ($cameraText -match 'No OpenCV Media Foundation cameras found') {
                throw 'No camera was available for TITAN qualification'
            }
            Add-Step 'camera-negotiated-modes' $true $cameraResult.log `
                'camera indices and probed negotiated modes recorded under an external watchdog'
        } catch {
            $qualificationErrors.Add($_.Exception.Message)
        }

        try {
            $strictReport = Join-Path $reportRoot 'cuda-strict-expected-failure.json'
            Invoke-Logged 'cuda-strict-expected-failure' $benchmark @(
                '--mode', 'deterministic', '--video', $videoPath,
                '--backend', 'rtmpose', '--provider', 'cuda-strict',
                '--warmup', '1', '--iterations', '1', '--deadline-seconds', '300',
                '--detector', $detector, '--pose', $pose, '--manifest', $manifest,
                '--output', $strictReport
            ) 360 -ExpectFailure | Out-Null
            Invoke-Logged 'cuda-strict-failure-report-validation' $benchmark @(
                '--validate-report', $strictReport
            ) 60 | Out-Null
            $benchmarkReports.Add($strictReport)

            $providerReport = Join-Path $reportRoot 'cuda-provider-smoke.json'
            Invoke-Logged 'cuda-provider-session' $benchmark @(
                '--mode', 'deterministic', '--video', $videoPath,
                '--backend', 'rtmpose', '--provider', 'cuda-audited',
                '--warmup', '1', '--iterations', '4', '--deadline-seconds', '300',
                '--detector', $detector, '--pose', $pose, '--manifest', $manifest,
                '--policy-manifest', $policyManifest,
                '--output', $providerReport
            ) 360 -MonitorResources | Out-Null
            Invoke-Logged 'cuda-provider-report-validation' $benchmark @(
                '--validate-report', $providerReport
            ) 60 | Out-Null
            Invoke-Logged 'cuda-provider-corruption-tests' cmake @(
                "-DBENCHMARK=$benchmark", "-DREPORT=$providerReport", '-P',
                (Join-Path $repositoryRoot 'tests\provider_evidence_corruption.cmake')
            ) 120 | Out-Null
            $providerEvidence = Get-Content -LiteralPath $providerReport -Raw | ConvertFrom-Json
            if ($providerEvidence.model.execution_provider -ne `
                    'CUDAExecutionProvider+CPUExecutionProvider' -or
                $providerEvidence.model.strict_provider_assignment -or
                -not $providerEvidence.model.checksums_verified -or
                -not $providerEvidence.model.provider_evidence.verified) {
                throw 'Audited CUDA assignment evidence is incomplete'
            }
            $benchmarkReports.Add($providerReport)
            Add-Step 'cuda-provider-assignment' $true $providerReport `
                'both model sessions passed the pinned audited CUDA policy before measurement'

            $cpuParityReport = Join-Path $reportRoot 'cpu-parity.json'
            Invoke-Logged 'cpu-parity' $benchmark @(
                '--mode', 'deterministic', '--video', $videoPath,
                '--backend', 'rtmpose', '--provider', 'cpu',
                '--warmup', '1', '--iterations', '4', '--deadline-seconds', '300',
                '--detector', $detector, '--pose', $pose, '--manifest', $manifest,
                '--output', $cpuParityReport
            ) 360 | Out-Null
            Invoke-Logged 'cpu-parity-report-validation' $benchmark @(
                '--validate-report', $cpuParityReport
            ) 60 | Out-Null
            $parityReport = Join-Path $reportRoot 'cpu-cuda-parity.json'
            Invoke-Logged 'cpu-cuda-parity-comparison' $benchmark @(
                '--compare-cpu-report', $cpuParityReport,
                '--compare-cuda-report', $providerReport,
                '--parity-output', $parityReport
            ) 120 | Out-Null
            $benchmarkReports.Add($cpuParityReport)
            $benchmarkReports.Add($parityReport)

            $unprofiledOverheadReport = Join-Path $reportRoot 'cuda-unprofiled-overhead.json'
            Invoke-Logged 'cuda-unprofiled-overhead' $benchmark @(
                '--mode', 'deterministic', '--video', $videoPath,
                '--backend', 'rtmpose', '--provider', 'cuda-audited',
                '--warmup', '10', '--iterations', '50', '--deadline-seconds', '600',
                '--detector', $detector, '--pose', $pose, '--manifest', $manifest,
                '--policy-manifest', $policyManifest, '--output', $unprofiledOverheadReport
            ) 660 -MonitorResources | Out-Null
            Invoke-Logged 'cuda-unprofiled-overhead-report-validation' $benchmark @(
                '--validate-report', $unprofiledOverheadReport
            ) 60 | Out-Null
            $benchmarkReports.Add($unprofiledOverheadReport)

            $profiledReport = Join-Path $reportRoot 'cuda-profiled-overhead.json'
            Invoke-Logged 'cuda-profiled-overhead' $benchmark @(
                '--mode', 'deterministic', '--video', $videoPath,
                '--backend', 'rtmpose', '--provider', 'cuda-audited',
                '--profile-production', 'true', '--warmup', '10', '--iterations', '50',
                '--deadline-seconds', '600', '--detector', $detector, '--pose', $pose,
                '--manifest', $manifest, '--policy-manifest', $policyManifest,
                '--output', $profiledReport
            ) 660 -MonitorResources | Out-Null
            Invoke-Logged 'cuda-profiled-overhead-report-validation' $benchmark @(
                '--validate-report', $profiledReport
            ) 60 | Out-Null
            $benchmarkReports.Add($profiledReport)

            foreach ($mode in @('deterministic', 'realtime')) {
                for ($repeat = 1; $repeat -le $BenchmarkRepeats; ++$repeat) {
                    $name = "cuda-$mode-repeat-$repeat"
                    $benchmarkReport = Join-Path $reportRoot "$name.json"
                    Invoke-Logged $name $benchmark @(
                        '--mode', $mode, '--video', $videoPath,
                        '--backend', 'rtmpose', '--provider', 'cuda-audited',
                        '--warmup', "$WarmupIterations", '--iterations', "$MeasuredIterations",
                        '--deadline-seconds', '1800', '--detector', $detector, '--pose', $pose,
                        '--manifest', $manifest, '--policy-manifest', $policyManifest,
                        '--output', $benchmarkReport
                    ) 1860 -MonitorResources | Out-Null
                    Invoke-Logged "$name-report-validation" $benchmark @(
                        '--validate-report', $benchmarkReport
                    ) 60 | Out-Null
                    $benchmarkReports.Add($benchmarkReport)
                }
            }

            $sustainedReport = Join-Path $reportRoot 'cuda-realtime-sustained.json'
            Invoke-Logged 'cuda-realtime-sustained' $benchmark @(
                '--mode', 'realtime', '--video', $videoPath,
                '--backend', 'rtmpose', '--provider', 'cuda-audited',
                '--warmup', "$WarmupIterations", '--iterations', '1',
                '--duration-seconds', "$SustainedSeconds", '--loop-video', 'true',
                '--deadline-seconds', "$($SustainedSeconds + 600)", '--detector', $detector,
                '--pose', $pose, '--manifest', $manifest, '--policy-manifest', $policyManifest,
                '--output', $sustainedReport
            ) ($SustainedSeconds + 660) -MonitorResources | Out-Null
            Invoke-Logged 'cuda-realtime-sustained-report-validation' $benchmark @(
                '--validate-report', $sustainedReport
            ) 60 | Out-Null
            $benchmarkReports.Add($sustainedReport)
        } catch {
            $qualificationErrors.Add($_.Exception.Message)
        }
    }
} finally {
    Pop-Location
}

$completedAt = Get-Date
$profilingOverhead = [ordered]@{ status = 'not-measured' }
if ($unprofiledOverheadReport -and $profiledReport -and
    (Test-Path -LiteralPath $unprofiledOverheadReport) -and
    (Test-Path -LiteralPath $profiledReport)) {
    $unprofiledEvidence = Get-Content -LiteralPath $unprofiledOverheadReport -Raw | ConvertFrom-Json
    $profiledEvidence = Get-Content -LiteralPath $profiledReport -Raw | ConvertFrom-Json
    $unprofiledMedian = [double]$unprofiledEvidence.measured.timings.estimator_end_to_end.median_ms
    $profiledMedian = [double]$profiledEvidence.measured.timings.estimator_end_to_end.median_ms
    $profilingOverhead = [ordered]@{
        status = 'measured'
        unprofiled_report = $unprofiledOverheadReport
        profiled_report = $profiledReport
        unprofiled_median_end_to_end_ms = $unprofiledMedian
        profiled_median_end_to_end_ms = $profiledMedian
        median_overhead_ms = $profiledMedian - $unprofiledMedian
        median_overhead_percent = if ($unprofiledMedian -gt 0.0) {
            100.0 * ($profiledMedian - $unprofiledMedian) / $unprofiledMedian
        } else { $null }
    }
}
$poseLabEvidence = Join-Path $reportRoot 'pose-lab-provider-evidence'
$manualCommand1080 = "& '$poseLab' --camera $CameraIndex --width 1920 --height 1080 --fps 60 --backend rtmpose --provider cuda-audited --detector '$detector' --pose '$pose' --policy-manifest '$policyManifest' --provider-evidence-dir '$poseLabEvidence'"
$manualCommand720 = "& '$poseLab' --camera $CameraIndex --width 1280 --height 720 --fps 60 --backend rtmpose --provider cuda-audited --detector '$detector' --pose '$pose' --policy-manifest '$policyManifest' --provider-evidence-dir '$poseLabEvidence'"
$qualification = [ordered]@{
    schema_version = 2
    machine = $inventory.actual_machine_name
    expected_machine = 'TITAN'
    machine_name_matches_expected = $inventory.machine_name_matches_expected
    started_at = $startedAt.ToUniversalTime().ToString('o')
    completed_at = $completedAt.ToUniversalTime().ToString('o')
    passed = $qualificationErrors.Count -eq 0
    errors = $qualificationErrors
    repository = $repositoryRoot
    qualification_video = $videoPath
    inventory = $inventoryPath
    steps = $steps
    benchmark_reports = $benchmarkReports
    resource_reports = $resourceReports
    profiling_overhead = $profilingOverhead
    physical_camera_observations = [ordered]@{
        status = 'requires-human-participation'
        automated_qualification = $false
        commands = @($manualCommand1080, $manualCommand720)
        checks = @(
            'stand naturally in full-body view',
            'perform slow comfortable squats',
            'perform fast comfortable squats',
            'move laterally left and right',
            'swipe the left and right arms independently',
            'combine lateral movement with arm swipes',
            'temporarily occlude each wrist',
            'temporarily occlude each leg',
            'leave and reenter the camera view',
            'have another person enter the camera view',
            'disconnect the camera and observe failure and shutdown',
            'confirm sustained preview responsiveness and absence of latency accumulation',
            'record delivered FPS, not only requested or negotiated FPS',
            'verify frame-synchronized skeletons and low-confidence landmark behavior'
        )
    }
    global_configuration_changes = @()
}
$qualificationPath = Join-Path $reportRoot 'qualification-report.json'
Write-JsonAtomically $qualification $qualificationPath 14
Write-Output "Qualification report: $qualificationPath"
Write-Output "Physical camera participation remains required: $manualCommand1080"
if ($qualificationErrors.Count -gt 0) {
    Write-Error ($qualificationErrors -join '; ')
    exit 1
}
exit 0
