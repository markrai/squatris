param(
    [Parameter(Mandatory = $true)]
    [string]$Video,
    [string]$OutputDirectory,
    [int]$CameraIndex = 0,
    [int]$WarmupIterations = 100,
    [int]$MeasuredIterations = 2000
)

$ErrorActionPreference = 'Stop'
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
    throw "Refusing to use a filesystem root for qualification output"
}
New-Item -ItemType Directory -Path $reportRoot -Force | Out-Null

$steps = [Collections.Generic.List[object]]::new()
$qualificationError = $null
$startedAt = Get-Date

function Add-Step {
    param([string]$Name, [bool]$Passed, [string]$Evidence, [string]$Message)
    $steps.Add([ordered]@{ name = $Name; passed = $Passed; evidence = $Evidence; message = $Message })
}

function Invoke-Logged {
    param([string]$Name, [string]$Program, [string[]]$Arguments)
    $logPath = Join-Path $reportRoot "$Name.log"
    & $Program @Arguments 2>&1 | Tee-Object -FilePath $logPath | Out-Host
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Add-Step $Name $false $logPath "$Program exited with code $exitCode"
        throw "$Name failed with exit code $exitCode"
    }
    Add-Step $Name $true $logPath 'completed'
    return $logPath
}

function Command-Version {
    param([string]$Program, [string[]]$Arguments)
    $command = Get-Command $Program -ErrorAction SilentlyContinue
    if (-not $command) {
        return $null
    }
    return ((& $Program @Arguments 2>&1) -join "`n")
}

$inventory = [ordered]@{}
try {
    $inventory.computer_name = $env:COMPUTERNAME
    $inventory.os = Get-CimInstance Win32_OperatingSystem |
        Select-Object Caption, Version, BuildNumber, OSArchitecture
    $inventory.processor = Get-CimInstance Win32_Processor |
        Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
    $inventory.video_controllers = @(Get-CimInstance Win32_VideoController |
        Select-Object Name, DriverVersion, AdapterRAM)
    $inventory.cameras = @(Get-PnpDevice -Class Camera, Image -ErrorAction SilentlyContinue |
        Select-Object Status, FriendlyName, InstanceId)
    $inventory.cmake = Command-Version cmake @('--version')
    $inventory.ninja = Command-Version ninja @('--version')
    $inventory.compiler = Command-Version cl @()
    $inventory.nvidia_smi = Command-Version nvidia-smi @()
    $inventory.nvcc = Command-Version nvcc @('--version')
    $inventory.cudnn9_locations = @(where.exe cudnn64_9.dll 2>$null)
    $inventory.expected_cuda_major = 12
    $inventory.expected_cudnn_major = 9
    $inventory.onnxruntime_version = '1.20.1'
    $inventoryPath = Join-Path $reportRoot 'inventory.json'
    $inventory | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $inventoryPath -Encoding utf8
    if (-not $inventory.nvidia_smi) {
        Add-Step 'hardware-inventory' $false $inventoryPath 'nvidia-smi is unavailable'
        throw 'TITAN qualification requires an NVIDIA driver visible to nvidia-smi'
    }
    Add-Step 'hardware-inventory' $true $inventoryPath 'inventory recorded without changing global configuration'
    $cudaVersionMatch = [regex]::Match([string]$inventory.nvidia_smi, 'CUDA Version:\s*(\d+)')
    if (-not $cudaVersionMatch.Success -or [int]$cudaVersionMatch.Groups[1].Value -lt 12) {
        Add-Step 'cuda-cudnn-compatibility-preflight' $false $inventoryPath 'driver does not advertise CUDA 12 or newer compatibility'
        throw 'The NVIDIA driver does not advertise the CUDA compatibility required by ONNX Runtime 1.20.1'
    }
    Add-Step 'cuda-cudnn-compatibility-preflight' $true $inventoryPath 'driver compatibility recorded; cuDNN 9 loading is proven by the strict CUDA session step'

    Push-Location $repositoryRoot
    try {
        $vcpkgLog = Join-Path $reportRoot 'dependency-vcpkg.log'
        & (Join-Path $repositoryRoot 'scripts\bootstrap-vcpkg.ps1') 2>&1 | Tee-Object -FilePath $vcpkgLog
        $vcpkgRoot = Join-Path $repositoryRoot '.deps\vcpkg'
        $env:VCPKG_ROOT = $vcpkgRoot
        Add-Step 'dependency-vcpkg' $true $vcpkgLog 'pinned vcpkg baseline verified'

        $ortLog = Join-Path $reportRoot 'dependency-onnxruntime-gpu.log'
        & (Join-Path $repositoryRoot 'scripts\acquire-onnxruntime.ps1') -Variant gpu 2>&1 |
            Tee-Object -FilePath $ortLog
        $ortRoot = Join-Path $repositoryRoot '.deps\onnxruntime\gpu'
        $env:SQUATRIS_ONNXRUNTIME_ROOT = $ortRoot
        Add-Step 'dependency-onnxruntime-gpu' $true $ortLog 'pinned GPU runtime verified in an isolated directory'

        $modelLog = Join-Path $reportRoot 'dependency-models.log'
        & (Join-Path $repositoryRoot 'scripts\acquire-models.ps1') 2>&1 |
            Tee-Object -FilePath $modelLog
        Add-Step 'dependency-models' $true $modelLog 'authoritative manifest checksums verified'

        Invoke-Logged 'configure-strict-release' cmake @(
            '--preset', 'windows-cuda-release',
            '-DSQUATRIS_ENABLE_HARDWARE_TESTS=ON',
            "-DSQUATRIS_HARDWARE_VIDEO=$videoPath",
            '-DSQUATRIS_HARDWARE_PROVIDER=cuda',
            '-DSQUATRIS_WARNINGS_AS_ERRORS=ON'
        ) | Out-Null
        Invoke-Logged 'build-strict-release' cmake @('--build', '--preset', 'windows-cuda-release') |
            Out-Null
        Invoke-Logged 'automated-tests' ctest @(
            '--test-dir', (Join-Path $repositoryRoot 'out\build\windows-cuda-release'),
            '--output-on-failure', '-LE', 'hardware-required'
        ) | Out-Null

        $benchmark = Join-Path $repositoryRoot 'out\build\windows-cuda-release\tools\benchmark\SquatrisPoseBenchmark.exe'
        $poseLab = Join-Path $repositoryRoot 'out\build\windows-cuda-release\apps\pose-lab\SquatrisPoseLab.exe'
        $detector = Join-Path $repositoryRoot 'models\detector.onnx'
        $pose = Join-Path $repositoryRoot 'models\pose.onnx'
        $manifest = Join-Path $repositoryRoot 'models\manifest.json'
        $providerReport = Join-Path $reportRoot 'cuda-provider-smoke.json'
        $benchmarkReport = Join-Path $reportRoot 'cuda-prerecorded-benchmark.json'

        Invoke-Logged 'cuda-provider-session' $benchmark @(
            '--mode', 'deterministic', '--backend', 'rtmpose', '--provider', 'cuda',
            '--warmup', '2', '--iterations', '5', '--deadline-seconds', '300',
            '--detector', $detector, '--pose', $pose, '--manifest', $manifest,
            '--output', $providerReport
        ) | Out-Null
        Invoke-Logged 'cuda-provider-report-validation' $benchmark @('--validate-report', $providerReport) |
            Out-Null
        $providerEvidence = Get-Content -LiteralPath $providerReport -Raw | ConvertFrom-Json
        if ($providerEvidence.model.execution_provider -ne 'CUDAExecutionProvider' -or
            -not $providerEvidence.model.strict_provider_assignment -or
            -not $providerEvidence.model.checksums_verified) {
            Add-Step 'cuda-provider-assignment' $false $providerReport 'CUDA assignment evidence is incomplete'
            throw 'CUDA provider assignment could not be verified'
        }
        Add-Step 'cuda-provider-assignment' $true $providerReport 'strict CUDA session initialized with verified models'

        Invoke-Logged 'cuda-prerecorded-benchmark' $benchmark @(
            '--mode', 'deterministic', '--video', $videoPath,
            '--backend', 'rtmpose', '--provider', 'cuda',
            '--warmup', "$WarmupIterations", '--iterations', "$MeasuredIterations",
            '--deadline-seconds', '1800', '--detector', $detector, '--pose', $pose,
            '--manifest', $manifest, '--output', $benchmarkReport
        ) | Out-Null
        Invoke-Logged 'cuda-benchmark-report-validation' $benchmark @('--validate-report', $benchmarkReport) |
            Out-Null

        $cameraLog = Invoke-Logged 'camera-enumeration' $poseLab @('--list-cameras')
        $cameraText = Get-Content -LiteralPath $cameraLog -Raw
        if ($cameraText -match 'No OpenCV Media Foundation cameras found') {
            Add-Step 'camera-negotiated-modes' $false $cameraLog 'no camera was enumerated'
            throw 'No camera was available for TITAN qualification'
        }
        Add-Step 'camera-negotiated-modes' $true $cameraLog 'camera indices and probed negotiated modes recorded'
    } finally {
        Pop-Location
    }
} catch {
    $qualificationError = $_.Exception.Message
}

$completedAt = Get-Date
$manualCommand = "& '$repositoryRoot\out\build\windows-cuda-release\apps\pose-lab\SquatrisPoseLab.exe' --camera $CameraIndex --backend rtmpose --provider cuda --detector '$repositoryRoot\models\detector.onnx' --pose '$repositoryRoot\models\pose.onnx'"
$qualification = [ordered]@{
    schema_version = 1
    machine = 'TITAN'
    started_at = $startedAt.ToUniversalTime().ToString('o')
    completed_at = $completedAt.ToUniversalTime().ToString('o')
    passed = ($null -eq $qualificationError)
    error = $qualificationError
    repository = $repositoryRoot
    qualification_video = $videoPath
    steps = $steps
    physical_camera_observations = [ordered]@{
        status = 'requires-human-participation'
        automated_qualification = $false
        command = $manualCommand
        checks = @(
            'full-body framing and negotiated live capture rate',
            'anatomical left and right under mirrored preview',
            'projected 2D knee-angle behavior through standing and squatting',
            'occlusion and recovery',
            'physical disconnect behavior',
            'sustained responsiveness and latency accumulation'
        )
    }
    global_configuration_changes = @()
}
$qualificationPath = Join-Path $reportRoot 'qualification-report.json'
$qualification | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $qualificationPath -Encoding utf8
Write-Output "Qualification report: $qualificationPath"
Write-Output "Physical camera participation remains required: $manualCommand"
if ($qualificationError) {
    Write-Error $qualificationError
    exit 1
}
exit 0
