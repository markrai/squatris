$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\ci-diagnostics.ps1')

$root = 'C:\dev\project\squatris'
$clang = "$root\tools\benchmark\report.cpp:507:14: error: constness prevents automatic move [performance-no-automatic-move,-warnings-as-errors]"
$msvc = "$root\core\pose.cpp(17,2): error C2065: undeclared identifier"
$cmake = 'CMake Error at CMakeLists.txt:13 (find_package):'
$gtest = "$root\tests\unit\pose_tests.cpp:19: Failure"
$ctest = '  12 - squatris-benchmark-smoke (Failed)'
$lines = @(
    '[58/68] Building CXX object tests\test.obj',
    'FAILED: tests/test.obj',
    '"cmake.exe" -E __run_co_compile',
    $clang,
    '144 warnings generated.',
    '3 warnings treated as errors',
    $clang,
    $msvc,
    $cmake,
    $gtest,
    '[  FAILED  ] PoseTest.Fixture',
    $ctest,
    'ninja: build stopped: subcommand failed.'
)
$actual = @(Get-CiDiagnostics -Lines $lines -RepositoryRoot $root)
$expected = @(
    'tools\benchmark\report.cpp:507:14: error: constness prevents automatic move [performance-no-automatic-move,-warnings-as-errors]',
    'core\pose.cpp(17,2): error C2065: undeclared identifier',
    $cmake,
    'tests\unit\pose_tests.cpp:19: Failure',
    '[  FAILED  ] PoseTest.Fixture',
    $ctest.Trim()
)
if (($actual -join "`n") -cne ($expected -join "`n")) {
    throw "CI diagnostic extraction mismatch: $($actual -join '; ')"
}
$fallback = @(Get-CiDiagnostics -Lines @('unrecognized failure') -RepositoryRoot $root)
if ($fallback.Count -ne 2 -or $fallback[1] -ne 'unrecognized failure') {
    throw 'CI diagnostic fallback lost original failure output'
}
Write-Host 'PASS: CI diagnostic extraction fixtures'
