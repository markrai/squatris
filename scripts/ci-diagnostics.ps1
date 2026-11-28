function Get-CiDiagnostics {
    param(
        [AllowEmptyCollection()][string[]]$Lines,
        [string]$RepositoryRoot
    )
    $seen = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $diagnostics = [System.Collections.Generic.List[string]]::new()
    $pattern = '(?i)(?:\b(?:fatal\s+)?error(?:\s+[A-Z]+\d+)?\s*:|^\s*CMake Error\b|^\s*\[\s*FAILED\s*\]|:\d+:\s*Failure\b|^\s*\d+\s+-\s+.*\((?:Failed|Timeout|Not Run|SEGFAULT|Subprocess aborted|Exception[^)]*)\)\s*$|^\s*No tests were found\b|^\s*Errors while running CTest\b)'
    foreach ($entry in $Lines) {
        $line = $entry.Trim()
        if ($line -notmatch $pattern) {
            continue
        }
        if ($RepositoryRoot) {
            $prefix = $RepositoryRoot.TrimEnd('\', '/') + '\'
            $line = $line.Replace($prefix, '', [StringComparison]::OrdinalIgnoreCase)
            $line = $line.Replace($prefix.Replace('\', '/'), '', [StringComparison]::OrdinalIgnoreCase)
        }
        if ($seen.Add($line)) {
            $diagnostics.Add($line)
        }
    }
    if ($diagnostics.Count -eq 0) {
        $diagnostics.Add('No recognized diagnostic; last output lines:')
        foreach ($entry in @($Lines | Select-Object -Last 15)) {
            $diagnostics.Add($entry)
        }
    }
    return $diagnostics.ToArray()
}
