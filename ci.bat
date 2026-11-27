@echo off
setlocal
where pwsh.exe >nul 2>&1
if errorlevel 1 (
    echo PowerShell 7 ^(pwsh.exe^) is required. 1>&2
    exit /b 1
)
pwsh.exe -NoLogo -NoProfile -File "%~dp0scripts\ci.ps1" %*
exit /b %ERRORLEVEL%
