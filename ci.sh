#!/usr/bin/env bash
set -euo pipefail
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) ;;
  *) printf '%s\n' 'Squatris CI requires Windows x64 and MSVC. Run ci.bat or use Git Bash on Windows.' >&2; exit 1 ;;
esac
if ! command -v pwsh.exe >/dev/null 2>&1; then
  printf '%s\n' 'PowerShell 7 (pwsh.exe) is required.' >&2
  exit 1
fi
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec pwsh.exe -NoLogo -NoProfile -File "$(cygpath -w "$root/scripts/ci.ps1")" "$@"
