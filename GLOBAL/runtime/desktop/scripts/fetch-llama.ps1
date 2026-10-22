# Downloads the pinned llama.cpp Windows Vulkan build and installs llama-server as the Tauri sidecar.
# Usage (from repo root):  powershell -ExecutionPolicy Bypass -File scripts/fetch-llama.ps1
param([string]$Build = "b11269")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$bin  = Join-Path $root "src-tauri\binaries"
$tmp  = Join-Path $env:TEMP "corecs-llama-$Build"
$zip  = "$tmp.zip"
$url  = "https://github.com/ggml-org/llama.cpp/releases/download/$Build/llama-$Build-bin-win-vulkan-x64.zip"

Write-Host "Downloading $url"
Invoke-WebRequest -Uri $url -OutFile $zip
if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
Expand-Archive -Path $zip -DestinationPath $tmp

New-Item -ItemType Directory -Force $bin | Out-Null
# llama-server.exe is a thin launcher for llama-server-impl.dll; ship only what the server loads.
$keep = @("ggml*.dll", "llama.dll", "llama-common.dll", "llama-server-impl.dll", "mtmd.dll", "libomp.dll", "LICENSE-LLVM-OpenMP")
foreach ($pattern in $keep) { Copy-Item (Join-Path $tmp $pattern) $bin -Force }
# Tauri requires sidecars to carry the target-triple suffix.
Copy-Item (Join-Path $tmp "llama-server.exe") (Join-Path $bin "llama-server-x86_64-pc-windows-msvc.exe") -Force
Set-Content -Path (Join-Path $bin "LLAMA_CPP_VERSION.txt") -Value "$Build-win-vulkan-x64" -Encoding utf8

Remove-Item -Recurse -Force $tmp, $zip
& (Join-Path $bin "llama-server-x86_64-pc-windows-msvc.exe") --version
