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

# llama-server links the VC++ runtime dynamically and the llama.cpp release does not ship it. Copy it
# from the local Visual Studio redist folder (app-local deployment), so the installed app also runs
# on a clean Windows without the VC++ Redistributable. corecs.exe itself links it statically.
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -property installationPath
$crt = Get-ChildItem (Join-Path $vs "VC\Redist\MSVC\*\x64\Microsoft.VC14*.CRT") -Directory |
    Sort-Object { [version]$_.Parent.Parent.Name } | Select-Object -Last 1
if (-not $crt) { throw "VC++ redist folder not found under $vs (install the MSVC build tools)." }
foreach ($dll in @("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll")) {
    Copy-Item (Join-Path $crt.FullName $dll) $bin -Force
}
Write-Host "Bundled VC++ runtime from $($crt.FullName)"

Remove-Item -Recurse -Force $tmp, $zip
& (Join-Path $bin "llama-server-x86_64-pc-windows-msvc.exe") --version
