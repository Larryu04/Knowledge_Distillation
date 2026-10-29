# Resets a Windows machine to "CoreCS was never installed", for repeatable first-install tests.
# Standalone: needs nothing from this repository, so it can be copied to a test laptop on its own.
#
#   1. closes CoreCS and its llama-server,
#   2. uninstalls CoreCS silently (if installed),
#   3. removes %APPDATA%\com.corecs.app (models, chat history, settings) and
#      %LOCALAPPDATA%\com.corecs.app (WebView2 cache), then checks nothing is left.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File reset-test-machine.ps1                  # full reset
#   powershell -ExecutionPolicy Bypass -File reset-test-machine.ps1 -BackupModels    # keep the models aside
#   powershell -ExecutionPolicy Bypass -File reset-test-machine.ps1 -RestoreModels   # put them back
#
# -BackupModels moves the downloaded models (*.gguf + *.corecs.json) to -BackupDir instead of
# deleting them (a same-drive move is instant). -RestoreModels moves them back into the app's
# models folder, after the test, and changes nothing else. -Yes skips the confirmation prompt.
param(
    [switch]$BackupModels,
    [switch]$RestoreModels,
    [string]$BackupDir = (Join-Path $env:USERPROFILE "CoreCS-model-backup"),
    [switch]$Yes
)

$ErrorActionPreference = "Stop"
$roaming   = Join-Path $env:APPDATA "com.corecs.app"
$local     = Join-Path $env:LOCALAPPDATA "com.corecs.app"
$modelsDir = Join-Path $roaming "models"
$uninstallKeys = @(
    "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*",
    "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*",
    "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*"
)

function Get-CoreCSInstall {
    Get-ItemProperty $uninstallKeys -ErrorAction SilentlyContinue |
        Where-Object { $_.DisplayName -eq "CoreCS" } | Select-Object -First 1
}

function Stop-CoreCS {
    Get-Process -Name "corecs", "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Milliseconds 500   # let Windows release file handles on the model files
}

function Get-ModelFiles([string]$dir) {
    if (-not (Test-Path $dir)) { return @() }
    @(Get-ChildItem $dir -File | Where-Object { $_.Name -like "*.gguf" -or $_.Name -like "*.corecs.json" })
}

if ($RestoreModels) {
    $files = Get-ModelFiles $BackupDir
    if ($files.Count -eq 0) { Write-Host "No models in $BackupDir, nothing to restore."; exit 0 }
    Stop-CoreCS
    New-Item -ItemType Directory -Force $modelsDir | Out-Null
    foreach ($f in $files) {
        Move-Item $f.FullName (Join-Path $modelsDir $f.Name) -Force
        Write-Host "Restored $($f.Name)"
    }
    Write-Host "Done. CoreCS will find the models in $modelsDir on its next launch."
    exit 0
}

# ---- show what will happen and confirm ----------------------------------------------------
$install = Get-CoreCSInstall
$models  = Get-ModelFiles $modelsDir
Write-Host "CoreCS test reset"
Write-Host "  Installed app : $(if ($install) { "$($install.DisplayName) $($install.DisplayVersion) -> will be uninstalled" } else { 'not installed' })"
Write-Host "  App data      : $roaming $(if (Test-Path $roaming) { '-> will be removed' } else { '(absent)' })"
Write-Host "  WebView cache : $local $(if (Test-Path $local) { '-> will be removed' } else { '(absent)' })"
foreach ($m in $models) {
    $size = "{0:N1} GB" -f ($m.Length / 1GB)
    $what = if ($BackupModels) { "moved to $BackupDir" } else { "DELETED" }
    Write-Host "  Model         : $($m.Name) ($size) -> $what"
}
if (-not $Yes) {
    $answer = Read-Host "Continue? (y/n)"
    if ($answer -notin @("y", "Y", "yes")) { Write-Host "Cancelled."; exit 1 }
}

# ---- reset --------------------------------------------------------------------------------
Stop-CoreCS

if ($BackupModels -and $models.Count -gt 0) {
    New-Item -ItemType Directory -Force $BackupDir | Out-Null
    foreach ($m in $models) {
        Move-Item $m.FullName (Join-Path $BackupDir $m.Name) -Force
        Write-Host "Backed up $($m.Name)"
    }
}

if ($install) {
    # Tauri's NSIS uninstaller: /S = silent. It relaunches itself from %TEMP%, so wait for the
    # uninstall entry to disappear rather than for the process.
    $exe = ($install.UninstallString -replace '"', '').Trim()
    Write-Host "Uninstalling CoreCS ($exe)"
    Start-Process -FilePath $exe -ArgumentList "/S" -Wait
    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-CoreCSInstall) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 1 }
    if ($install.InstallLocation -and (Test-Path $install.InstallLocation)) {
        Remove-Item -Recurse -Force $install.InstallLocation -ErrorAction SilentlyContinue
    }
}

foreach ($dir in @($roaming, $local)) {
    if (Test-Path $dir) { Remove-Item -Recurse -Force $dir; Write-Host "Removed $dir" }
}

# ---- verify -------------------------------------------------------------------------------
$left = @()
if (Get-CoreCSInstall) { $left += "CoreCS is still listed in Installed apps" }
foreach ($dir in @($roaming, $local)) { if (Test-Path $dir) { $left += "$dir still exists" } }
if ($left.Count -gt 0) {
    Write-Host ""
    Write-Host "NOT clean:" -ForegroundColor Red
    $left | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
}
Write-Host ""
Write-Host "Clean: CoreCS is not installed and has no data on this machine." -ForegroundColor Green
if ($BackupModels -and $models.Count -gt 0) {
    Write-Host "Models kept in $BackupDir (put them back later with -RestoreModels)."
}
