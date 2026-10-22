@echo off
rem Starts the offline tutor: llama-server serves the model AND the chat UI on 127.0.0.1 only.
setlocal
cd /d "%~dp0"
set "PORT=8088"
set "CTX=4096"

set "MODEL="
for %%f in ("%~dp0model\*.gguf") do set "MODEL=%%~ff"
if not defined MODEL for %%f in ("%~dp0..\..\model\*.gguf") do set "MODEL=%%~ff"
if not defined MODEL (
  echo No .gguf model found. Put the model file in "%~dp0model" and run this again.
  pause
  exit /b 1
)

rem The model's profile (training preamble + sampling) travels next to it as <name>.corecs.json.
set "PROFILE=%MODEL:.gguf=.corecs.json%"
if exist "%PROFILE%" copy /y "%PROFILE%" "%~dp0webui\profile.json" >nul

set "SERVER=%~dp0bin\llama-server.exe"
if not exist "%SERVER%" (
  where llama-server >nul 2>nul || (
    echo llama-server was not found. Install llama.cpp ^(winget install llama.cpp^) and run this again.
    pause
    exit /b 1
  )
  set "SERVER=llama-server"
)

echo.
echo   Model: %MODEL%
echo   Your browser opens at http://127.0.0.1:%PORT% once the model is loaded.
echo   Everything runs on this computer. Close this window to stop the tutor.
echo.
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 300;$i++){try{Invoke-WebRequest -UseBasicParsing http://127.0.0.1:%PORT%/health | Out-Null; Start-Process http://127.0.0.1:%PORT%; break}catch{Start-Sleep 1}}"
"%SERVER%" --model "%MODEL%" --host 127.0.0.1 --port %PORT% --ctx-size %CTX% --parallel 1 --jinja --path "%~dp0webui"
if errorlevel 1 pause
