@echo off
REM Windows entry point: launches run_master.sh inside WSL Ubuntu, where all
REM the actual tools (python, torch, azcopy, nvidia-smi) live. Double-click
REM this file, or run it from cmd.exe.
REM
REM Optional: set MODEL=qwen2-vl (or llava-1.5-7b), N_ABLATION, N_AGENTIC as
REM Windows environment variables before running to skip the interactive
REM model-selection prompt, e.g.:
REM     set MODEL=qwen2-vl
REM     run_master.bat

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0

where wsl >nul 2>nul
if errorlevel 1 (
    echo WSL was not found on this machine. Install it first: wsl --install
    pause
    exit /b 1
)

for /f "usebackq delims=" %%i in (`wsl wslpath -a "%SCRIPT_DIR%"`) do set WSL_DIR=%%i

if "%WSL_DIR%"=="" (
    echo Could not resolve a WSL path for this folder.
    pause
    exit /b 1
)

echo Launching run_master.sh inside WSL Ubuntu...
echo Repo path in WSL: %WSL_DIR%
echo.

wsl bash -lc "cd '%WSL_DIR%' && MODEL='%MODEL%' N_ABLATION='%N_ABLATION%' N_AGENTIC='%N_AGENTIC%' bash run_master.sh"

echo.
echo === run_master.sh exited with code %ERRORLEVEL% ===
pause
