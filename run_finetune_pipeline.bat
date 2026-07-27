@echo off
REM Windows entry point: launches run_finetune_pipeline.sh inside WSL Ubuntu.
REM Same launch pattern as run_master.bat. Run run_master.bat at least once
REM first (sets up .venv + CheXpert data); this one adds the fine-tuning
REM comparison on top of that.
REM
REM Optional: set N_ABLATION, N_AGENTIC, N_FINETUNE_STUDIES, EPOCHS,
REM RUN_FULL_FT as Windows/PowerShell environment variables before running,
REM e.g.:
REM     $env:N_ABLATION = "1000"
REM     $env:N_AGENTIC = "50"
REM     $env:N_FINETUNE_STUDIES = "300"
REM     $env:EPOCHS = "1"
REM     $env:RUN_FULL_FT = "1"   # set to "0" to skip full fine-tuning if it OOMs
REM     .\run_finetune_pipeline.bat

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

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

echo Launching run_finetune_pipeline.sh inside WSL Ubuntu...
echo Repo path in WSL: %WSL_DIR%
echo.

wsl bash -lc "cd '%WSL_DIR%' && N_ABLATION='%N_ABLATION%' N_AGENTIC='%N_AGENTIC%' N_FINETUNE_STUDIES='%N_FINETUNE_STUDIES%' EPOCHS='%EPOCHS%' RUN_FULL_FT='%RUN_FULL_FT%' bash run_finetune_pipeline.sh"

echo.
echo === run_finetune_pipeline.sh exited with code %ERRORLEVEL% ===
pause
