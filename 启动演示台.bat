@echo off
chcp 65001 >nul
title Voice Steganography Demo
cd /d "%~dp0"
set "PY=C:\Users\ILLENIUM\.workbuddy\binaries\python\envs\hinet\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] Python environment not found:
  echo %PY%
  pause
  exit /b 1
)
set "PYTHONPATH="
set "PYTHONIOENCODING=utf-8"
"%PY%" -u launch.py
echo.
echo Demo stopped. Press any key to close.
pause >nul
