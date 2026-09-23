@echo off
setlocal
cd /d "%~dp0"
set "PYEXE=%CD%\.venv\Scripts\python.exe"
if not exist "%PYEXE%" (
  if exist "..\gpu_v51_elite_v3\.venv\Scripts\python.exe" (
    set "PYEXE=..\gpu_v51_elite_v3\.venv\Scripts\python.exe"
  ) else (
    echo ERROR: no V4 or sibling V3 virtual environment found.
    exit /b 2
  )
)
"%PYEXE%" validate_v4.py %*
exit /b %errorlevel%
