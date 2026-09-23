@echo off
setlocal
cd /d "%~dp0"
set PIP_DISABLE_PIP_VERSION_CHECK=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1

set "PYEXE=%CD%\.venv\Scripts\python.exe"
if not exist "%PYEXE%" (
  if exist "..\gpu_v51_elite_v3\.venv\Scripts\python.exe" (
    set "PYEXE=..\gpu_v51_elite_v3\.venv\Scripts\python.exe"
    echo Reusing V3 virtual environment.
  ) else (
    py -3.12 -m venv .venv 2>nul
    if errorlevel 1 py -3.11 -m venv .venv 2>nul
    if errorlevel 1 (
      echo ERROR: Python 3.11 or 3.12 is required.
      exit /b 2
    )
    set "PYEXE=%CD%\.venv\Scripts\python.exe"
  )
)

"%PYEXE%" -c "import torch; assert torch.cuda.is_available()" >nul 2>nul
if errorlevel 1 (
  echo Installing PyTorch CUDA...
  "%PYEXE%" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
  if errorlevel 1 exit /b %errorlevel%
)

"%PYEXE%" -m pip install -q -r requirements_v4.txt
if errorlevel 1 exit /b %errorlevel%

"%PYEXE%" -c "import sys,pathlib; sys.path.insert(0,'vendor'); import kaggle_environments as k; p=pathlib.Path(k.__file__).resolve(); assert 'vendor' in p.parts; print('Vendored Kaggriculture engine:',k.__version__)"
if errorlevel 1 exit /b %errorlevel%

echo.
echo ============================================================
echo WINNER V4 exact-v51 training
echo SEARCH -^> AMPLIFY -^> EXPLOIT+EXPLORE
echo Ctrl+C = stop safely. Run again = auto resume.
echo ============================================================
echo.
"%PYEXE%" winner_train.py %*
exit /b %errorlevel%
