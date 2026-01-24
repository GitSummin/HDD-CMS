@echo off
setlocal EnableExtensions

set "CHECKPOINT=output\checkpoints\run_01\epoch_50.pth"
set "OUTPUT_DIR=output\evaluation\run_01"
set "TEST_FILE=data\fire_ms_dataset_dataset.csv"
set "CHECKPOINT_DIR=output\checkpoints\run_01"
set "N_OUTPUTS=5"
set "RADIUS=1"

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

python test.py "%CHECKPOINT%" "%OUTPUT_DIR%" ^
  --test_file "%TEST_FILE%" ^
  --checkpoint_dir "%CHECKPOINT_DIR%" ^
  --n_outputs %N_OUTPUTS% ^
  --radius %RADIUS%

if errorlevel 1 (
  echo [ERROR] Evaluation failed.
  exit /b 1
)

echo [INFO] Evaluation finished.
pause
endlocal
