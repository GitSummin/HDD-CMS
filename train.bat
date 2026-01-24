@echo off
setlocal EnableExtensions

set "DATASET="

set "TRAIN_FILE=data\unlabeled_train_ms_dataset.csv"
set "TEST_FILE=data\fire_ms_dataset_dataset.csv"

set "RADIUS=1"
set "DIM=50"
set "LAYER_HIDDEN=6"
set "BATCH_TRAIN=256"
set "BATCH_TEST=256"
set "LR=1e-3"
set "EPOCHS=200"
set "N_OUTPUTS=5"

set "CHECKPOINT_DIR=output\checkpoints\run_01"

set "NOISE_MODE=hybrid_gamma"
set "LOSS_MODE=wass_to_kl"

if not exist "%CHECKPOINT_DIR%" mkdir "%CHECKPOINT_DIR%"

echo ============================================
echo [INFO] DATASET        = %DATASET%
echo [INFO] TRAIN_FILE     = %TRAIN_FILE%
echo [INFO] TEST_FILE      = %TEST_FILE%
echo [INFO] RADIUS         = %RADIUS%
echo [INFO] DIM            = %DIM%
echo [INFO] LAYER_HIDDEN   = %LAYER_HIDDEN%
echo [INFO] BATCH_TRAIN    = %BATCH_TRAIN%
echo [INFO] BATCH_TEST     = %BATCH_TEST%
echo [INFO] LR             = %LR%
echo [INFO] EPOCHS         = %EPOCHS%
echo [INFO] N_OUTPUTS      = %N_OUTPUTS%
echo [INFO] CHECKPOINT_DIR = %CHECKPOINT_DIR%
echo [INFO] NOISE_MODE     = %NOISE_MODE%
echo [INFO] LOSS_MODE      = %LOSS_MODE%
echo ============================================

set "NOISE_MODE=%NOISE_MODE%"
set "LOSS_MODE=%LOSS_MODE%"

python train.py ^
  "%DATASET%" ^
  %RADIUS% ^
  %DIM% ^
  %LAYER_HIDDEN% ^
  %BATCH_TRAIN% ^
  %BATCH_TEST% ^
  %LR% ^
  %EPOCHS% ^
  "%CHECKPOINT_DIR%" ^
  --train_file "%TRAIN_FILE%" ^
  --test_file "%TEST_FILE%" ^
  --n_outputs %N_OUTPUTS%

if errorlevel 1 (
  echo [ERROR] Training failed.
  exit /b 1
)

echo [INFO] Training finished.
pause
endlocal
