@echo off
setlocal
python src\main.py --generate-synthetic --output outputs
if errorlevel 1 exit /b %errorlevel%
python src\main.py --use-synthetic-detections --output outputs
exit /b %errorlevel%
