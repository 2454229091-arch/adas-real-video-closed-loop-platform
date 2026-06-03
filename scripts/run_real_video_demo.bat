@echo off
setlocal
if "%~1"=="" (
  echo Usage: scripts\run_real_video_demo.bat data\raw\your_video.mp4
  exit /b 1
)
python src\main.py --video "%~1" --output outputs\real_video_demo
exit /b %errorlevel%
