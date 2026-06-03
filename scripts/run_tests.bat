@echo off
setlocal
python -m pytest tests
if errorlevel 1 exit /b %errorlevel%
python -m compileall src tests
if errorlevel 1 exit /b %errorlevel%
python -m py_compile dashboard.py
exit /b %errorlevel%
