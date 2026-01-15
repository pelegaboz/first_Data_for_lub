@echo off
title ADHD Data Analysis System
color 0A

echo ======================================================
echo   ADHD Data Analysis System
echo   Starting automated analysis...
echo ======================================================
echo.

REM Try multiple Python commands
py main.py
if %ERRORLEVEL% NEQ 0 (
    python main.py
)
if %ERRORLEVEL% NEQ 0 (
    python3 main.py
)

echo.
echo ======================================================
echo   Analysis Complete
echo ======================================================
pause