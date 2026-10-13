@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo FaceTrack - Recorded Video Test
echo ============================================================
echo.
echo 1. Put testing.mp4 in this folder (or edit video_path in config.py)
echo 2. Set input_mode = "video" in config.py
 echo.
".venv\Scripts\python.exe" main.py
pause
