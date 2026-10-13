@echo off
REM ============================================================
REM  run_facetrack.bat
REM  Restart-on-crash wrapper for FaceTrack Pro.
REM  If ".venv\Scripts\python.exe" main.py exits for ANY reason (crash, error, power
REM  event recovery), this waits 10 seconds and restarts it —
REM  instead of the process just staying dead until someone
REM  notices and manually re-runs it.
REM
REM  Run this directly to test it manually, or point a Task
REM  Scheduler task at it (see TASK_SCHEDULER_SETUP.md) so it
REM  also survives full machine reboots without anyone logging in.
REM ============================================================

cd /d "%~dp0"
if not exist logs mkdir logs

:loop
echo [%date% %time%] Starting FaceTrack Pro... >> logs\restart_wrapper.log
call .venv\Scripts\activate.bat
".venv\Scripts\python.exe" main.py
echo [%date% %time%] FaceTrack Pro exited (code %ERRORLEVEL%). Restarting in 10s... >> logs\restart_wrapper.log
timeout /t 10 /nobreak
goto loop
