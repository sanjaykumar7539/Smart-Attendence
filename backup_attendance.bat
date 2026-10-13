@echo off
REM ============================================================
REM  backup_attendance.bat
REM  Copies attendance.db to a timestamped file in .\backups
REM  Meant to run once a day via Task Scheduler (see
REM  TASK_SCHEDULER_SETUP.md). Keeps the last 30 days' worth —
REM  older backups are deleted automatically.
REM ============================================================

cd /d D:\Projects\facetrack_project_hp
if not exist backups mkdir backups
if not exist logs mkdir logs

REM %date% format depends on Windows locale settings — use PowerShell for a
REM reliable ISO date regardless of regional settings (India uses dd-mm-yyyy,
REM which would silently break simple %date% string-slicing).
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%i

copy /Y database\attendance.db backups\attendance_%TODAY%.db >> logs\backup.log 2>&1
echo [%date% %time%] Backed up attendance.db to backups\attendance_%TODAY%.db >> logs\backup.log

REM Delete backups older than 30 days
forfiles /p backups /m attendance_*.db /d -30 /c "cmd /c del @path" 2>nul
