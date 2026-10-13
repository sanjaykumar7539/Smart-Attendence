@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
	".venv\Scripts\python.exe" run_students.py %*
) else (
	python run_students.py %*
)
pause
