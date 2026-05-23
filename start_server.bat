@echo off
echo Starting ScoutAI server...
cd /d D:\project\ScoutAI

REM Try port 8001 first; fall back to 8030 if it's stuck
netstat -ano | findstr ":8001 " | findstr LISTENING >nul 2>&1
if %errorlevel%==0 (
    echo Port 8001 is in use - starting on 8030 instead
    python -m uvicorn server:app --host 0.0.0.0 --port 8030 --reload
) else (
    echo Port 8001 is free - starting on 8001
    python -m uvicorn server:app --host 0.0.0.0 --port 8001 --reload
)
