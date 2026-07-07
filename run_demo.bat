@echo off
REM tiny-llm local demo launcher.
REM Double-click this file, or run it from a terminal, then open the URL below.
cd /d "%~dp0"

echo.
echo   tiny-llm demo starting...
echo   When you see "Application startup complete", open:
echo.
echo        http://localhost:8000
echo.
echo   Press Ctrl+C in this window to stop the server.
echo.

.venv\Scripts\python.exe -m uvicorn serve.server:app --port 8000
pause
