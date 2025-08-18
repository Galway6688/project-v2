@echo off
echo Starting Interactive Multimodal GPT Application...
echo.

echo Starting Frontend (React)...
start cmd /k "npm start"

echo.
echo Waiting 3 seconds before starting backend...
timeout /t 3 /nobreak > nul

echo Starting Backend (FastAPI)...
start cmd /k "cd backend && python run.py"

echo.
echo Both services are starting...
echo Frontend: http://localhost:3000
echo Backend API: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
pause
