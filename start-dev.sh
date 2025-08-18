#!/bin/bash

echo "Starting Interactive Multimodal GPT Application..."
echo

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "Port $1 is already in use"
        return 1
    else
        return 0
    fi
}

# Check if ports are available
if ! check_port 3000; then
    echo "Frontend port 3000 is already in use. Please stop the service or use a different port."
    exit 1
fi

if ! check_port 8000; then
    echo "Backend port 8000 is already in use. Please stop the service or use a different port."
    exit 1
fi

echo "Starting Frontend (React)..."
npm start &
FRONTEND_PID=$!

echo "Waiting 3 seconds before starting backend..."
sleep 3

echo "Starting Backend (FastAPI)..."
cd backend
python run.py &
BACKEND_PID=$!

echo
echo "Both services are starting..."
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo
echo "To stop both services, press Ctrl+C"

# Wait for both processes
wait $FRONTEND_PID $BACKEND_PID
