#!/bin/bash

echo "Starting RamanSPy MAX..."

# Function to kill processes on exit
cleanup() {
    echo "Stopping servers..."
    kill $(jobs -p)
    exit
}
trap cleanup SIGINT SIGTERM

# Check if backend is setup
if [ ! -d "backend" ]; then
    echo "Backend directory not found!"
    exit 1
fi

# Start Backend
echo "Starting Backend on http://localhost:8000..."
cd backend
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
cd ..

# Start Frontend
echo "Starting Frontend on http://localhost:3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "Application running! Open http://localhost:3000 in your browser."
wait $BACKEND_PID $FRONTEND_PID
