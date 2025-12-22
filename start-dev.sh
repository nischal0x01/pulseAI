#!/bin/bash

# Development startup script for Cuffless Blood Pressure System

set -e

echo "🫀 Starting Cuffless Blood Pressure Development Environment"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed"
    exit 1
fi

# Function to start backend
start_backend() {
    echo "📡 Starting Mock ESP32 WebSocket Server..."
    
    # Check if requirements are installed
    if ! python3 -c "import fastapi, uvicorn" &> /dev/null; then
        echo "📦 Installing Python dependencies..."
        pip3 install -r requirements.txt
    fi
    
    # Start the backend server
    echo "🚀 Backend server starting on http://localhost:8080"
    python3 mock_esp32_server.py &
    BACKEND_PID=$!
    echo "Backend PID: $BACKEND_PID"
}

# Function to start frontend
start_frontend() {
    echo "🖥️  Starting Next.js Frontend..."
    
    cd frontend
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        echo "📦 Installing Node.js dependencies..."
        if command -v pnpm &> /dev/null; then
            pnpm install
        else
            npm install
        fi
    fi
    
    # Start the frontend server
    echo "🚀 Frontend server starting on http://localhost:3000"
    if command -v pnpm &> /dev/null; then
        pnpm dev &
    else
        npm run dev &
    fi
    FRONTEND_PID=$!
    echo "Frontend PID: $FRONTEND_PID"
    
    cd ..
}

# Function to cleanup on exit
cleanup() {
    echo
    echo "🛑 Shutting down development servers..."
    
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
        echo "✅ Backend server stopped"
    fi
    
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
        echo "✅ Frontend server stopped"
    fi
    
    echo "👋 Development environment stopped"
    exit 0
}

# Setup signal handlers
trap cleanup SIGINT SIGTERM

# Parse command line arguments
case "${1:-both}" in
    "backend")
        start_backend
        echo "📡 Backend server running. Press Ctrl+C to stop."
        wait $BACKEND_PID
        ;;
    "frontend")
        start_frontend
        echo "🖥️  Frontend server running. Press Ctrl+C to stop."
        wait $FRONTEND_PID
        ;;
    "both"|*)
        start_backend
        sleep 2  # Give backend time to start
        start_frontend
        
        echo
        echo "✅ Both servers are running:"
        echo "   📡 Backend:  http://localhost:8080"
        echo "   🖥️  Frontend: http://localhost:3000"
        echo
        echo "🔗 Connect your application to ws://localhost:8080/signals"
        echo "📊 View the web interface at http://localhost:8080"
        echo
        echo "Press Ctrl+C to stop all servers"
        
        # Wait for both processes
        wait $BACKEND_PID $FRONTEND_PID
        ;;
esac
