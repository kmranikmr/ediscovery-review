#!/bin/bash

# Stop Services Script
# Run this script to stop all running services

echo "🛑 Stopping LLM Retrieval System services..."

# Stop FastAPI
echo "🔥 Stopping FastAPI server..."
pkill -f "python.*main.py" || echo "No FastAPI process found"

# Stop Streamlit  
echo "🎨 Stopping Streamlit app..."
pkill -f "streamlit.*run" || echo "No Streamlit process found"

# Wait for processes to stop
sleep 3

# Check if processes are still running
if pgrep -f "python.*main.py" > /dev/null; then
    echo "⚠️  FastAPI process still running, force killing..."
    pkill -9 -f "python.*main.py"
fi

if pgrep -f "streamlit.*run" > /dev/null; then
    echo "⚠️  Streamlit process still running, force killing..."
    pkill -9 -f "streamlit.*run"
fi

echo "✅ All services stopped successfully!"

# Show port status
echo ""
echo "📊 Port status:"
if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port 8001 still in use:"
    lsof -Pi :8001 -sTCP:LISTEN
else
    echo "✅ Port 8001 is free"
fi

if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port 8501 still in use:"
    lsof -Pi :8501 -sTCP:LISTEN
else
    echo "✅ Port 8501 is free"
fi
