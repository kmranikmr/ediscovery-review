#!/bin/bash

# Local Development Script
# Run this script to start the application locally

set -e

echo "🚀 Starting LLM Retrieval System locally..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create logs directory
mkdir -p logs

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    if [ -f ".env.template" ]; then
        cp .env.template .env
        echo "📄 Please edit .env file with your configuration"
    else
        cat > .env <<EOF
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here

# OpenSearch Configuration  
OPENSEARCH_ENDPOINT=http://localhost:9200

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# Debug Mode
DEBUG_MODE=true

# Docker Environment Flags
SKIP_OPENSEARCH=false
SKIP_OLLAMA=false
EOF
        echo "📄 Created .env file. Please edit it with your configuration."
    fi
fi

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  Port $1 is already in use"
        return 1
    fi
    return 0
}

# Check ports
echo "🔍 Checking ports..."
if ! check_port 8001; then
    echo "Please stop the service using port 8001 or kill the process:"
    lsof -Pi :8001 -sTCP:LISTEN
    exit 1
fi

if ! check_port 8501; then
    echo "Please stop the service using port 8501 or kill the process:"
    lsof -Pi :8501 -sTCP:LISTEN  
    exit 1
fi

# Start FastAPI in background
echo "🔥 Starting FastAPI server on port 8001..."
nohup python main.py > logs/main.log 2>&1 &
FASTAPI_PID=$!

# Wait for FastAPI to start
echo "⏳ Waiting for FastAPI to start..."
sleep 5

# Check if FastAPI started successfully
if ! ps -p $FASTAPI_PID > /dev/null; then
    echo "❌ FastAPI failed to start. Check logs/main.log for details."
    cat logs/main.log
    exit 1
fi

# Test FastAPI health
echo "🩺 Testing FastAPI health..."
if curl -s http://localhost:8001/ > /dev/null; then
    echo "✅ FastAPI is running on http://localhost:8001"
else
    echo "❌ FastAPI health check failed"
    exit 1
fi

# Start Streamlit
echo "🎨 Starting Streamlit on port 8501..."
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0 &
STREAMLIT_PID=$!

# Wait for Streamlit to start
sleep 5

echo ""
echo "🎉 LLM Retrieval System started successfully!"
echo ""
echo "🌐 Access your applications:"
echo "   📡 FastAPI API: http://localhost:8001"
echo "   🎨 Streamlit UI: http://localhost:8501"
echo ""
echo "📋 Process IDs:"
echo "   FastAPI: $FASTAPI_PID"
echo "   Streamlit: $STREAMLIT_PID"
echo ""
echo "📊 To view logs:"
echo "   FastAPI: tail -f logs/main.log"
echo "   Streamlit: Use the terminal where this script is running"
echo ""
echo "🛑 To stop services:"
echo "   kill $FASTAPI_PID $STREAMLIT_PID"
echo "   OR run: pkill -f 'python.*main.py' && pkill -f 'streamlit.*run'"

# Keep the script running to show Streamlit output
wait $STREAMLIT_PID
