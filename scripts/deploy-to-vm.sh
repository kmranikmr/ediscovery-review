#!/bin/bash

# Simple deployment script for existing Azure VM
# Run this script on your Azure VM to deploy the application

set -e

echo "🚀 Deploying LLM Retrieval System to existing Azure VM..."

# Get current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"

echo "📁 Application directory: $APP_DIR"
cd "$APP_DIR"

# Stop existing services gracefully
echo "🛑 Stopping existing services..."
pkill -f "python.*main.py" || echo "No FastAPI process found"
pkill -f "streamlit.*run" || echo "No Streamlit process found"
sleep 3

# Backup current deployment if it exists
if [ -f "main.py" ]; then
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    echo "💾 Creating backup: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    cp -r . "$BACKUP_DIR/" 2>/dev/null || true
fi

# Activate virtual environment or create if doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and install dependencies
echo "📚 Installing/updating dependencies..."
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found, installing basic packages..."
    pip install fastapi uvicorn streamlit pydantic python-multipart requests
fi

# Create logs directory
mkdir -p logs

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📄 Creating .env file from template..."
    if [ -f ".env.template" ]; then
        cp .env.template .env
    else
        cat > .env <<EOF
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here

# OpenSearch Configuration
OPENSEARCH_ENDPOINT=http://localhost:9200

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# Debug Mode
DEBUG_MODE=false

# Azure VM Configuration
SKIP_OPENSEARCH=true
SKIP_OLLAMA=false
EOF
    fi
    echo "⚠️ Please edit .env file with your actual configuration!"
fi

# Start FastAPI server
echo "🔥 Starting FastAPI server..."
nohup python main.py > logs/main.log 2>&1 &
FASTAPI_PID=$!
echo "FastAPI PID: $FASTAPI_PID"

# Wait for FastAPI to start
echo "⏳ Waiting for FastAPI to start..."
for i in {1..30}; do
    if curl -s http://localhost:8001/ > /dev/null 2>&1; then
        echo "✅ FastAPI is running"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ FastAPI failed to start. Check logs:"
        tail -n 20 logs/main.log
        exit 1
    fi
    sleep 2
done

# Start Streamlit app
echo "🎨 Starting Streamlit application..."
nohup streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0 > logs/streamlit.log 2>&1 &
STREAMLIT_PID=$!
echo "Streamlit PID: $STREAMLIT_PID"

# Wait for Streamlit to start
echo "⏳ Waiting for Streamlit to start..."
sleep 10

# Final status check
echo ""
echo "🔍 Final status check..."

if pgrep -f "python.*main.py" > /dev/null; then
    echo "✅ FastAPI server is running (PID: $(pgrep -f 'python.*main.py'))"
else
    echo "❌ FastAPI server is not running"
    echo "Recent FastAPI logs:"
    tail -n 10 logs/main.log
fi

if pgrep -f "streamlit.*run" > /dev/null; then
    echo "✅ Streamlit app is running (PID: $(pgrep -f 'streamlit.*run'))"
else
    echo "❌ Streamlit app is not running"
    echo "Recent Streamlit logs:"
    tail -n 10 logs/streamlit.log
fi

# Get VM IP address
VM_IP=$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')

echo ""
echo "🎉 Deployment completed!"
echo ""
echo "🌐 Access your applications:"
echo "   📡 FastAPI API: http://$VM_IP:8001"
echo "   🎨 Streamlit UI: http://$VM_IP:8501"
echo "   📚 API Docs: http://$VM_IP:8001/docs"
echo ""
echo "📊 Monitor your services:"
echo "   FastAPI logs: tail -f $APP_DIR/logs/main.log"
echo "   Streamlit logs: tail -f $APP_DIR/logs/streamlit.log"
echo ""
echo "🛑 To stop services:"
echo "   pkill -f 'python.*main.py' && pkill -f 'streamlit.*run'"
echo ""

# Save process IDs for easy management
cat > pids.txt <<EOF
FASTAPI_PID=$FASTAPI_PID
STREAMLIT_PID=$STREAMLIT_PID
DEPLOYED_AT=$(date)
VM_IP=$VM_IP
EOF

echo "💾 Process IDs saved to pids.txt"
