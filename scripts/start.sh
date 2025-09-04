#!/bin/bash
"""
Production Startup Script for eDiscovery LLM Retrieval System
Starts all services in the correct order
"""

set -e

echo "🚀 Starting eDiscovery LLM Retrieval System"
echo "=============================================="

# Check if required services are running
check_service() {
    local service_name=$1
    local url=$2
    local timeout=${3:-30}
    
    echo "🔍 Checking $service_name..."
    
    for i in $(seq 1 $timeout); do
        if curl -s $url > /dev/null 2>&1; then
            echo "✅ $service_name is running"
            return 0
        fi
        sleep 1
    done
    
    echo "❌ $service_name is not responding after ${timeout}s"
    return 1
}

# Function to start a service in background
start_service() {
    local service_name=$1
    local command=$2
    local pidfile=$3
    
    echo "🔄 Starting $service_name..."
    
    # Kill existing process if pidfile exists
    if [ -f "$pidfile" ]; then
        local old_pid=$(cat "$pidfile")
        if kill -0 "$old_pid" 2>/dev/null; then
            echo "Stopping existing $service_name (PID: $old_pid)"
            kill "$old_pid"
            sleep 2
        fi
        rm -f "$pidfile"
    fi
    
    # Start new process
    nohup $command > logs/${service_name}.log 2>&1 &
    local pid=$!
    echo $pid > "$pidfile"
    
    echo "✅ $service_name started (PID: $pid)"
}

# Create necessary directories
mkdir -p logs data models cache

# Check Python and dependencies
echo "🐍 Checking Python environment..."
python --version
pip list | grep -E "(fastapi|streamlit|torch|transformers)" || {
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements.txt
}

# Check external services
echo ""
echo "🔍 Checking external services..."

# Check Ollama
if check_service "Ollama" "http://localhost:11434/api/tags" 10; then
    OLLAMA_AVAILABLE=true
else
    echo "⚠️  Ollama not available - LLM features will be limited"
    OLLAMA_AVAILABLE=false
fi

# Check OpenSearch  
if check_service "OpenSearch" "http://localhost:9200" 10; then
    OPENSEARCH_AVAILABLE=true
else
    echo "⚠️  OpenSearch not available - using in-memory storage"
    OPENSEARCH_AVAILABLE=false
fi

# Check Redis
if check_service "Redis" "http://localhost:6379" 5; then
    REDIS_AVAILABLE=true
else
    echo "⚠️  Redis not available - task queue features disabled"
    REDIS_AVAILABLE=false
fi

echo ""

# Start API server
echo "🚀 Starting API server..."
start_service "api-server" "python main.py" "logs/api.pid"

# Wait for API to be ready
echo "⏳ Waiting for API server to initialize..."
if check_service "API Server" "http://localhost:8001/api/v1/health" 30; then
    echo "✅ API server is ready"
else
    echo "❌ API server failed to start"
    exit 1
fi

# Start Streamlit UI
echo "🎨 Starting Streamlit UI..."
start_service "streamlit-ui" "streamlit run streamlit/app.py --server.port 8501 --server.address 0.0.0.0" "logs/streamlit.pid"

# Wait for Streamlit to be ready
echo "⏳ Waiting for Streamlit UI to initialize..."
if check_service "Streamlit UI" "http://localhost:8501/_stcore/health" 20; then
    echo "✅ Streamlit UI is ready"
else
    echo "❌ Streamlit UI failed to start"
fi

echo ""
echo "🎉 eDiscovery LLM Retrieval System is running!"
echo "=============================================="
echo "📊 Service Status:"
echo "  • API Server:    http://localhost:8001"
echo "  • Streamlit UI:  http://localhost:8501"
echo "  • API Docs:      http://localhost:8001/api/v1/docs"
echo ""
echo "🔧 External Services:"
echo "  • Ollama:        $([ "$OLLAMA_AVAILABLE" = true ] && echo "✅ Available" || echo "❌ Not available")"
echo "  • OpenSearch:    $([ "$OPENSEARCH_AVAILABLE" = true ] && echo "✅ Available" || echo "❌ Not available")"
echo "  • Redis:         $([ "$REDIS_AVAILABLE" = true ] && echo "✅ Available" || echo "❌ Not available")"
echo ""
echo "📝 Logs located in: ./logs/"
echo "🛑 To stop services: ./scripts/stop.sh"
echo ""

# Keep script running and monitor services
echo "🔍 Monitoring services... (Press Ctrl+C to stop)"
while true; do
    sleep 30
    
    # Check API health
    if ! curl -s http://localhost:8001/api/v1/health > /dev/null 2>&1; then
        echo "⚠️  API server health check failed"
    fi
    
    # Check Streamlit health  
    if ! curl -s http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        echo "⚠️  Streamlit UI health check failed"
    fi
done
