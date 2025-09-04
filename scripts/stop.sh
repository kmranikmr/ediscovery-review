#!/bin/bash
"""
Stop script for eDiscovery LLM Retrieval System
Gracefully stops all services
"""

echo "🛑 Stopping eDiscovery LLM Retrieval System"
echo "==========================================="

# Function to stop a service
stop_service() {
    local service_name=$1
    local pidfile=$2
    
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "🛑 Stopping $service_name (PID: $pid)"
            kill "$pid"
            
            # Wait for graceful shutdown
            for i in {1..10}; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                echo "⚡ Force killing $service_name"
                kill -9 "$pid"
            fi
            
            echo "✅ $service_name stopped"
        else
            echo "⚠️  $service_name was not running"
        fi
        rm -f "$pidfile"
    else
        echo "⚠️  No pidfile found for $service_name"
    fi
}

# Stop services
stop_service "Streamlit UI" "logs/streamlit.pid"
stop_service "API Server" "logs/api.pid"

# Clean up any remaining processes
echo "🧹 Cleaning up remaining processes..."
pkill -f "streamlit run" 2>/dev/null || true
pkill -f "python main.py" 2>/dev/null || true

echo "✅ All services stopped"
