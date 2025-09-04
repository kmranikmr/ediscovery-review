#!/bin/bash
"""
Test runner script for eDiscovery LLM Retrieval System
"""

echo "🧪 Running eDiscovery LLM Retrieval System Tests"
echo "==============================================="

# Check if API is running
check_api() {
    if curl -s http://localhost:8001/api/v1/health > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Start API if not running
if ! check_api; then
    echo "🚀 Starting API server for testing..."
    python main.py &
    API_PID=$!
    
    # Wait for API to be ready
    echo "⏳ Waiting for API to start..."
    for i in {1..30}; do
        if check_api; then
            echo "✅ API server is ready"
            break
        fi
        sleep 1
    done
    
    if ! check_api; then
        echo "❌ Failed to start API server"
        kill $API_PID 2>/dev/null
        exit 1
    fi
else
    echo "✅ API server is already running"
    API_PID=""
fi

# Run tests
echo ""
echo "🧪 Running comprehensive tests..."
python -m pytest tests/test_comprehensive.py -v

# Quick smoke test
echo ""
echo "⚡ Running quick smoke test..."
python tests/test_comprehensive.py

# Cleanup
if [ ! -z "$API_PID" ]; then
    echo "🛑 Stopping test API server..."
    kill $API_PID 2>/dev/null
fi

echo "✅ Testing completed"
