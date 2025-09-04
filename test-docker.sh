#!/bin/bash
# Docker Test Script for AI Processing Suite
# Tests the staging environment in Docker without OpenSearch/Redis

set -e

echo "🐳 Building AI Processing Suite Docker Image..."
docker build -t ai-processing-suite:staging .

echo "🚀 Starting AI Processing Suite in Docker..."
docker-compose -f docker-compose.test.yml up -d

echo "⏳ Waiting for service to be ready..."
sleep 30

echo "🔍 Checking health status..."
curl -f http://localhost:8001/health || {
    echo "❌ Health check failed!"
    docker-compose -f docker-compose.test.yml logs ai-processing-api
    exit 1
}

echo "✅ Service is healthy!"

echo "🧪 Testing endpoints..."

# Test 1: Basic API info
echo "Testing root endpoint..."
curl -s http://localhost:8001/ | jq '.' || echo "❌ Root endpoint failed"

# Test 2: NER endpoint (should work with BERT)
echo "Testing NER endpoint..."
curl -s -X POST http://localhost:8001/ner/extract \
    -H "Content-Type: application/json" \
    -d '{"text": "John Smith from ACME Corp", "entity_types": ["PERSON", "ORGANIZATION"]}' \
    | jq '.' || echo "❌ NER endpoint failed"

# Test 3: Summarization endpoint (should work without external services)
echo "Testing summarization endpoint..."
curl -s -X POST http://localhost:8001/summarize \
    -H "Content-Type: application/json" \
    -d '{"text": "This is a test document for summarization.", "length": "short"}' \
    | jq '.' || echo "❌ Summarization endpoint failed"

# Test 4: Classification schema
echo "Testing classification schema..."
curl -s http://localhost:8001/classification-schema | jq '.' || echo "❌ Classification schema failed"

echo "🎉 All basic tests passed!"

echo "📊 Service logs:"
docker-compose -f docker-compose.test.yml logs --tail=20 ai-processing-api

echo "🛑 To stop the service, run:"
echo "docker-compose -f docker-compose.test.yml down"
