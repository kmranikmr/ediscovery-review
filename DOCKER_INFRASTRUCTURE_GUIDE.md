# AI Processing Suite - Docker Setup & Infrastructure Guide

## Table of Contents
1. [Quick Setup](#quick-setup)
2. [Environment Configuration](#environment-configuration)
3. [Docker Deployment Options](#docker-deployment-options)
4. [Testing & Validation](#testing--validation)
5. [Troubleshooting](#troubleshooting)
6. [Production Deployment](#production-deployment)
7. [Performance Optimization](#performance-optimization)
8. [Monitoring & Maintenance](#monitoring--maintenance)

## Quick Setup

### Prerequisites
- Docker 20.10+ and Docker Compose 2.0+
- 8GB+ RAM (16GB recommended for full functionality)
- 10GB+ free disk space

### 1-Minute Setup
```bash
# Clone and navigate to staging directory
cd /home/bbk/anil/from164/src/llm-retrieval-system/staging

# Build and start (automated testing)
./test-docker.sh
```

### Manual Setup
```bash
# Build the image
docker build -t ai-processing-suite:staging .

# Start with docker-compose
docker-compose -f docker-compose.test.yml up

# Or start in background
docker-compose -f docker-compose.test.yml up -d
```

### Verify Installation
```bash
# Health check
curl http://localhost:8001/health

# Quick API test
curl -X POST http://localhost:8001/ner/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "John Smith from ACME Corp", "entity_types": ["PERSON", "ORGANIZATION"]}'
```

## Environment Configuration

### Docker Environment Variables

#### Core Settings
```bash
# Skip external services (for testing)
SKIP_OPENSEARCH=true        # Disable OpenSearch dependency
SKIP_REDIS=true            # Disable Redis dependency
SKIP_OLLAMA=false          # Keep Ollama enabled (default: false)
DOCKER_ENV=true            # Enable Docker-specific configurations

# Service URLs (Docker networking)
OLLAMA_BASE_URL=http://host.docker.internal:11434
OPENSEARCH_HOST=host.docker.internal
REDIS_HOST=host.docker.internal

# Model Configuration
DEFAULT_MODEL=mistral      # Default Ollama model
MODEL_TEMPERATURE=0.1      # LLM temperature setting
MAX_TOKENS=2048           # Maximum token limit

# API Configuration
API_HOST=0.0.0.0          # Bind to all interfaces in container
API_PORT=8001             # Internal container port
LOG_LEVEL=INFO            # Logging verbosity
```

#### Development vs Production
```bash
# Development (testing without external services)
SKIP_OPENSEARCH=true
SKIP_REDIS=true
DEBUG_MODE=true
LOG_LEVEL=DEBUG

# Production (full functionality)
SKIP_OPENSEARCH=false
SKIP_REDIS=false
DEBUG_MODE=false
LOG_LEVEL=INFO
OPENSEARCH_HOST=opensearch.internal
REDIS_HOST=redis.internal
```

### Dockerfile Configuration

The Dockerfile includes all necessary environment variables:

```dockerfile
# Environment variables for service control
ENV SKIP_OPENSEARCH=true
ENV SKIP_REDIS=true
ENV DOCKER_ENV=true
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

# API configuration
ENV API_HOST=0.0.0.0
ENV API_PORT=8001

# Model settings
ENV DEFAULT_MODEL=mistral
ENV MODEL_TEMPERATURE=0.1
```

## Docker Deployment Options

### Option 1: Test Configuration (Minimal Dependencies)

**File: `docker-compose.test.yml`**
```yaml
version: '3.8'

services:
  ai-processing-api:
    build: .
    ports:
      - "8001:8001"
    environment:
      - SKIP_OPENSEARCH=true
      - SKIP_REDIS=true
      - DOCKER_ENV=true
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

**Usage:**
```bash
docker-compose -f docker-compose.test.yml up
```

### Option 2: Development Configuration (With External Services)

**File: `docker-compose.dev.yml`**
```yaml
version: '3.8'

services:
  ai-processing-api:
    build: .
    ports:
      - "8001:8001"
    environment:
      - SKIP_OPENSEARCH=false
      - SKIP_REDIS=false
      - DOCKER_ENV=true
      - OPENSEARCH_HOST=opensearch
      - REDIS_HOST=redis
    depends_on:
      - opensearch
      - redis
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data

  opensearch:
    image: opensearchproject/opensearch:2.0.0
    environment:
      - discovery.type=single-node
      - plugins.security.disabled=true
    ports:
      - "9200:9200"
    volumes:
      - opensearch_data:/usr/share/opensearch/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  opensearch_data:
  redis_data:
```

### Option 3: Production Configuration (Full Stack)

**File: `docker-compose.prod.yml`**
```yaml
version: '3.8'

services:
  ai-processing-api:
    image: ai-processing-suite:production
    ports:
      - "8001:8001"
    environment:
      - SKIP_OPENSEARCH=false
      - SKIP_REDIS=false
      - DOCKER_ENV=true
      - OPENSEARCH_HOST=opensearch
      - REDIS_HOST=redis
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - opensearch
      - redis
      - ollama
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: always
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 8G
          cpus: '4'

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  opensearch:
    image: opensearchproject/opensearch:2.0.0
    environment:
      - discovery.type=single-node
      - plugins.security.disabled=false
      - OPENSEARCH_JAVA_OPTS=-Xms2g -Xmx2g
    volumes:
      - opensearch_data:/usr/share/opensearch/data
    deploy:
      resources:
        limits:
          memory: 4G

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    deploy:
      resources:
        limits:
          memory: 1G

volumes:
  opensearch_data:
  redis_data:
  ollama_models:
```

## Testing & Validation

### Automated Testing Script

**`test-docker.sh`** - Comprehensive testing automation:

```bash
#!/bin/bash
set -e

echo "🐳 Building AI Processing Suite Docker Image..."
docker build -t ai-processing-suite:staging .

echo "🚀 Starting AI Processing Suite in Docker..."
docker-compose -f docker-compose.test.yml up -d

echo "⏳ Waiting for service to be ready..."
sleep 30

echo "🔍 Running health checks..."
curl -f http://localhost:8001/health || exit 1

echo "🧪 Testing core endpoints..."

# Test NER
curl -s -X POST http://localhost:8001/ner/extract \
    -H "Content-Type: application/json" \
    -d '{"text": "John Smith from ACME Corp", "entity_types": ["PERSON", "ORGANIZATION"]}' \
    | jq '.' || echo "❌ NER test failed"

# Test Summarization
curl -s -X POST http://localhost:8001/summarize \
    -H "Content-Type: application/json" \
    -d '{"text": "Test document for summarization.", "length": "short"}' \
    | jq '.' || echo "❌ Summarization test failed"

# Test Classification Schema
curl -s http://localhost:8001/classification-schema | jq '.' || echo "❌ Schema test failed"

echo "✅ All tests passed!"
echo "📊 Service logs:"
docker-compose -f docker-compose.test.yml logs --tail=20 ai-processing-api
```

### Manual Testing Procedures

#### 1. Health and Status Checks
```bash
# Basic health
curl http://localhost:8001/health

# API information
curl http://localhost:8001/ | jq '.'

# Available endpoints
curl http://localhost:8001/ | jq '.endpoints'
```

#### 2. Core Functionality Tests
```bash
# NER Test
curl -X POST http://localhost:8001/ner/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Sarah Johnson from TechCorp contacted legal@company.com about the merger.",
    "entity_types": ["PERSON", "ORGANIZATION", "EMAIL"]
  }' | jq '.result.entities'

# Classification Test
curl -X POST http://localhost:8001/classify \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [{"content": "Confidential merger agreement between parties."}],
    "response_format": "standard"
  }' | jq '.result'

# Simple QA Test (if index exists)
curl -X POST http://localhost:8001/qa-simple \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is this document about?",
    "collection_id": "default"
  }' | jq '.result'
```

#### 3. Performance Tests
```bash
# Load test with multiple concurrent requests
for i in {1..10}; do
  curl -X POST http://localhost:8001/ner/extract \
    -H "Content-Type: application/json" \
    -d '{"text": "Test document '$i'", "entity_types": ["PERSON"]}' &
done
wait
```

### Test Data and Examples

#### Sample Documents for Testing
```bash
# Create test data directory
mkdir -p test-data

# Sample email for Family QA
cat > test-data/sample-email.json << 'EOF'
{
  "query": "What was decided about the project timeline?",
  "documents": [
    {
      "content": "From: pm@company.com\nTo: team@company.com\nSubject: Project Timeline Update\n\nBased on our discussion, we're moving the deadline to March 15th to ensure quality delivery.",
      "meta": {
        "type": "email",
        "participants": ["pm@company.com", "team@company.com"],
        "date": "2024-02-01"
      }
    }
  ]
}
EOF

# Test with sample data
curl -X POST http://localhost:8001/qa/family \
  -H "Content-Type: application/json" \
  -d @test-data/sample-email.json
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Service Won't Start

**Problem:** Container fails to start or exits immediately
```bash
# Check container logs
docker-compose -f docker-compose.test.yml logs ai-processing-api

# Check container status
docker-compose -f docker-compose.test.yml ps
```

**Solutions:**
- Check port availability: `lsof -i :8001`
- Verify Docker resources: `docker system df`
- Restart with fresh build: `docker-compose down && docker build --no-cache -t ai-processing-suite:staging .`

#### 2. Health Check Fails

**Problem:** `/health` endpoint returns errors
```bash
# Check detailed logs
docker-compose -f docker-compose.test.yml logs --tail=50 ai-processing-api

# Check if Ollama is accessible
docker exec -it staging_ai-processing-api_1 curl http://host.docker.internal:11434/api/version
```

**Solutions:**
- Verify Ollama is running on host: `curl http://localhost:11434/api/version`
- Check firewall settings for port 11434
- Increase container memory limits

#### 3. NER Functionality Issues

**Problem:** NER endpoints return errors
```bash
# Test BERT-based NER
curl -X POST http://localhost:8001/ner/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Test", "method": "bert"}'

# Test LLM-based NER fallback
curl -X POST http://localhost:8001/ner/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Test", "method": "llm"}'
```

**Solutions:**
- Check if BERT models are loaded: Look for model download logs
- Verify sufficient memory for ML models
- Test with smaller text inputs

#### 4. External Service Connection Issues

**Problem:** OpenSearch/Redis connection failures (when enabled)
```bash
# Check service connectivity
docker-compose -f docker-compose.dev.yml logs opensearch
docker-compose -f docker-compose.dev.yml logs redis

# Test direct connection
docker exec -it staging_ai-processing-api_1 curl http://opensearch:9200/_cluster/health
```

**Solutions:**
- Use test configuration: `SKIP_OPENSEARCH=true SKIP_REDIS=true`
- Check service startup order in docker-compose
- Verify network connectivity between containers

#### 5. Performance Issues

**Problem:** Slow response times or timeouts
```bash
# Monitor resource usage
docker stats

# Check container logs for performance warnings
docker-compose -f docker-compose.test.yml logs | grep -i "slow\|timeout\|memory"
```

**Solutions:**
- Increase container memory: Edit docker-compose resource limits
- Optimize model settings: Reduce `MAX_TOKENS`, increase `MODEL_TEMPERATURE`
- Use GPU acceleration for Ollama (production)

### Debug Mode

Enable debug mode for detailed logging:
```bash
# Start with debug enabled
DOCKER_DEBUG=true docker-compose -f docker-compose.test.yml up

# Or modify environment in docker-compose.test.yml
environment:
  - DEBUG_MODE=true
  - LOG_LEVEL=DEBUG
```

### Log Analysis

```bash
# View real-time logs
docker-compose -f docker-compose.test.yml logs -f ai-processing-api

# Search for specific errors
docker-compose -f docker-compose.test.yml logs ai-processing-api | grep -i error

# Export logs for analysis
docker-compose -f docker-compose.test.yml logs ai-processing-api > api-logs.txt
```

## Production Deployment

### Security Configuration

#### 1. Environment Security
```bash
# Use secrets for sensitive data
echo "your-api-key" | docker secret create api_key -

# Secure environment file
cat > .env.prod << 'EOF'
# Production API Configuration
API_HOST=0.0.0.0
API_PORT=8001
LOG_LEVEL=INFO

# External Services
OPENSEARCH_HOST=opensearch.internal.company.com
REDIS_HOST=redis.internal.company.com
OLLAMA_BASE_URL=http://ollama.internal.company.com:11434

# Security
SSL_ENABLED=true
API_KEY_REQUIRED=true
RATE_LIMIT_ENABLED=true
EOF
```

#### 2. Network Security
```yaml
# docker-compose.prod.yml network configuration
networks:
  internal:
    driver: bridge
    internal: true
  external:
    driver: bridge

services:
  ai-processing-api:
    networks:
      - internal
      - external
    ports:
      - "8001:8001"
```

### Load Balancing and Scaling

#### 1. Multi-Instance Deployment
```yaml
# docker-compose.prod.yml
services:
  ai-processing-api:
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
        max_attempts: 3
```

#### 2. Load Balancer Configuration
```yaml
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - ai-processing-api
```

### Backup and Recovery

#### 1. Data Backup
```bash
# Backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)

# Backup OpenSearch data
docker run --rm -v staging_opensearch_data:/data -v $(pwd)/backups:/backup \
  alpine tar czf /backup/opensearch_$DATE.tar.gz -C /data .

# Backup Redis data
docker run --rm -v staging_redis_data:/data -v $(pwd)/backups:/backup \
  alpine tar czf /backup/redis_$DATE.tar.gz -C /data .

# Backup application logs
docker-compose -f docker-compose.prod.yml logs > backups/logs_$DATE.txt
```

#### 2. Disaster Recovery
```bash
# Recovery script
#!/bin/bash
BACKUP_DATE=$1

# Stop services
docker-compose -f docker-compose.prod.yml down

# Restore data
docker run --rm -v staging_opensearch_data:/data -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/opensearch_$BACKUP_DATE.tar.gz -C /data

docker run --rm -v staging_redis_data:/data -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/redis_$BACKUP_DATE.tar.gz -C /data

# Restart services
docker-compose -f docker-compose.prod.yml up -d
```

## Performance Optimization

### Resource Optimization

#### 1. Memory Management
```yaml
# Optimized resource allocation
services:
  ai-processing-api:
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 4G
          cpus: '2'
    environment:
      - PYTHON_GC_THRESHOLD=700,10,10
      - OMP_NUM_THREADS=4
```

#### 2. Model Optimization
```bash
# Environment variables for model optimization
MODEL_QUANTIZATION=true
MODEL_CACHE_SIZE=2GB
MAX_CONCURRENT_REQUESTS=10
BATCH_SIZE=8
```

### Caching Configuration

#### 1. Redis Caching (Production)
```yaml
  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
```

#### 2. Application-Level Caching
```python
# Cache configuration in application
CACHE_CONFIG = {
    "model_cache_ttl": 3600,
    "response_cache_ttl": 300,
    "max_cache_size": "1GB"
}
```

## Monitoring & Maintenance

### Health Monitoring

#### 1. Health Check Endpoints
```bash
# Automated health monitoring script
#!/bin/bash
while true; do
  if ! curl -f http://localhost:8001/health > /dev/null 2>&1; then
    echo "$(date): Health check failed - restarting service"
    docker-compose -f docker-compose.prod.yml restart ai-processing-api
  fi
  sleep 30
done
```

#### 2. Performance Monitoring
```bash
# Monitor resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

# Monitor API response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8001/health
```

### Log Management

#### 1. Log Rotation
```yaml
# docker-compose.prod.yml
services:
  ai-processing-api:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
```

#### 2. Centralized Logging
```yaml
  fluentd:
    image: fluent/fluentd:v1.14
    volumes:
      - ./fluentd.conf:/fluentd/etc/fluent.conf
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
```

### Maintenance Procedures

#### 1. Regular Updates
```bash
# Update script
#!/bin/bash
echo "Starting maintenance update..."

# Pull latest images
docker-compose -f docker-compose.prod.yml pull

# Rolling update
docker-compose -f docker-compose.prod.yml up -d --no-deps ai-processing-api

# Verify health
sleep 30
curl -f http://localhost:8001/health || exit 1

echo "Update completed successfully"
```

#### 2. Cleanup Procedures
```bash
# Cleanup script
#!/bin/bash

# Remove old containers
docker container prune -f

# Remove unused images
docker image prune -f

# Remove unused volumes (careful!)
docker volume prune -f

# Remove unused networks
docker network prune -f
```

---

## Quick Reference Commands

```bash
# Start test environment
./test-docker.sh

# Start development environment  
docker-compose -f docker-compose.dev.yml up

# Start production environment
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose logs -f ai-processing-api

# Stop and clean
docker-compose down
docker system prune -f

# Health check
curl http://localhost:8001/health

# API information
curl http://localhost:8001/ | jq '.'
```

For detailed API usage and integration examples, see the Complete API Documentation.
