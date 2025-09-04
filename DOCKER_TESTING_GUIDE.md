# Docker Testing Guide for AI Processing Suite

## Overview
This guide covers testing the AI Processing Suite in Docker without external dependencies (Redis/OpenSearch).

## Environment Variables
The Docker setup uses these environment variables to skip external services:
- `SKIP_OPENSEARCH=true` - Disables OpenSearch dependency
- `SKIP_REDIS=true` - Disables Redis dependency  
- `DOCKER_ENV=true` - Enables Docker-specific configurations

## Quick Start

### 1. Build and Run
```bash
# From the staging directory
cd /home/bbk/anil/from164/src/llm-retrieval-system/staging

# Build Docker image
docker build -t ai-processing-suite:staging .

# Start with docker-compose
docker-compose -f docker-compose.test.yml up
```

### 2. Automated Testing
```bash
# Run the automated test script
./test-docker.sh
```

### 3. Manual Testing
Once the service is running, test endpoints manually:

```bash
# Health check
curl http://localhost:8001/health

# API info
curl http://localhost:8001/

# NER extraction
curl -X POST http://localhost:8001/ner/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "John Smith from ACME Corp", "entity_types": ["PERSON", "ORGANIZATION"]}'

# Summarization
curl -X POST http://localhost:8001/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test document for summarization.", "length": "short"}'
```

## Available Endpoints in Docker Mode

### Core Endpoints (Work without external services)
- `/` - API information
- `/health` - Health check
- `/ner/extract` - Named Entity Recognition (uses BERT)
- `/summarize` - Text summarization 
- `/classification-schema` - Available classification types
- `/classify` - Document classification
- `/parse` - Document parsing (basic text extraction)

### Limited Functionality (May require external services)
- `/qa/ask` - Question Answering (basic functionality)
- `/family-qa/ask` - Family QA (limited without vector search)
- `/search` - Full-text search (limited without OpenSearch)
- `/analytics` - Analytics (limited without Redis)

## Expected Behavior

### What Works
✅ Document parsing and text extraction  
✅ Named Entity Recognition with BERT  
✅ Text classification  
✅ Basic summarization  
✅ Schema endpoints  
✅ Health checks  

### What's Limited
⚠️ Vector similarity search (no OpenSearch)  
⚠️ Session management (no Redis)  
⚠️ Advanced analytics (no Redis)  
⚠️ Document indexing (no OpenSearch)  

## Troubleshooting

### Service Won't Start
```bash
# Check logs
docker-compose -f docker-compose.test.yml logs ai-processing-api

# Check if port is in use
lsof -i :8001

# Restart with fresh build
docker-compose -f docker-compose.test.yml down
docker build --no-cache -t ai-processing-suite:staging .
docker-compose -f docker-compose.test.yml up
```

### Health Check Fails
```bash
# Check service status
docker-compose -f docker-compose.test.yml ps

# View detailed logs
docker-compose -f docker-compose.test.yml logs --tail=50 ai-processing-api
```

### Memory Issues
```bash
# Increase Docker memory limits in docker-compose.test.yml
# Add under 'ai-processing-api' service:
deploy:
  resources:
    limits:
      memory: 4G
    reservations:
      memory: 2G
```

## Environment Validation

The application should detect and log these settings on startup:
```
INFO: SKIP_OPENSEARCH=true - OpenSearch disabled
INFO: SKIP_REDIS=true - Redis disabled  
INFO: DOCKER_ENV=true - Docker mode enabled
INFO: Using in-memory storage for session management
INFO: Using BERT for NER without vector indexing
```

## Stop Services
```bash
# Stop and remove containers
docker-compose -f docker-compose.test.yml down

# Remove images (optional)
docker rmi ai-processing-suite:staging
```

## Integration with Existing Documentation
- See `API_DOCUMENTATION.md` for complete endpoint reference
- See `FAMILY_QA_EXAMPLES.md` for Family QA usage examples
- See `INTEGRATION_EXAMPLES.md` for client library integration

## Production Notes
- This Docker setup is for testing only
- Production deployments should include OpenSearch and Redis
- Use environment variables to configure external service connections
- Monitor memory usage with ML models in containers
