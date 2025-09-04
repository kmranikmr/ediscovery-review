# AI Processing Suite - Staging Environment

## Documentation Structure

This staging environment now has streamlined documentation organized into two comprehensive guides:

### 📚 **COMPLETE_API_DOCUMENTATION.md**
- **Complete API Reference**: All 25+ endpoints with detailed schemas
- **Family QA Specialization**: Email thread analysis and business communication
- **Client Libraries**: Ready-to-use Python and JavaScript integration code
- **Advanced Features**: Filtering, batch operations, complex queries
- **Error Handling**: Complete error codes and handling patterns
- **Response Schemas**: Detailed data structures for all endpoints

### 🐳 **DOCKER_INFRASTRUCTURE_GUIDE.md**
- **Quick Setup**: 1-minute deployment with automated testing
- **Environment Configuration**: Development, testing, and production configs
- **Docker Deployment Options**: Multiple deployment patterns
- **Testing & Validation**: Automated and manual testing procedures
- **Troubleshooting**: Common issues and detailed solutions
- **Production Deployment**: Security, scaling, and monitoring
- **Performance Optimization**: Resource management and caching

## Quick Start

### For API Integration
```bash
# Read the complete API documentation
cat COMPLETE_API_DOCUMENTATION.md

# Test basic endpoints
curl http://localhost:8001/health
curl -X POST http://localhost:8001/ner/extract -H "Content-Type: application/json" -d '{"text": "John Smith from ACME Corp"}'
```

### For Docker Deployment
```bash
# Quick Docker test
./test-docker.sh

# Read infrastructure guide for advanced setup
cat DOCKER_INFRASTRUCTURE_GUIDE.md
```
- Transformers 4.35+
- Haystack AI 2.0+

### External Services (Optional)
- **OpenSearch 2.11+**: Document storage and search
- **Ollama**: Local LLM inference (Mistral model)
- **Redis 7+**: Task queue and caching

## 🛠️ Installation

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd staging

# Copy environment file and configure
cp .env.example .env
# Edit .env with your settings

# Start with Docker Compose
docker-compose up -d

# Services will be available at:
# - API: http://localhost:8001
# - Streamlit UI: http://localhost:8501
# - API Docs: http://localhost:8001/api/v1/docs
```

### Option 2: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Start services
./scripts/start.sh

# Or start individually:
# API Server:
python main.py

# Streamlit UI:
streamlit run streamlit/app.py --server.port 8501
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# API Configuration
HOST=0.0.0.0
PORT=8001
DEBUG=false

# OpenSearch
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200

# Ollama (for LLM features)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral

# BERT NER
BERT_MODEL_NAME=dbmdz/bert-large-cased-finetuned-conll03-english
BERT_DEVICE=auto  # auto, cpu, cuda
```

### External Services Setup

#### Ollama (for LLM features)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve

# Pull Mistral model
ollama pull mistral
```

#### OpenSearch (for document storage)
```bash
# Using Docker
docker run -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  opensearchproject/opensearch:2.11.0
```

## 📚 API Documentation

### Core Endpoints

#### Question Answering
```bash
# Simple QA
POST /api/v1/qa/simple
{
  "query": "What are the main findings?",
  "collection_id": "my_docs",
  "top_k": 5
}

# Direct index QA
POST /api/v1/qa/direct-index
{
  "query": "Financial information",
  "index_name": "financial_docs",
  "filters": {"document_type": "report"}
}
```

#### Named Entity Recognition
```bash
# BERT NER
POST /api/v1/ner/extract
{
  "text": "John Smith works at Microsoft",
  "method": "bert",
  "entity_types": ["PERSON", "ORGANIZATION"],
  "include_pii": true,
  "min_score": 0.7
}

# LLM NER
POST /api/v1/ner/extract
{
  "text": "Contact jane@company.com",
  "method": "llm",
  "include_pii": true
}
```

#### Summarization
```bash
# Regular summarization
POST /api/v1/summarization/regular
{
  "text": "Long document text...",
  "summary_type": "general",
  "max_length": 150
}

# Thread summarization
POST /api/v1/summarization/thread
{
  "text": "Email thread text...",
  "summary_type": "thread_summary",
  "max_length": 100
}
```

#### Classification
```bash
POST /api/v1/classification/comprehensive
{
  "text": "Document to classify...",
  "classification_types": ["document_type", "priority", "sentiment"]
}
```

#### Document Indexing
```bash
POST /api/v1/documents/index
{
  "documents": [
    {
      "content": "Document content",
      "meta": {
        "document_id": "doc1",
        "document_type": "email"
      }
    }
  ],
  "collection_id": "my_collection",
  "index_name": "my_index"
}
```

## 🧪 Testing

### Run Tests
```bash
# All tests
./scripts/test.sh

# Specific test categories
python -m pytest tests/test_comprehensive.py::TestQAEndpoints -v
python -m pytest tests/test_comprehensive.py::TestNEREndpoints -v

# Quick smoke test
python tests/test_comprehensive.py
```

### Test Coverage
- ✅ Health checks and API connectivity
- ✅ QA functionality with various query types
- ✅ NER with BERT and LLM methods
- ✅ Summarization (regular and thread)
- ✅ Classification (document type, priority, sentiment)
- ✅ Document indexing and retrieval
- ✅ Error handling and edge cases
- ✅ Performance and concurrency tests

## 🎯 Streamlit UI

### Features
- **Interactive QA**: Ask questions about indexed documents
- **Real-time NER**: Extract entities with confidence visualization
- **Document Summarization**: Generate summaries with keyword extraction
- **Classification Dashboard**: Classify documents by type and sentiment
- **Advanced Filtering**: Date ranges, document types, metadata filters
- **Method Selection**: Choose between BERT/LLM for NER and other tasks

### Access
- URL: http://localhost:8501
- Features: All API endpoints with user-friendly interface
- Configuration: Index selection, processing parameters, filters

## 🔄 Celery Task Queue (Future Extension)

The system is designed for easy Celery integration:

```python
# Example async task
from celery import Celery

app = Celery('ediscovery')

@app.task
def process_large_document(document_id):
    # Long-running document processing
    pass

# Usage
result = process_large_document.delay(doc_id)
```

### Redis Setup
```bash
# Start Redis
redis-server

# Configure in docker-compose.yml
redis:
  image: redis:7-alpine
  ports: ["6379:6379"]
```

## 🐳 Docker Deployment

### Full Stack Deployment
```bash
# Production deployment
docker-compose -f docker-compose.yml up -d

# With external services
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale services
docker-compose up -d --scale api=3
```

### Service Health Monitoring
```bash
# Check service health
curl http://localhost:8001/api/v1/health

# View logs
docker-compose logs -f api
docker-compose logs -f streamlit
```

## 📊 Monitoring and Logging

### Application Logs
- Location: `./logs/`
- Files: `app.log`, `api.log`, `streamlit.log`
- Format: Structured JSON with timestamps
- Rotation: 10MB files, 5 backups

### Health Checks
- API: `/api/v1/health`
- Streamlit: `/_stcore/health`
- Services: OpenSearch, Redis, Ollama status

### Metrics (Planned)
- Request latency and throughput
- Model inference times
- Error rates and types
- Resource utilization

## 🔒 Security Considerations

### Current Features
- Input validation with Pydantic
- CORS configuration
- Request timeouts
- Error message sanitization

### Production Recommendations
- API authentication (JWT tokens)
- Rate limiting
- Input sanitization for sensitive data
- HTTPS/TLS termination
- Network segmentation

## 🚀 Performance Optimization

### Current Optimizations
- GPU acceleration for BERT NER
- Async request handling
- Connection pooling
- Efficient text chunking

### Scaling Recommendations
- Load balancer for multiple API instances
- Redis caching for frequent queries
- OpenSearch cluster for large datasets
- Celery workers for heavy processing

## 🛠️ Development

### Project Structure
```
staging/
├── app/                 # Core application
│   ├── api/            # FastAPI endpoints
│   ├── core/           # Configuration and logging
│   ├── services/       # Business logic
│   └── schemas/        # Pydantic models
├── streamlit/          # UI application
├── tests/              # Test suite
├── scripts/            # Deployment scripts
├── docker/             # Docker configurations
└── requirements.txt    # Dependencies
```

### Adding New Endpoints
1. Create endpoint in `app/api/endpoints/`
2. Add schema in `app/schemas/`
3. Implement service logic in `app/services/`
4. Add tests in `tests/`
5. Update API documentation

### Contributing
1. Fork repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

## 📝 License

[Your License Here]

## 🤝 Support

- Documentation: This README and inline code comments
- API Docs: http://localhost:8001/api/v1/docs
- Issues: GitHub Issues
- Email: [Your Contact Email]

---

**Built with ❤️ for production eDiscovery workflows**
