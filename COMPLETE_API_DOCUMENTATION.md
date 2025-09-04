# AI Processing Suite - Complete API Documentation

## Table of Contents
1. [API Overview](#api-overview)
2. [Quick Start](#quick-start)
3. [Authentication & Headers](#authentication--headers)
4. [Core Endpoints](#core-endpoints)
5. [Family QA Specialization](#family-qa-specialization)
6. [Advanced Features](#advanced-features)
7. [Client Libraries](#client-libraries)
8. [Error Handling](#error-handling)
9. [Response Schemas](#response-schemas)

## API Overview

Base URL: `http://localhost:8001`
API Type: REST
Response Format: JSON
Documentation Standard: OpenAPI 3.0

### Available Services
- **Document Processing**: Parse, index, and extract from various document types
- **Question Answering**: Context-aware QA with document retrieval
- **Family QA**: Specialized email thread and attachment analysis
- **Named Entity Recognition**: Extract people, organizations, locations, PII
- **Text Classification**: Legal, business, and eDiscovery classification
- **Summarization**: Document and email thread summarization
- **Advanced Analytics**: Document similarity, trend analysis

## Quick Start

### 1. Health Check
```bash
curl http://localhost:8001/health
```

### 2. Basic Question Answering
```bash
curl -X POST http://localhost:8001/qa-simple \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key contract terms?",
    "collection_id": "legal_documents"
  }'
```

### 3. Family QA (Email Analysis)
```bash
curl -X POST http://localhost:8001/qa/family \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was discussed about the merger timeline?",
    "documents": [
      {
        "content": "Email thread about merger discussions...",
        "meta": {"type": "email_thread", "participants": ["john@corp.com", "legal@corp.com"]}
      }
    ]
  }'
```

## Authentication & Headers

### Standard Headers
```
Content-Type: application/json
Accept: application/json
```

### Optional Headers
```
X-Request-ID: unique-request-identifier
X-Collection-ID: default-collection-override
X-Model-Override: specific-model-name
```

## Core Endpoints

### 1. Document Indexing

#### Index Documents
```
POST /index
```

**Request:**
```json
{
  "documents": [
    {
      "content": "Contract terms and conditions...",
      "meta": {
        "document_id": "contract_001",
        "source": "legal_department",
        "document_type": "contract",
        "date": "2024-01-15"
      }
    }
  ],
  "collection_id": "legal_contracts"
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "indexed_documents": 1,
    "collection_id": "legal_contracts",
    "processing_time": 1.2
  }
}
```

### 2. Question Answering

#### Simple QA
```
POST /qa-simple
```

**Request:**
```json
{
  "query": "What is the termination clause?",
  "collection_id": "legal_contracts",
  "num_results": 5,
  "model": "mistral"
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "answer": "The termination clause allows either party to terminate with 30 days notice...",
    "sources": [
      {
        "content": "Relevant document excerpt...",
        "metadata": {"document_id": "contract_001", "confidence": 0.95}
      }
    ],
    "confidence": 0.92
  }
}
```

#### Direct Index QA
```
POST /qa-direct-index
```

**Basic Request (using filters):**
```json
{
  "query": "Find merger discussions",
  "index_name": "company_emails",
  "top_k": 10,
  "filters": {
    "date": {"range": {"gte": "2024-01-01", "lte": "2024-12-31"}},
    "participants": ["ceo@company.com", "legal@company.com"]
  }
}
```

**Advanced Request (using raw OpenSearch query):**
```json
{
  "query": "What were the key points in contract negotiations?",
  "index_name": "deephousedeephouse_ediscovery_docs_chunks",
  "top_k": 15,
  "raw_query_body": {
    "query": {
      "bool": {
        "must": [
          {"match": {"content": "contract negotiation"}},
          {"range": {"meta.date": {"gte": "2024-01-01"}}}
        ],
        "should": [
          {"term": {"meta.department": "legal"}},
          {"wildcard": {"meta.participants": "*@legal.company.com"}}
        ],
        "minimum_should_match": 1
      }
    },
    "sort": [{"meta.date": {"order": "desc"}}],
    "aggs": {
      "contract_types": {
        "terms": {"field": "meta.contract_type.keyword", "size": 5}
      }
    }
  }
}
```

**Note**: When `raw_query_body` is provided:
- It overrides the `query` text search and `filters` parameters for document retrieval
- The `query` field is still used for QA answer generation  
- The `top_k` parameter will override any `size` specified in the raw query
- Highlighting and `_source` fields are automatically added if not present

### 3. Named Entity Recognition

#### Extract Entities from Text
```
POST /ner/extract
```

**Request:**
```json
{
  "text": "John Smith from ACME Corporation contacted our legal team about the merger.",
  "entity_types": ["PERSON", "ORGANIZATION", "LEGAL_ENTITY"],
  "include_pii": true,
  "min_score": 0.8,
  "method": "bert"
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "entities": {
      "PERSON": [
        {"text": "John Smith", "confidence": 0.95, "start": 0, "end": 10}
      ],
      "ORGANIZATION": [
        {"text": "ACME Corporation", "confidence": 0.92, "start": 16, "end": 32}
      ]
    },
    "statistics": {
      "total_entities": 2,
      "entity_types": 2,
      "avg_confidence": 0.935
    }
  }
}
```

#### Extract Entities from File
```
POST /ner/extract-from-file
```

**Request:**
```json
{
  "file_path": "/path/to/document.pdf",
  "entity_types": ["PERSON", "ORGANIZATION", "EMAIL", "PHONE"],
  "include_content": true,
  "min_score": 0.7
}
```

### 4. Document Classification

#### eDiscovery Classification
```
POST /classify
```

**Request:**
```json
{
  "documents": [
    {
      "content": "Email regarding quarterly financial projections...",
      "meta": {"source": "email", "date": "2024-03-15"}
    }
  ],
  "classifications": ["responsive", "privileged", "confidential"],
  "user_prompt": "Focus on financial information relevance",
  "discovery_context": "Seeking documents related to financial planning and forecasting",
  "response_format": "comprehensive",
  "include_detailed_reasoning": true,
  "fields_to_include": ["responsiveness", "privilege", "confidentiality", "business_relevance"]
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "classification": "Business Financial Document",
    "confidence": 0.94,
    "responsiveness": "RESPONSIVE",
    "privilege": "Not Privileged",
    "confidentiality": "Internal",
    "business_relevance": "High",
    "topic_analysis": {
      "primary_topic": "Financial Forecasting",
      "subject_matter": "Quarterly Planning"
    },
    "reasoning": {
      "responsiveness_reasoning": "Document directly relates to financial planning as specified in discovery context",
      "context_analysis": "Content matches discovery scope for financial forecasting documents"
    }
  }
}
```

### 5. Text Summarization

#### Text Summarization
```
POST /summarize
```

**Request:**
```json
{
  "text": "Long document content to be summarized...",
  "length": "medium",
  "focus": "business",
  "extract_keywords": true
}
```

#### BART-Only Summarization
```
POST /summarize/bart-only
```

**Request:**
```json
{
  "email_text": "Email content for summarization...",
  "summary_type": "business",
  "max_length": 150,
  "min_length": 40
}
```

## Family QA Specialization

Family QA is designed for comprehensive email thread and attachment analysis, treating related documents as a unified context.

### Key Features
- **Email Thread Analysis**: Understands conversation flow and context
- **Attachment Integration**: Processes emails with their attachments as single units
- **Participant Tracking**: Identifies and tracks communication patterns
- **Business Context Extraction**: Extracts business decisions, timelines, and commitments

### Family QA Endpoint
```
POST /qa/family
```

### Example: Email Thread Analysis
```json
{
  "query": "What decisions were made about the product launch timeline?",
  "documents": [
    {
      "content": "From: CEO <ceo@company.com>\nTo: Product Team <product@company.com>\nSubject: Q3 Product Launch\n\nWe need to accelerate our timeline...",
      "meta": {
        "type": "email",
        "thread_id": "thread_001",
        "participants": ["ceo@company.com", "product@company.com"],
        "timestamp": "2024-03-01T10:00:00Z"
      }
    },
    {
      "content": "From: Product Manager <pm@company.com>\nTo: CEO <ceo@company.com>\nRe: Q3 Product Launch\n\nBased on our analysis, we can move the launch to July 15th...",
      "meta": {
        "type": "email_reply",
        "thread_id": "thread_001",
        "participants": ["pm@company.com", "ceo@company.com"],
        "timestamp": "2024-03-01T14:30:00Z"
      }
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "answer": "Based on the email thread, the product launch timeline was accelerated with a final decision to move the launch date to July 15th. The CEO initially requested acceleration, and the Product Manager confirmed feasibility after analysis.",
    "documents_processed": 2,
    "sources": [
      {
        "content": "From: Product Manager... we can move the launch to July 15th...",
        "metadata": {"thread_id": "thread_001", "timestamp": "2024-03-01T14:30:00Z"}
      }
    ]
  }
}
```

### Family QA with Index Search
```json
{
  "query": "Find all discussions about merger activities in Q1 2024",
  "index_name": "corporate_communications",
  "filters": {
    "date": {"range": {"gte": "2024-01-01", "lte": "2024-03-31"}},
    "type": ["email", "email_thread"],
    "keywords": ["merger", "acquisition", "due diligence"]
  }
}
```

## Advanced Features

### 1. Direct Index Operations

#### Search Index
```
POST /search/direct-index
```

#### Get Index Statistics
```
POST /index/stats
```

#### Browse Documents
```
POST /documents/browse
```

#### Raw OpenSearch Query (NEW)
```
POST /opensearch/raw-query
```

**Request:**
```json
{
  "index_name": "company_emails",
  "query_body": {
    "query": {
      "bool": {
        "must": [
          {"match": {"content": "contract negotiation"}},
          {"range": {"meta.date": {"gte": "2024-01-01"}}}
        ],
        "should": [
          {"term": {"meta.department": "legal"}},
          {"wildcard": {"meta.participants": "*@legal.company.com"}}
        ],
        "minimum_should_match": 1
      }
    },
    "aggs": {
      "by_month": {
        "date_histogram": {
          "field": "meta.date",
          "calendar_interval": "month"
        }
      },
      "top_participants": {
        "terms": {"field": "meta.participants.keyword", "size": 10}
      }
    },
    "sort": [{"meta.date": {"order": "desc"}}],
    "size": 20
  },
  "include_qa": true,
  "qa_query": "What were the key points discussed in contract negotiations?"
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "hits": {
      "total": {"value": 156},
      "hits": [
        {
          "_id": "doc123",
          "_score": 8.5,
          "_source": {
            "content": "Contract negotiation email content...",
            "meta": {"date": "2024-03-15", "participants": ["legal@company.com"]}
          },
          "highlight": {
            "content": ["...contract <em>negotiation</em> highlights..."]
          }
        }
      ]
    },
    "aggregations": {
      "by_month": {
        "buckets": [
          {"key_as_string": "2024-03", "doc_count": 45},
          {"key_as_string": "2024-02", "doc_count": 32}
        ]
      },
      "top_participants": {
        "buckets": [
          {"key": "legal@company.com", "doc_count": 23},
          {"key": "contracts@company.com", "doc_count": 18}
        ]
      }
    },
    "qa_answer": {
      "answer": "Based on the contract negotiation emails, key points included pricing terms, delivery schedules, and liability clauses...",
      "confidence": 0.89,
      "sources_used": 5
    },
    "took": 45,
    "total_hits": 156
  }
}
```

#### Advanced Hybrid Query (NEW)
```
POST /opensearch/advanced-query
```

**Request:**
```json
{
  "text_query": "merger and acquisition discussions",
  "index_name": "corporate_communications",
  "raw_filters": {
    "bool": {
      "must": [
        {"range": {"meta.date": {"gte": "2024-01-01", "lte": "2024-12-31"}}},
        {"terms": {"meta.type": ["email", "meeting_notes", "legal_document"]}}
      ]
    }
  },
  "raw_aggregations": {
    "timeline": {
      "date_histogram": {
        "field": "meta.date",
        "calendar_interval": "week"
      }
    },
    "key_participants": {
      "terms": {"field": "meta.participants.keyword", "size": 15}
    },
    "document_types": {
      "terms": {"field": "meta.type.keyword"}
    }
  },
  "raw_sort": [
    {"meta.importance_score": {"order": "desc"}},
    {"meta.date": {"order": "desc"}}
  ],
  "size": 25,
  "include_qa": true
}
```

#### Query Builder Help (NEW)
```
POST /opensearch/query-builder
```

**Get Examples:**
```json
{
  "query_type": "examples"
}
```

**Validate Query:**
```json
{
  "query_type": "validate",
  "query_body": {
    "query": {"match": {"content": "test"}},
    "size": 10
  }
}
```

### 2. Advanced Filtering

#### Metadata Filters
```json
{
  "filters": {
    "document_type": ["email", "contract"],
    "date": {"range": {"gte": "2024-01-01"}},
    "participants": {"wildcard": "*@legal.company.com"},
    "importance": {"exists": true},
    "custom_field": {"query": {"term": {"value": "merger"}}}
  }
}
```

#### Complex Query Filters
```json
{
  "filters": {
    "bool": {
      "must": [
        {"term": {"meta.department": "legal"}},
        {"range": {"meta.date": {"gte": "2024-01-01"}}}
      ],
      "should": [
        {"match": {"content": "contract"}},
        {"match": {"content": "agreement"}}
      ],
      "must_not": [
        {"term": {"meta.status": "draft"}}
      ]
    }
  }
}
```

### 3. Batch Operations

#### Batch Classification
```json
{
  "documents": [
    {"content": "Document 1...", "meta": {"id": "doc1"}},
    {"content": "Document 2...", "meta": {"id": "doc2"}},
    {"content": "Document 3...", "meta": {"id": "doc3"}}
  ],
  "response_format": "standard",
  "include_detailed_reasoning": false
}
```

## Client Libraries

### Python Client Library

```python
import requests
import json

class AIProcessingClient:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def health_check(self):
        """Check API health status"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def family_qa(self, query, documents=None, index_name=None, filters=None):
        """Perform Family QA analysis"""
        payload = {"query": query}
        
        if documents:
            payload["documents"] = documents
        if index_name:
            payload["index_name"] = index_name
        if filters:
            payload["filters"] = filters
        
        response = self.session.post(f"{self.base_url}/qa/family", json=payload)
        return response.json()
    
    def extract_entities(self, text, entity_types=None, method="bert"):
        """Extract named entities from text"""
        payload = {
            "text": text,
            "entity_types": entity_types or ["PERSON", "ORGANIZATION", "LOCATION"],
            "method": method
        }
        
        response = self.session.post(f"{self.base_url}/ner/extract", json=payload)
        return response.json()
    
    def classify_documents(self, documents, discovery_context=None, response_format="standard"):
        """Classify documents for eDiscovery"""
        payload = {
            "documents": documents,
            "discovery_context": discovery_context,
            "response_format": response_format,
            "include_detailed_reasoning": True
        }
        
        response = self.session.post(f"{self.base_url}/classify", json=payload)
        return response.json()
    
    def raw_opensearch_query(self, index_name, query_body, include_qa=False, qa_query=None):
        """Execute raw OpenSearch query"""
        payload = {
            "index_name": index_name,
            "query_body": query_body,
            "include_qa": include_qa
        }
        
        if qa_query:
            payload["qa_query"] = qa_query
        
        response = self.session.post(f"{self.base_url}/opensearch/raw-query", json=payload)
        return response.json()
    
    def advanced_opensearch_query(self, text_query, index_name, raw_filters=None, raw_aggregations=None, raw_sort=None):
        """Execute advanced hybrid query"""
        payload = {
            "text_query": text_query,
            "index_name": index_name,
            "include_qa": True
        }
        
        if raw_filters:
            payload["raw_filters"] = raw_filters
        if raw_aggregations:
            payload["raw_aggregations"] = raw_aggregations
        if raw_sort:
            payload["raw_sort"] = raw_sort
        
        response = self.session.post(f"{self.base_url}/opensearch/advanced-query", json=payload)
        return response.json()
    
    def get_query_examples(self):
        """Get OpenSearch query examples"""
        payload = {"query_type": "examples"}
        response = self.session.post(f"{self.base_url}/opensearch/query-builder", json=payload)
        return response.json()
    
    def validate_opensearch_query(self, query_body):
        """Validate OpenSearch query"""
        payload = {
            "query_type": "validate",
            "query_body": query_body
        }
        response = self.session.post(f"{self.base_url}/opensearch/query-builder", json=payload)
        return response.json()

    def simple_qa(self, query, collection_id="default", num_results=5):
        """Simple question answering"""
        payload = {
            "query": query,
            "collection_id": collection_id,
            "num_results": num_results
        }
        
        response = self.session.post(f"{self.base_url}/qa-simple", json=payload)
        return response.json()

# Usage Example
client = AIProcessingClient()

# Health check
status = client.health_check()
print(f"API Status: {status}")

# Family QA example
result = client.family_qa(
    query="What was decided about the merger timeline?",
    documents=[
        {
            "content": "Email discussing merger timeline acceleration...",
            "meta": {"type": "email", "participants": ["ceo@corp.com"]}
        }
    ]
)
print(f"Answer: {result['result']['answer']}")

# Entity extraction
entities = client.extract_entities(
    text="John Smith from ACME Corp discussed the contract with our legal team.",
    entity_types=["PERSON", "ORGANIZATION"]
)
print(f"Entities: {entities['result']['entities']}")

# Raw OpenSearch Query Examples
# 1. Complex Boolean Query with Aggregations
complex_query_result = client.raw_opensearch_query(
    index_name="deephousedeephouse_ediscovery_docs_chunks",
    query_body={
        "query": {
            "bool": {
                "must": [
                    {"match": {"content": "merger acquisition"}},
                    {"range": {"meta.date": {"gte": "2024-01-01"}}}
                ],
                "should": [
                    {"term": {"meta.department": "legal"}},
                    {"wildcard": {"meta.participants": "*@legal.company.com"}}
                ],
                "must_not": [
                    {"term": {"meta.status": "draft"}}
                ],
                "minimum_should_match": 1
            }
        },
        "aggs": {
            "monthly_timeline": {
                "date_histogram": {
                    "field": "meta.date",
                    "calendar_interval": "month"
                }
            },
            "top_participants": {
                "terms": {"field": "meta.participants.keyword", "size": 10}
            }
        },
        "sort": [{"meta.date": {"order": "desc"}}],
        "size": 20
    },
    include_qa=True,
    qa_query="What were the main merger discussion points?"
)

# 2. Advanced Hybrid Query (Text + Raw Components)
hybrid_result = client.advanced_opensearch_query(
    text_query="contract negotiations timeline",
    index_name="deephousedeephouse_ediscovery_docs_chunks",
    raw_filters={
        "bool": {
            "must": [
                {"range": {"meta.date": {"gte": "2024-01-01"}}},
                {"terms": {"meta.type": ["email", "document", "meeting_notes"]}}
            ]
        }
    },
    raw_aggregations={
        "contract_types": {
            "terms": {"field": "meta.contract_type.keyword", "size": 5}
        },
        "negotiation_phases": {
            "terms": {"field": "meta.phase.keyword", "size": 3}
        }
    },
    raw_sort=[
        {"meta.importance_score": {"order": "desc"}},
        {"_score": {"order": "desc"}}
    ]
)

# 3. Get Query Examples and Validate
examples = client.get_query_examples()
print("Available query patterns:", examples['result']['examples'].keys())

# Validate a custom query
validation_result = client.validate_opensearch_query({
    "query": {"match": {"content": "test"}},
    "size": 5,
    "aggs": {"doc_count": {"value_count": {"field": "_id"}}}
})
print("Query validation:", validation_result['result']['valid'])
```

### JavaScript Client Library

```javascript
class AIProcessingClient {
    constructor(baseUrl = 'http://localhost:8001') {
        this.baseUrl = baseUrl;
        this.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        };
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const config = {
            headers: this.headers,
            ...options
        };

        const response = await fetch(url, config);
        return await response.json();
    }

    async healthCheck() {
        return await this.request('/health');
    }

    async familyQA(query, documents = null, indexName = null, filters = null) {
        const payload = { query };
        
        if (documents) payload.documents = documents;
        if (indexName) payload.index_name = indexName;
        if (filters) payload.filters = filters;

        return await this.request('/qa/family', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    }

    async extractEntities(text, entityTypes = null, method = 'bert') {
        const payload = {
            text,
            entity_types: entityTypes || ['PERSON', 'ORGANIZATION', 'LOCATION'],
            method
        };

        return await this.request('/ner/extract', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    }

    async classifyDocuments(documents, discoveryContext = null, responseFormat = 'standard') {
        const payload = {
            documents,
            discovery_context: discoveryContext,
            response_format: responseFormat,
            include_detailed_reasoning: true
        };

        return await this.request('/classify', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    }

    async simpleQA(query, collectionId = 'default', numResults = 5) {
        const payload = {
            query,
            collection_id: collectionId,
            num_results: numResults
        };

        return await this.request('/qa-simple', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    }
}

// Usage Example
const client = new AIProcessingClient();

// Family QA
client.familyQA(
    "What decisions were made about the product launch?",
    [
        {
            content: "Email thread about product launch decisions...",
            meta: { type: "email_thread", participants: ["pm@company.com"] }
        }
    ]
).then(result => {
    console.log('Answer:', result.result.answer);
});

// Entity extraction
client.extractEntities(
    "Sarah Johnson from TechCorp Inc. sent the proposal to our legal department.",
    ["PERSON", "ORGANIZATION"]
).then(result => {
    console.log('Entities:', result.result.entities);
});
```

## Error Handling

### Standard Error Response
```json
{
  "success": false,
  "error": "Error description",
  "error_code": "ERROR_TYPE",
  "details": {
    "field": "Additional error context"
  }
}
```

### Common Error Codes
- `INVALID_REQUEST`: Malformed request data
- `MODEL_UNAVAILABLE`: AI model not accessible
- `INDEX_NOT_FOUND`: Specified index doesn't exist
- `PROCESSING_ERROR`: Document processing failed
- `TIMEOUT_ERROR`: Request timeout exceeded
- `AUTHENTICATION_ERROR`: Invalid credentials
- `RATE_LIMIT_EXCEEDED`: Too many requests

### Error Handling Examples

#### Python
```python
try:
    result = client.family_qa(query="Test query", documents=[])
    if result.get("success"):
        print(result["result"])
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.ConnectionError:
    print("Failed to connect to API")
```

#### JavaScript
```javascript
try {
    const result = await client.familyQA("Test query", []);
    if (result.success) {
        console.log(result.result);
    } else {
        console.error('Error:', result.error);
    }
} catch (error) {
    console.error('Network error:', error.message);
}
```

## Response Schemas

### Standard API Response
```json
{
  "success": boolean,
  "result": object | null,
  "error": string | null,
  "processing_time": number,
  "request_id": string
}
```

### QA Response Schema
```json
{
  "answer": string,
  "confidence": number,
  "sources": [
    {
      "content": string,
      "metadata": object,
      "relevance_score": number
    }
  ],
  "processing_info": {
    "documents_searched": number,
    "retrieval_time": number,
    "generation_time": number
  }
}
```

### Classification Response Schema
```json
{
  "classification": string,
  "confidence": number,
  "responsiveness": "RESPONSIVE" | "NON-RESPONSIVE" | "PARTIALLY RESPONSIVE",
  "privilege": string,
  "confidentiality": string,
  "document_type": string,
  "business_relevance": string,
  "topic_analysis": {
    "primary_topic": string,
    "subject_matter": string,
    "key_concepts": array
  },
  "reasoning": {
    "responsiveness_reasoning": string,
    "context_analysis": string
  }
}
```

### NER Response Schema
```json
{
  "entities": {
    "PERSON": [
      {
        "text": string,
        "confidence": number,
        "start": number,
        "end": number
      }
    ],
    "ORGANIZATION": [...],
    "LOCATION": [...]
  },
  "statistics": {
    "total_entities": number,
    "entity_types": number,
    "avg_confidence": number
  },
  "method": string
}
```

---

## Support & Resources

- **API Health**: `GET /health`
- **OpenAPI Spec**: `GET /docs` (when available)
- **Collection Management**: `GET /collections`
- **Model Information**: `GET /` (root endpoint)

For additional support and advanced use cases, refer to the Docker Setup & Infrastructure Guide.
