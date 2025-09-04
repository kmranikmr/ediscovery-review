# AI Processing Suite API Documentation

## Overview
Complete REST API documentation for the AI Processing Suite running on `http://localhost:8001`. This system provides advanced document processing capabilities including Question Answering, Classification, Summarization, and Named Entity Recognition (NER).

## Base Configuration
- **Base URL**: `http://localhost:8001`
- **Response Format**: JSON with standardized structure
- **Authentication**: None required (local development)
- **Content-Type**: `application/json`

## Standard API Response Format
All endpoints return a standardized response:
```json
{
  "success": boolean,
  "result": any,
  "error": string | null
}
```

## Core Endpoints

### 1. Health Check & System Information

#### GET `/`
**Description**: Root endpoint with system information
**Response**:
```json
{
  "message": "Haystack Email Processing API",
  "available_pipelines": ["summarization", "qa", "family_qa", "family_summarization", "thread_summarization", "classification"],
  "endpoints": {
    "summarization": "/summarize",
    "bart_summarization": "/summarize/bart-only",
    "family_summarization": "/summarize/family",
    "thread_summarization": "/summarize/thread",
    "qa": "/qa",
    "family_qa": "/qa/family",
    "classification": "/classify",
    "bart_classification": "/classify/bart-only",
    "ner_extraction": "/ner/extract",
    "ner_file_extraction": "/ner/extract-from-file",
    "indexing": "/index",
    "simple_qa": "/qa-simple",
    "collections": "/collections"
  }
}
```

#### GET `/health`
**Description**: Health check endpoint
**Response**:
```json
{
  "status": "healthy",
  "pipelines_loaded": 6
}
```

## 2. Question Answering (QA) Endpoints

### 2.1 Direct Index QA (Primary QA Method)

#### POST `/qa-direct-index`
**Description**: Query documents directly from OpenSearch index with advanced filtering
**Request Body**:
```json
{
  "query": "What information do you have about Phillip Allen?",
  "index_name": "deephousedeephouse_ediscovery_docs_chunks",
  "top_k": 10,
  "direct_access": true,
  "filters": {
    "document_id": "doc123",
    "type": "email",
    "author": "john.smith@company.com",
    "created_date": {
      "range": {
        "gte": "2001-01-01T00:00:00",
        "lte": "2001-12-31T23:59:59"
      }
    }
  }
}
```

**Request Schema**:
- `query` (string, required): The question to ask
- `index_name` (string, required): OpenSearch index name
- `top_k` (integer, optional, default: 5): Number of documents to retrieve
- `direct_access` (boolean, optional, default: true): Bypass collection filtering
- `filters` (object, optional): Advanced metadata filters

**Supported Filter Types**:
- **Simple**: `{"field": "value"}`
- **Multiple values**: `{"field": ["val1", "val2"]}`
- **Range**: `{"field": {"range": {"gte": 100, "lte": 200}}}`
- **Wildcard**: `{"field": {"wildcard": "prefix*"}}`
- **Document ID**: `{"document_id": "doc123"}` or `{"id": "doc123"}`
- **Exists**: `{"field": {"exists": true}}`

**Response**:
```json
{
  "success": true,
  "result": {
    "answer": "Based on the retrieved documents, Phillip Allen is mentioned in several email communications...",
    "sources": [
      {
        "content": "Email content excerpt...",
        "metadata": {
          "document_id": "email_123",
          "author": "phillip.allen@enron.com",
          "date": "2001-05-15"
        }
      }
    ],
    "total_documents_searched": 15
  }
}
```

### 2.2 Family QA

#### POST `/qa/family`
**Description**: Answer questions about email families (email + attachments as single unit). Analyzes related documents as cohesive units for context-aware responses.

**Key Features**:
- Analyzes email threads and related attachments together
- Provides context-aware answers about document families
- Supports complex filtering for thread analysis
- Ideal for legal discovery and business communication analysis

**Request Body**:
```json
{
  "query": "What are the key topics in this email thread?",
  "documents": [],
  "index_name": "deephousedeephouse_ediscovery_docs_chunks",
  "top_k": 10,
  "filters": {
    "thread_id": "thread_456",
    "author": ["manager@company.com", "team.lead@company.com"],
    "created_date": {
      "range": {
        "gte": "2001-01-01T00:00:00",
        "lte": "2001-12-31T23:59:59"
      }
    }
  }
}
```

**Request Schema**:
- `query` (string, required): The question to ask about the document family
- `documents` (array, optional): Can be empty - will search index for related documents
- `index_name` (string, optional): Index to search (default: uses system default)
- `top_k` (integer, optional, default: 10): Number of documents to retrieve
- `filters` (object, optional): Advanced filtering options
  - `thread_id` (string): Specific email thread identifier
  - `author` (string|array): Filter by author(s)
  - `participants` (array): Filter by email participants
  - `document_type` (array): Filter by document types
  - `created_date` (object): Date range filtering
  - `subject` (object): Subject line filtering with wildcards

**Common Use Cases**:
1. **Email Thread Analysis**: "What decisions were made in this thread?"
2. **Legal Discovery**: "What privileged information was discussed?"
3. **Business Analysis**: "What concerns were raised about the project?"
4. **Timeline Analysis**: "What was the progression of events?"

**Response**:
```json
{
  "success": true,
  "result": {
    "answer": "The email thread discusses project timelines and budget concerns. Key points include: 1) The launch date was moved from June 1st to July 15th, 2) Budget increased by 15% for additional testing, 3) Three team members were reassigned to address timeline issues, and 4) Weekly status meetings were established starting April 1st.",
    "documents_processed": 8,
    "sources": [
      {
        "content": "After reviewing the timeline, we need to push the launch date to July 15th to accommodate the additional testing phase...",
        "metadata": {
          "thread_id": "thread_456",
          "message_index": 1,
          "author": "project.manager@company.com",
          "date": "2001-03-15",
          "subject": "Re: Project Timeline Update"
        }
      },
      {
        "content": "I agree with the timeline extension. The budget impact will be approximately 15% increase but it's necessary for quality assurance...",
        "metadata": {
          "thread_id": "thread_456", 
          "message_index": 2,
          "author": "finance.director@company.com",
          "date": "2001-03-16"
        }
      }
    ]
  }
}
```

**Family QA vs Regular QA**:
- **Family QA**: Analyzes related documents as cohesive units, maintains thread context
- **Regular QA**: Searches individual documents independently
- **Family QA**: Better for understanding conversations and document relationships
- **Regular QA**: Better for finding specific facts across unrelated documents

## 3. Document Classification

### 3.1 Standard Classification

#### POST `/classify`
**Description**: Classify documents for eDiscovery with customizable analysis depth
**Request Body**:
```json
{
  "documents": [
    {
      "content": "This is a confidential merger agreement between Company A and Company B...",
      "meta": {
        "document_id": "contract_001",
        "source": "legal_department"
      }
    }
  ],
  "classifications": ["business", "legal", "confidential"],
  "user_prompt": "Focus on responsiveness to contract disputes",
  "discovery_context": "All documents related to merger activities and contract negotiations",
  "include_detailed_reasoning": true,
  "include_topic_analysis": true,
  "response_format": "comprehensive",
  "fields_to_include": ["responsiveness", "privilege", "confidentiality", "document_type"]
}
```

**Request Schema**:
- `documents` (array, required): List of documents to classify
  - `content` (string, required): Document text content
  - `meta` (object, optional): Document metadata
- `classifications` (array, optional): Classification categories
- `user_prompt` (string, optional): Additional context for classification
- `discovery_context` (string, optional): Discovery request context
- `include_detailed_reasoning` (boolean, default: true): Include reasoning analysis
- `include_topic_analysis` (boolean, default: true): Include topic analysis
- `response_format` (string, default: "comprehensive"): "minimal", "standard", "comprehensive"
- `fields_to_include` (array, optional): Specific fields to include

**Response**:
```json
{
  "success": true,
  "result": {
    "classification": "Legal Document",
    "confidence": 0.95,
    "responsiveness": "RESPONSIVE",
    "privilege": "Not Privileged",
    "confidentiality": "Highly Confidential",
    "document_type": "Contract",
    "business_relevance": "Critical",
    "contains_pii": "No",
    "topic_analysis": {
      "primary_topic": "Merger Agreement",
      "subject_matter": "Corporate Legal Matter",
      "secondary_topics": ["Due Diligence", "Financial Terms"],
      "key_concepts": ["merger", "acquisition", "confidentiality"]
    },
    "reasoning": {
      "responsiveness_reasoning": "Document directly relates to merger activities as specified in discovery context",
      "context_analysis": "Content matches business/legal domain of discovery request"
    }
  }
}
```

### 3.2 BART-Only Classification

#### POST `/classify/bart-only`
**Description**: Classification using BART model only (no Ollama integration)
**Request Body**:
```json
{
  "email_text": "Subject: Urgent Contract Review\nPlease review the attached contract...",
  "classification_schemes": ["business", "legal", "sentiment", "priority"],
  "include_advanced_analysis": true,
  "confidence_threshold": 0.7
}
```

**Response**:
```json
{
  "success": true,
  "result": {
    "classifications": {
      "business": 0.89,
      "legal": 0.94,
      "sentiment": "urgent",
      "priority": "high"
    },
    "metadata": {
      "model_used": "Enhanced ML Processor (BART + Traditional ML)",
      "confidence_threshold": 0.7
    }
  }
}
```

## 4. Summarization Endpoints

### 4.1 Text Summarization

#### POST `/summarize`
**Description**: Summarize text content with customizable parameters
**Request Body**:
```json
{
  "text": "Long document text to be summarized...",
  "length": "medium",
  "focus": "business",
  "extract_keywords": true
}
```

**Request Schema**:
- `text` (string, required): Text to summarize
- `length` (string, default: "medium"): "short", "medium", "long"
- `focus` (string, default: "general"): Focus area for summarization
- `extract_keywords` (boolean, default: true): Extract keywords

**Response**:
```json
{
  "success": true,
  "result": {
    "summary": "The document discusses key business initiatives and strategic planning...",
    "insights": "Analyzed 15 sentences with business focus",
    "keywords": ["business", "strategy", "planning", "initiatives", "growth"]
  }
}
```

### 4.2 BART-Only Summarization

#### POST `/summarize/bart-only`
**Description**: BART-only summarization for model comparison
**Request Body**:
```json
{
  "email_text": "Email content to summarize...",
  "summary_type": "business",
  "max_length": 150,
  "min_length": 40
}
```

**Response**:
```json
{
  "success": true,
  "result": {
    "summary": "Business-focused summary of the email content...",
    "business_facts": {
      "key_points": ["point1", "point2"],
      "action_items": ["action1", "action2"]
    },
    "model_used": "BART-only (facebook/bart-large-cnn)",
    "processing_time": 1.2,
    "confidence_score": 0.87
  }
}
```

### 4.3 Family & Thread Summarization

#### POST `/summarize/family`
**Description**: Summarize email families (email + attachments)

#### POST `/summarize/thread`
**Description**: Summarize email threads

Both endpoints use the same request format:
```json
{
  "documents": [
    {
      "content": "Email content...",
      "meta": {"thread_id": "thread_123", "message_index": 1}
    }
  ]
}
```

## 5. Named Entity Recognition (NER)

### 5.1 Text NER Extraction

#### POST `/ner/extract`
**Description**: Extract named entities and PII from text with position information
**Request Body**:
```json
{
  "text": "John Smith from ACME Corp called about the meeting on January 15th, 2024.",
  "entity_types": ["PERSON", "ORGANIZATION", "DATE", "LOCATION", "EMAIL", "PHONE"],
  "include_pii": true,
  "min_score": 0.7,
  "method": "bert"
}
```

**Request Schema**:
- `text` (string, required): Text to analyze
- `entity_types` (array, optional): Entity types to extract
- `include_pii` (boolean, default: true): Include PII detection
- `min_score` (float, default: 0.7): Minimum confidence score
- `method` (string, default: "bert"): "bert" or "llm"

**Supported Entity Types**:
- `PERSON`: Person names
- `ORGANIZATION`: Company/organization names
- `LOCATION`: Places and locations
- `EMAIL`: Email addresses
- `PHONE`: Phone numbers
- `DATE`: Dates and times
- `MONEY`: Monetary amounts
- `MISC`: Miscellaneous entities

**Response**:
```json
{
  "success": true,
  "result": {
    "entities": {
      "PERSON": [
        {
          "text": "John Smith",
          "confidence": 0.99,
          "start": 0,
          "end": 10
        }
      ],
      "ORGANIZATION": [
        {
          "text": "ACME Corp",
          "confidence": 0.95,
          "start": 16,
          "end": 25
        }
      ],
      "DATE": [
        {
          "text": "January 15th, 2024",
          "confidence": 0.98,
          "start": 54,
          "end": 72
        }
      ]
    },
    "statistics": {
      "total_entities": 3,
      "entity_types": 3,
      "avg_confidence": 0.97
    },
    "method": "bert_model",
    "position_data": [
      {
        "text": "John Smith",
        "label": "PERSON",
        "start": 0,
        "end": 10,
        "confidence": 0.99
      }
    ]
  }
}
```

### 5.2 File NER Extraction

#### POST `/ner/extract-from-file`
**Description**: Extract entities from a file
**Request Body**:
```json
{
  "file_path": "/path/to/document.txt",
  "entity_types": ["PERSON", "ORGANIZATION", "EMAIL"],
  "include_pii": true,
  "min_score": 0.7,
  "include_content": false,
  "method": "bert"
}
```

**Request Schema**:
- `file_path` (string, required): Path to file to process
- `entity_types` (array, optional): Entity types to extract
- `include_pii` (boolean, default: true): Include PII detection
- `min_score` (float, default: 0.7): Minimum confidence score
- `include_content` (boolean, default: false): Include file content in response
- `method` (string, default: "bert"): NER method

## 6. Document Management

### 6.1 Index Documents

#### POST `/index`
**Description**: Index documents for later QA queries
**Request Body**:
```json
{
  "documents": [
    {
      "content": "Document content to index...",
      "meta": {
        "document_id": "doc_123",
        "source": "user_upload",
        "author": "john.doe@company.com"
      }
    }
  ],
  "collection_id": "financial_docs"
}
```

### 6.2 Direct Index Search

#### POST `/search-direct-index`
**Description**: Search documents directly in an index
**Request Body**:
```json
{
  "query": "search terms",
  "index_name": "deephousedeephouse_ediscovery_docs_chunks",
  "filters": {
    "document_type": "email",
    "date_range": {
      "gte": "2001-01-01",
      "lte": "2001-12-31"
    }
  },
  "fuzzy": true,
  "top_k": 10
}
```

### 6.3 Index Statistics

#### POST `/index-stats`
**Description**: Get statistics about an index
**Request Body**:
```json
{
  "index_name": "deephousedeephouse_ediscovery_docs_chunks"
}
```

### 6.4 Browse Documents

#### POST `/browse-documents`
**Description**: Browse documents in an index
**Request Body**:
```json
{
  "index_name": "deephousedeephouse_ediscovery_docs_chunks",
  "start": 0,
  "limit": 20,
  "filters": {
    "document_type": "email"
  }
}
```

## Integration Examples

### Python Integration Example
```python
import requests
import json

# Base configuration
API_BASE_URL = "http://localhost:8001"

def call_api(endpoint, data):
    """Call API endpoint with error handling"""
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=data)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

# Example 1: Family QA for Email Thread Analysis
family_qa_request = {
    "query": "What decisions were made about the project timeline and budget?",
    "index_name": "deephousedeephouse_ediscovery_docs_chunks",
    "filters": {
        "thread_id": "project_timeline_discussion_2001",
        "author": ["project.manager@company.com", "finance.director@company.com"],
        "created_date": {
            "range": {
                "gte": "2001-03-01T00:00:00",
                "lte": "2001-03-31T23:59:59"
            }
        }
    },
    "top_k": 15
}

family_result = call_api("/qa/family", family_qa_request)
if family_result.get("success"):
    result_data = family_result["result"]
    print("Family QA Answer:", result_data["answer"])
    print("Documents processed:", result_data["documents_processed"])
    print("Sources found:", len(result_data.get("sources", [])))

# Example 2: Regular Question Answering with Advanced Filters
qa_request = {
    "query": "What information do you have about Phillip Allen?",
    "index_name": "deephousedeephouse_ediscovery_docs_chunks",
    "top_k": 5,
    "filters": {
        "author": "phillip.allen@enron.com",
        "created_date": {
            "range": {
                "gte": "2001-01-01T00:00:00",
                "lte": "2001-12-31T23:59:59"
            }
        },
        "document_type": "email"
    }
}

result = call_api("/qa-direct-index", qa_request)
if result.get("success"):
    print("Answer:", result["result"]["answer"])
    print("Sources found:", len(result["result"]["sources"]))

# Example 3: NER Extraction with Position Data
ner_request = {
    "text": "John Smith from ACME Corp will attend the meeting on January 15th at the Chicago office.",
    "entity_types": ["PERSON", "ORGANIZATION", "DATE", "LOCATION"],
    "min_score": 0.8,
    "method": "bert"
}

ner_result = call_api("/ner/extract", ner_request)
if ner_result.get("success"):
    entities = ner_result["result"]["entities"]
    for entity_type, entity_list in entities.items():
        print(f"{entity_type}: {[e['text'] for e in entity_list]}")
        # Print position information for highlighting
        for entity in entity_list:
            print(f"  - '{entity['text']}' at position {entity['start']}-{entity['end']} (confidence: {entity['confidence']:.2f})")

# Example 4: Document Classification with Discovery Context
classification_request = {
    "documents": [
        {
            "content": "CONFIDENTIAL ATTORNEY-CLIENT COMMUNICATION\n\nRe: Merger Legal Strategy\n\nThis email contains our legal assessment of the proposed merger with TechCorp. We have identified several regulatory hurdles that may require FTC approval.",
            "meta": {"document_id": "merger_legal_001", "author": "legal@company.com"}
        }
    ],
    "user_prompt": "Focus on attorney-client privilege and merger responsiveness",
    "discovery_context": "All communications related to merger activities and legal strategy discussions",
    "include_detailed_reasoning": True,
    "response_format": "comprehensive"
}

class_result = call_api("/classify", classification_request)
if class_result.get("success"):
    result_data = class_result["result"]
    print(f"Classification: {result_data['classification']}")
    print(f"Responsiveness: {result_data['responsiveness']}")
    print(f"Privilege: {result_data['privilege']}")
    print(f"Confidence: {result_data['confidence']:.2f}")
    
    # Print detailed reasoning if included
    if "reasoning" in result_data:
        print("\nDetailed Reasoning:")
        print(f"Responsiveness: {result_data['reasoning']['responsiveness_reasoning']}")
        print(f"Privilege: {result_data['reasoning']['privilege_reasoning']}")

# Example 5: Advanced Document Search
search_request = {
    "query": "contract negotiation timeline",
    "index_name": "deephousedeephouse_ediscovery_docs_chunks",
    "filters": {
        "document_type": ["email", "pdf", "document"],
        "department": "legal",
        "created_date": {
            "range": {
                "gte": "2001-01-01T00:00:00",
                "lte": "2001-12-31T23:59:59"
            }
        },
        "keywords": {
            "wildcard": "*contract*"
        }
    },
    "fuzzy": True,
    "top_k": 20
}

search_result = call_api("/search-direct-index", search_request)
if search_result.get("success"):
    documents = search_result["result"]["documents"]
    print(f"Found {len(documents)} documents matching search criteria")
    for doc in documents[:3]:  # Show first 3 results
        print(f"- {doc['content'][:100]}...")
        print(f"  Source: {doc.get('metadata', {}).get('author', 'Unknown')}")

# Example 6: Text Summarization with Keywords
summary_request = {
    "text": "The quarterly board meeting covered revenue performance showing 22% growth, successful contract negotiations worth $2.95M, and approved $3.5M investment for European expansion targeting Germany and France. Technology upgrades scheduled for August including server capacity expansion.",
    "length": "short",
    "focus": "business",
    "extract_keywords": True
}

summary_result = call_api("/summarize", summary_request)
if summary_result.get("success"):
    result_data = summary_result["result"]
    print("Summary:", result_data["summary"])
    if "keywords" in result_data:
        print("Keywords:", ", ".join(result_data["keywords"]))
```

### JavaScript/Node.js Integration Example
```javascript
const axios = require('axios');

const API_BASE_URL = 'http://localhost:8001';

async function callAPI(endpoint, data) {
    try {
        const response = await axios.post(`${API_BASE_URL}${endpoint}`, data);
        return response.data;
    } catch (error) {
        return { success: false, error: error.message };
    }
}

// Example: Text Summarization
async function summarizeText() {
    const request = {
        text: "Long document text to summarize...",
        length: "medium",
        extract_keywords: true
    };

    const result = await callAPI('/summarize', request);
    if (result.success) {
        console.log('Summary:', result.result.summary);
        console.log('Keywords:', result.result.keywords);
    }
}

// Example: Family QA
async function familyQA() {
    const request = {
        query: "What are the main topics discussed?",
        index_name: "deephousedeephouse_ediscovery_docs_chunks",
        top_k: 5
    };

    const result = await callAPI('/qa/family', request);
    if (result.success) {
        console.log('Answer:', result.result.answer);
        console.log('Documents processed:', result.result.documents_processed);
    }
}
```

## Error Handling

### Common Error Responses
```json
{
  "success": false,
  "error": "Error description",
  "detail": "Additional error details"
}
```

### HTTP Status Codes
- **200**: Success
- **400**: Bad Request (invalid input)
- **404**: Not Found (endpoint or resource not found)
- **500**: Internal Server Error
- **503**: Service Unavailable (model not loaded)

## Performance Notes

1. **Index Selection**: Use existing index `deephousedeephouse_ediscovery_docs_chunks` for best performance
2. **Batch Processing**: Process multiple documents in single requests when possible
3. **Filter Optimization**: Use specific filters to reduce search scope
4. **Model Selection**: BERT NER method generally faster than LLM method
5. **Response Formats**: Use "minimal" or "standard" for faster classification responses

## Service Dependencies

- **OpenSearch**: Document indexing and search (port 9200)
- **Ollama**: LLM processing (port 11434)
- **BERT Models**: NER and classification
- **API Server**: Main FastAPI service (port 8001)

## Support & Troubleshooting

### Check Service Status
```bash
# Check API health
curl http://localhost:8001/health

# Check available endpoints
curl http://localhost:8001/
```

### Common Issues
1. **"NER functionality not available"**: BERT models not loaded
2. **"Pipeline not found"**: Haystack pipelines not initialized
3. **"Index not found"**: OpenSearch index doesn't exist
4. **Connection errors**: Check if services are running on correct ports

This documentation covers all available endpoints and integration patterns for the AI Processing Suite. For additional support or custom integrations, refer to the source code or contact the development team.
