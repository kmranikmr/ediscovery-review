# API Quick Reference Guide

## Base URL: `http://localhost:8001`

## Quick Start Examples

### 1. Family QA (Email Thread Analysis)
```bash
curl -X POST http://localhost:8001/qa/family \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What decisions were made in this email thread?",
    "index_name": "deephousedeephouse_ediscovery_docs_chunks",
    "filters": {"thread_id": "project_discussion_001"},
    "top_k": 15
  }'
```

### 2. Question Answering with Filters
```bash
curl -X POST http://localhost:8001/qa-direct-index \
  -H "Content-Type: application/json" \
  -d '{
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
      }
    }
  }'
```

### 3. NER Entity Extraction
```bash
curl -X POST http://localhost:8001/ner/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "John Smith from ACME Corp called about the meeting.",
    "entity_types": ["PERSON", "ORGANIZATION", "DATE"],
    "min_score": 0.8,
    "method": "bert"
  }'
```

### 4. Document Classification
```bash
curl -X POST http://localhost:8001/classify \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [{"content": "Confidential merger agreement..."}],
    "user_prompt": "Focus on legal privilege",
    "discovery_context": "Merger-related communications",
    "response_format": "standard"
  }'
```

### 5. Text Summarization
```bash
curl -X POST http://localhost:8001/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Long document text...",
    "length": "medium",
    "extract_keywords": true
  }'
```

## Core Endpoints Summary

| Endpoint | Method | Purpose | Key Parameters |
|----------|--------|---------|----------------|
| `/qa-direct-index` | POST | Query documents with filters | `query`, `index_name`, `filters` |
| `/qa/family` | POST | Family-based QA | `query`, `index_name` |
| `/classify` | POST | Document classification | `documents`, `user_prompt` |
| `/ner/extract` | POST | Extract entities from text | `text`, `entity_types` |
| `/summarize` | POST | Summarize text | `text`, `length`, `focus` |
| `/search-direct-index` | POST | Search documents | `query`, `index_name`, `filters` |
| `/health` | GET | Check API status | None |

## Filter Examples for QA/Search

### Date Range Filter
```json
{
  "created_date": {
    "range": {
      "gte": "2001-01-01T00:00:00",
      "lte": "2001-12-31T23:59:59"
    }
  }
}
```

### Multiple Values Filter
```json
{
  "document_type": ["email", "pdf", "text"]
}
```

### Simple Field Filter
```json
{
  "author": "phillip.allen@enron.com",
  "department": "legal"
}
```

## Response Format (All Endpoints)
```json
{
  "success": true|false,
  "result": { /* actual data */ },
  "error": "error message if failed"
}
```

## Default Index
Use `"deephousedeephouse_ediscovery_docs_chunks"` as your primary index name for best results.

## Entity Types for NER
- `PERSON` - Person names
- `ORGANIZATION` - Companies/organizations  
- `LOCATION` - Places and locations
- `EMAIL` - Email addresses
- `PHONE` - Phone numbers
- `DATE` - Dates and times
- `MONEY` - Monetary amounts

## Classification Response Fields
- `responsiveness` - RESPONSIVE/NON-RESPONSIVE/PARTIALLY RESPONSIVE
- `privilege` - Attorney-Client Privileged/Work Product/Not Privileged
- `confidentiality` - Public/Internal/Confidential/Highly Confidential
- `document_type` - Email/Contract/Report/Legal Document/Financial/Other
- `business_relevance` - Critical/High/Medium/Low/None
- `contains_pii` - Yes/No/Uncertain

For complete documentation see `API_DOCUMENTATION.md`
