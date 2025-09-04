# Integration Code Examples

## Python Integration

### Complete Python Client Class
```python
import requests
import json
from typing import Dict, List, Any, Optional

class AIProcessingSuiteClient:
    """Python client for AI Processing Suite API"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def health_check(self) -> bool:
        """Check if API is available"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False
    
    def query_documents(self, 
                       query: str, 
                       index_name: str = "deephousedeephouse_ediscovery_docs_chunks",
                       top_k: int = 5,
                       filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Query documents with advanced filtering"""
        data = {
            "query": query,
            "index_name": index_name,
            "top_k": top_k,
            "direct_access": True
        }
        if filters:
            data["filters"] = filters
        
        response = self.session.post(f"{self.base_url}/qa-direct-index", json=data)
        return response.json()
    
    def extract_entities(self, 
                        text: str,
                        entity_types: List[str] = None,
                        min_score: float = 0.7,
                        method: str = "bert") -> Dict[str, Any]:
        """Extract named entities from text"""
        if entity_types is None:
            entity_types = ["PERSON", "ORGANIZATION", "LOCATION", "EMAIL", "PHONE", "DATE"]
        
        data = {
            "text": text,
            "entity_types": entity_types,
            "min_score": min_score,
            "method": method
        }
        
        response = self.session.post(f"{self.base_url}/ner/extract", json=data)
        return response.json()
    
    def classify_document(self,
                         content: str,
                         user_prompt: str = None,
                         discovery_context: str = None,
                         response_format: str = "standard") -> Dict[str, Any]:
        """Classify document for eDiscovery"""
        data = {
            "documents": [{"content": content}],
            "response_format": response_format
        }
        
        if user_prompt:
            data["user_prompt"] = user_prompt
        if discovery_context:
            data["discovery_context"] = discovery_context
        
        response = self.session.post(f"{self.base_url}/classify", json=data)
        return response.json()
    
    def summarize_text(self,
                      text: str,
                      length: str = "medium",
                      focus: str = "general",
                      extract_keywords: bool = True) -> Dict[str, Any]:
        """Summarize text content"""
        data = {
            "text": text,
            "length": length,
            "focus": focus,
            "extract_keywords": extract_keywords
        }
        
        response = self.session.post(f"{self.base_url}/summarize", json=data)
        return response.json()

# Usage Examples
if __name__ == "__main__":
    client = AIProcessingSuiteClient()
    
    # Check if API is running
    if not client.health_check():
        print("❌ API not available")
        exit(1)
    
    print("✅ API is available")
    
    # Example 1: Question Answering with filters
    print("\n=== Question Answering Example ===")
    qa_result = client.query_documents(
        query="What information do you have about Phillip Allen?",
        filters={
            "author": "phillip.allen@enron.com",
            "created_date": {
                "range": {
                    "gte": "2001-01-01T00:00:00",
                    "lte": "2001-12-31T23:59:59"
                }
            }
        }
    )
    
    if qa_result.get("success"):
        result = qa_result["result"]
        print(f"Answer: {result.get('answer', 'No answer')[:200]}...")
        print(f"Sources found: {len(result.get('sources', []))}")
    else:
        print(f"Error: {qa_result.get('error')}")
    
    # Example 2: NER Extraction
    print("\n=== NER Extraction Example ===")
    sample_text = "John Smith from ACME Corporation called about the quarterly meeting on January 15th, 2024. His email is john.smith@acme.com and phone is (555) 123-4567."
    
    ner_result = client.extract_entities(sample_text)
    
    if ner_result.get("success"):
        entities = ner_result["result"]["entities"]
        for entity_type, entity_list in entities.items():
            print(f"{entity_type}: {[e['text'] for e in entity_list]}")
        print(f"Total entities: {ner_result['result']['statistics']['total_entities']}")
    else:
        print(f"Error: {ner_result.get('error')}")
    
    # Example 3: Document Classification
    print("\n=== Document Classification Example ===")
    sample_document = """
    CONFIDENTIAL MERGER AGREEMENT
    
    This agreement between Company A and Company B outlines the terms
    of the proposed merger, including financial considerations, due
    diligence requirements, and regulatory approvals needed.
    
    Attorney-client privileged information is contained within
    regarding legal strategy and potential risks.
    """
    
    classification_result = client.classify_document(
        content=sample_document,
        user_prompt="Focus on legal privilege and merger-related responsiveness",
        discovery_context="All documents related to merger activities and legal advice"
    )
    
    if classification_result.get("success"):
        result = classification_result["result"]
        print(f"Classification: {result.get('classification')}")
        print(f"Responsiveness: {result.get('responsiveness')}")
        print(f"Privilege: {result.get('privilege')}")
        print(f"Confidentiality: {result.get('confidentiality')}")
        print(f"Document Type: {result.get('document_type')}")
        print(f"Confidence: {result.get('confidence', 0):.2f}")
    else:
        print(f"Error: {classification_result.get('error')}")
    
    # Example 4: Text Summarization
    print("\n=== Text Summarization Example ===")
    long_text = """
    The quarterly business review meeting covered several important topics
    including revenue performance, market expansion strategies, and operational
    efficiency improvements. The sales team reported a 15% increase in revenue
    compared to the previous quarter, driven primarily by new client acquisitions
    in the technology sector. Marketing initiatives have successfully generated
    qualified leads, with conversion rates improving by 8%. The operations team
    highlighted process automation projects that have reduced manual work by 30%
    and improved customer satisfaction scores. Looking ahead, the company plans
    to invest in new technologies and expand into emerging markets while
    maintaining focus on customer retention and product quality improvements.
    """
    
    summary_result = client.summarize_text(
        text=long_text,
        length="short",
        focus="business"
    )
    
    if summary_result.get("success"):
        result = summary_result["result"]
        print(f"Summary: {result.get('summary')}")
        if result.get('keywords'):
            print(f"Keywords: {', '.join(result['keywords'])}")
    else:
        print(f"Error: {summary_result.get('error')}")
```

## JavaScript/Node.js Integration

### Complete Node.js Client Class
```javascript
const axios = require('axios');

class AIProcessingSuiteClient {
    constructor(baseUrl = 'http://localhost:8001') {
        this.baseUrl = baseUrl;
        this.client = axios.create({
            baseURL: baseUrl,
            headers: {
                'Content-Type': 'application/json'
            },
            timeout: 30000
        });
    }

    async healthCheck() {
        try {
            const response = await this.client.get('/health');
            return response.status === 200;
        } catch (error) {
            return false;
        }
    }

    async queryDocuments(query, options = {}) {
        const {
            indexName = 'deephousedeephouse_ediscovery_docs_chunks',
            topK = 5,
            filters = null
        } = options;

        const data = {
            query,
            index_name: indexName,
            top_k: topK,
            direct_access: true
        };

        if (filters) {
            data.filters = filters;
        }

        try {
            const response = await this.client.post('/qa-direct-index', data);
            return response.data;
        } catch (error) {
            return {
                success: false,
                error: error.response?.data?.error || error.message
            };
        }
    }

    async extractEntities(text, options = {}) {
        const {
            entityTypes = ['PERSON', 'ORGANIZATION', 'LOCATION', 'EMAIL', 'PHONE', 'DATE'],
            minScore = 0.7,
            method = 'bert'
        } = options;

        const data = {
            text,
            entity_types: entityTypes,
            min_score: minScore,
            method
        };

        try {
            const response = await this.client.post('/ner/extract', data);
            return response.data;
        } catch (error) {
            return {
                success: false,
                error: error.response?.data?.error || error.message
            };
        }
    }

    async classifyDocument(content, options = {}) {
        const {
            userPrompt = null,
            discoveryContext = null,
            responseFormat = 'standard'
        } = options;

        const data = {
            documents: [{ content }],
            response_format: responseFormat
        };

        if (userPrompt) data.user_prompt = userPrompt;
        if (discoveryContext) data.discovery_context = discoveryContext;

        try {
            const response = await this.client.post('/classify', data);
            return response.data;
        } catch (error) {
            return {
                success: false,
                error: error.response?.data?.error || error.message
            };
        }
    }

    async summarizeText(text, options = {}) {
        const {
            length = 'medium',
            focus = 'general',
            extractKeywords = true
        } = options;

        const data = {
            text,
            length,
            focus,
            extract_keywords: extractKeywords
        };

        try {
            const response = await this.client.post('/summarize', data);
            return response.data;
        } catch (error) {
            return {
                success: false,
                error: error.response?.data?.error || error.message
            };
        }
    }
}

// Usage Examples
async function runExamples() {
    const client = new AIProcessingSuiteClient();

    // Check API availability
    const isHealthy = await client.healthCheck();
    if (!isHealthy) {
        console.log('❌ API not available');
        return;
    }
    console.log('✅ API is available');

    // Example 1: Question Answering
    console.log('\n=== Question Answering Example ===');
    const qaResult = await client.queryDocuments(
        'What information do you have about Phillip Allen?',
        {
            filters: {
                author: 'phillip.allen@enron.com',
                created_date: {
                    range: {
                        gte: '2001-01-01T00:00:00',
                        lte: '2001-12-31T23:59:59'
                    }
                }
            }
        }
    );

    if (qaResult.success) {
        const result = qaResult.result;
        console.log(`Answer: ${result.answer?.substring(0, 200)}...`);
        console.log(`Sources found: ${result.sources?.length || 0}`);
    } else {
        console.log(`Error: ${qaResult.error}`);
    }

    // Example 2: NER Extraction
    console.log('\n=== NER Extraction Example ===');
    const sampleText = 'John Smith from ACME Corporation called about the quarterly meeting on January 15th, 2024. His email is john.smith@acme.com.';
    
    const nerResult = await client.extractEntities(sampleText);
    
    if (nerResult.success) {
        const entities = nerResult.result.entities;
        Object.entries(entities).forEach(([entityType, entityList]) => {
            console.log(`${entityType}: ${entityList.map(e => e.text).join(', ')}`);
        });
        console.log(`Total entities: ${nerResult.result.statistics.total_entities}`);
    } else {
        console.log(`Error: ${nerResult.error}`);
    }

    // Example 3: Document Classification
    console.log('\n=== Document Classification Example ===');
    const sampleDocument = `
        CONFIDENTIAL MERGER AGREEMENT
        
        This agreement between Company A and Company B outlines the terms
        of the proposed merger, including financial considerations and
        regulatory approvals needed.
    `;

    const classificationResult = await client.classifyDocument(sampleDocument, {
        userPrompt: 'Focus on legal privilege and merger-related responsiveness',
        discoveryContext: 'All documents related to merger activities'
    });

    if (classificationResult.success) {
        const result = classificationResult.result;
        console.log(`Classification: ${result.classification}`);
        console.log(`Responsiveness: ${result.responsiveness}`);
        console.log(`Privilege: ${result.privilege}`);
        console.log(`Confidentiality: ${result.confidentiality}`);
        console.log(`Confidence: ${result.confidence?.toFixed(2) || 'N/A'}`);
    } else {
        console.log(`Error: ${classificationResult.error}`);
    }

    // Example 4: Text Summarization
    console.log('\n=== Text Summarization Example ===');
    const longText = `
        The quarterly business review meeting covered several important topics
        including revenue performance, market expansion strategies, and operational
        efficiency improvements. The sales team reported a 15% increase in revenue
        compared to the previous quarter, driven primarily by new client acquisitions
        in the technology sector.
    `;

    const summaryResult = await client.summarizeText(longText, {
        length: 'short',
        focus: 'business'
    });

    if (summaryResult.success) {
        const result = summaryResult.result;
        console.log(`Summary: ${result.summary}`);
        if (result.keywords) {
            console.log(`Keywords: ${result.keywords.join(', ')}`);
        }
    } else {
        console.log(`Error: ${summaryResult.error}`);
    }
}

// Run examples
runExamples().catch(console.error);

module.exports = AIProcessingSuiteClient;
```

## cURL Command Examples

### 1. Question Answering with Date Filter
```bash
curl -X POST http://localhost:8001/qa-direct-index \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What meetings were scheduled?",
    "index_name": "deephousedeephouse_ediscovery_docs_chunks",
    "top_k": 5,
    "filters": {
      "created_date": {
        "range": {
          "gte": "2001-01-01T00:00:00",
          "lte": "2001-12-31T23:59:59"
        }
      },
      "document_type": "email"
    }
  }'
```

### 2. NER Entity Extraction
```bash
curl -X POST http://localhost:8001/ner/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Please contact John Smith at john.smith@acme.com or call (555) 123-4567 regarding the meeting scheduled for January 15th.",
    "entity_types": ["PERSON", "EMAIL", "PHONE", "DATE"],
    "min_score": 0.8,
    "method": "bert"
  }'
```

### 3. Document Classification with Context
```bash
curl -X POST http://localhost:8001/classify \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "content": "ATTORNEY-CLIENT PRIVILEGED COMMUNICATION\n\nRe: Merger Strategy Discussion\n\nThis email contains confidential legal advice regarding the proposed acquisition.",
        "meta": {"document_id": "legal_001"}
      }
    ],
    "user_prompt": "Focus on attorney-client privilege and merger responsiveness",
    "discovery_context": "All communications regarding merger and acquisition activities",
    "include_detailed_reasoning": true,
    "response_format": "comprehensive"
  }'
```

### 4. Text Summarization
```bash
curl -X POST http://localhost:8001/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The board meeting discussed quarterly performance metrics showing strong growth in revenue and customer acquisition. Key initiatives for next quarter include expanding the sales team, implementing new customer retention programs, and investing in product development. The finance team reported improved cash flow and reduced operational costs through automation initiatives.",
    "length": "short",
    "focus": "business",
    "extract_keywords": true
  }'
```

### 5. Family QA (Email Thread Analysis)
```bash
curl -X POST http://localhost:8001/qa/family \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main discussion points in this email thread?",
    "index_name": "deephousedeephouse_ediscovery_docs_chunks",
    "filters": {
      "thread_id": "thread_123"
    },
    "top_k": 10
  }'
```

### 6. Search Documents Directly
```bash
curl -X POST http://localhost:8001/search-direct-index \
  -H "Content-Type: application/json" \
  -d '{
    "query": "contract negotiation",
    "index_name": "deephousedeephouse_ediscovery_docs_chunks",
    "filters": {
      "document_type": ["email", "pdf"],
      "author": "legal@company.com"
    },
    "fuzzy": true,
    "top_k": 20
  }'
```

## Error Handling Examples

### Python Error Handling
```python
def safe_api_call(client, method, *args, **kwargs):
    """Wrapper for safe API calls with comprehensive error handling"""
    try:
        result = getattr(client, method)(*args, **kwargs)
        
        if not result.get("success"):
            error_msg = result.get("error", "Unknown error")
            print(f"API Error in {method}: {error_msg}")
            return None
        
        return result.get("result")
    
    except requests.exceptions.ConnectionError:
        print(f"Connection Error: API server not available")
        return None
    except requests.exceptions.Timeout:
        print(f"Timeout Error: API request took too long")
        return None
    except Exception as e:
        print(f"Unexpected Error in {method}: {str(e)}")
        return None

# Usage
client = AIProcessingSuiteClient()
result = safe_api_call(client, "query_documents", "What is the budget?")
if result:
    print(f"Answer: {result.get('answer')}")
```

### JavaScript Error Handling
```javascript
async function safeApiCall(client, method, ...args) {
    try {
        const result = await client[method](...args);
        
        if (!result.success) {
            console.error(`API Error in ${method}:`, result.error);
            return null;
        }
        
        return result.result;
    } catch (error) {
        if (error.code === 'ECONNREFUSED') {
            console.error('Connection Error: API server not available');
        } else if (error.code === 'ETIMEDOUT') {
            console.error('Timeout Error: API request took too long');
        } else {
            console.error(`Unexpected Error in ${method}:`, error.message);
        }
        return null;
    }
}

// Usage
const client = new AIProcessingSuiteClient();
const result = await safeApiCall(client, 'queryDocuments', 'What is the budget?');
if (result) {
    console.log('Answer:', result.answer);
}
```

## Integration Best Practices

1. **Always check health before processing**: Use health check endpoint
2. **Handle errors gracefully**: Implement proper error handling
3. **Use appropriate timeouts**: Set reasonable timeout values
4. **Batch operations when possible**: Process multiple documents together
5. **Cache results appropriately**: Avoid repeated identical requests
6. **Use specific filters**: Improve performance with targeted queries
7. **Monitor response times**: Track API performance
8. **Implement retry logic**: Handle temporary failures
9. **Validate input data**: Check data before sending to API
10. **Log operations**: Keep track of API usage and errors
