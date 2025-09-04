# Family QA and Advanced Examples

## Family QA Examples

Family QA is designed to analyze email families (email + attachments) as cohesive units, providing context-aware answers about email threads and related documents.

### Example 1: Basic Family QA Query

**Request:**
```json
POST /qa/family
{
  "query": "What are the main discussion points in this email thread?",
  "index_name": "deephousedeephouse_ediscovery_docs_chunks",
  "top_k": 10
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "answer": "The email thread primarily discusses quarterly budget planning and resource allocation. Key discussion points include: 1) Marketing budget increase of 15% for Q2, 2) Staff hiring freeze until Q3, 3) Technology infrastructure upgrades scheduled for July, and 4) Concerns about client retention rates in the northeast region.",
    "documents_processed": 8,
    "sources": [
      {
        "content": "Subject: Q2 Budget Planning\nI wanted to follow up on our discussion about the marketing budget...",
        "metadata": {
          "thread_id": "thread_456",
          "message_index": 1,
          "author": "john.manager@company.com",
          "date": "2001-03-15"
        }
      },
      {
        "content": "Re: Q2 Budget Planning\nRegarding the staff hiring freeze, I think we should reconsider...",
        "metadata": {
          "thread_id": "thread_456", 
          "message_index": 2,
          "author": "sarah.hr@company.com",
          "date": "2001-03-16"
        }
      }
    ]
  }
}
```

### Example 2: Family QA with Thread Filtering

**Request:**
```json
POST /qa/family
{
  "query": "What decisions were made about the project timeline?",
  "index_name": "deephousedeephouse_ediscovery_docs_chunks",
  "filters": {
    "thread_id": "project_timeline_discussion",
    "author": ["project.manager@company.com", "team.lead@company.com"]
  },
  "top_k": 15
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "answer": "Based on the email thread, three key decisions were made about the project timeline: 1) The launch date was moved from June 1st to July 15th to accommodate additional testing, 2) The beta testing phase was extended by 3 weeks, and 3) The team agreed to implement a phased rollout starting with the east coast offices first.",
    "documents_processed": 12,
    "sources": [
      {
        "content": "After careful consideration, we've decided to push the launch date to July 15th...",
        "metadata": {
          "thread_id": "project_timeline_discussion",
          "author": "project.manager@company.com",
          "subject": "Updated Project Timeline - Final Decision"
        }
      }
    ]
  }
}
```

### Example 3: Family QA for Legal Discovery

**Request:**
```json
POST /qa/family
{
  "query": "What information was shared about the merger negotiations?",
  "index_name": "deephousedeephouse_ediscovery_docs_chunks",
  "filters": {
    "document_type": "email",
    "contains_privileged": false,
    "date_range": {
      "gte": "2001-01-01T00:00:00",
      "lte": "2001-06-30T23:59:59"
    },
    "participants": ["legal@company.com", "ceo@company.com"]
  },
  "top_k": 20
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "answer": "The email thread reveals several key pieces of information about merger negotiations: 1) Initial discussions began in February 2001 with TechCorp Industries, 2) The proposed valuation was $45 million with a mix of cash and stock, 3) Due diligence was scheduled for April-May timeframe, 4) Regulatory approval was expected to take 6-8 months, and 5) Employee retention packages were being developed for key staff. Note: This information appears to be from business communications and does not include attorney-client privileged content.",
    "documents_processed": 18,
    "sources": [
      {
        "content": "The TechCorp merger discussions have progressed to the point where we need to begin formal due diligence...",
        "metadata": {
          "thread_id": "merger_negotiations_2001",
          "confidentiality": "Internal",
          "business_unit": "Corporate Development"
        }
      }
    ]
  }
}
```

## Additional Endpoint Examples

### Regular QA with Advanced Filtering

#### Example 1: Date Range + Author Filter
**Request:**
```json
POST /qa-direct-index
{
  "query": "What meetings were scheduled with external clients?",
  "index_name": "deephousedeephouse_ediscovery_docs_chunks",
  "top_k": 10,
  "filters": {
    "author": "scheduler@company.com",
    "document_type": "email",
    "created_date": {
      "range": {
        "gte": "2001-03-01T00:00:00",
        "lte": "2001-03-31T23:59:59"
      }
    },
    "subject": {
      "wildcard": "*meeting*"
    }
  }
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "answer": "During March 2001, several external client meetings were scheduled: 1) TechStart Industries on March 5th for product demo, 2) Global Solutions Inc. on March 12th for contract negotiations, 3) Innovation Partners on March 18th for quarterly review, and 4) Future Systems Corp on March 25th for partnership discussions. All meetings were coordinated by the scheduling department and included senior management participation.",
    "sources": [
      {
        "content": "Meeting scheduled with TechStart Industries for March 5th at 2 PM...",
        "metadata": {
          "author": "scheduler@company.com",
          "subject": "Client Meeting - TechStart Industries",
          "meeting_type": "product_demo"
        }
      }
    ],
    "total_documents_searched": 8
  }
}
```

#### Example 2: Multi-Field Complex Filter
**Request:**
```json
POST /qa-direct-index
{
  "query": "What budget concerns were raised?",
  "index_name": "deephousedeephouse_ediscovery_docs_chunks",
  "filters": {
    "department": ["finance", "accounting", "budget"],
    "priority": ["high", "urgent"],
    "contains_financial_data": true,
    "fiscal_year": "2001",
    "keywords": {
      "wildcard": "*budget*"
    }
  },
  "top_k": 15
}
```

### Classification Examples

#### Example 1: eDiscovery Classification with Context
**Request:**
```json
POST /classify
{
  "documents": [
    {
      "content": "CONFIDENTIAL ATTORNEY-CLIENT COMMUNICATION\n\nSubject: Re: Merger Strategy Legal Analysis\n\nThis email contains our legal assessment of the proposed merger with TechCorp. Based on our analysis of the due diligence materials, we have identified several potential regulatory hurdles that may require additional documentation and approvals from the FTC.\n\nKey legal concerns:\n1. Market concentration in the northeast region\n2. Patent portfolio overlap requiring careful review\n3. Employee contract harmonization issues\n\nWe recommend proceeding with caution and implementing the phased approach we discussed.\n\nThis communication is protected by attorney-client privilege.",
      "meta": {
        "document_id": "legal_001",
        "author": "legal.counsel@company.com",
        "recipient": "ceo@company.com",
        "date": "2001-04-15",
        "document_type": "email"
      }
    }
  ],
  "user_prompt": "Focus on attorney-client privilege and merger-related responsiveness",
  "discovery_context": "All communications and documents related to merger activities, acquisition discussions, and strategic planning between January 2001 and December 2001",
  "include_detailed_reasoning": true,
  "include_topic_analysis": true,
  "response_format": "comprehensive"
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "classification": "Legal Communication",
    "confidence": 0.98,
    "responsiveness": "RESPONSIVE",
    "privilege": "Attorney-Client Privileged",
    "confidentiality": "Highly Confidential",
    "document_type": "Email",
    "business_relevance": "Critical",
    "contains_pii": "No",
    "topic_analysis": {
      "primary_topic": "Merger Legal Strategy",
      "subject_matter": "Legal Analysis and Risk Assessment",
      "secondary_topics": ["Regulatory Compliance", "Due Diligence", "Market Analysis"],
      "key_concepts": ["attorney-client privilege", "merger", "FTC approval", "regulatory hurdles", "due diligence"]
    },
    "reasoning": {
      "responsiveness_reasoning": "Document directly relates to merger activities as specified in discovery context. The communication discusses merger strategy, legal analysis, and regulatory concerns, all of which fall within the scope of merger-related discovery requests.",
      "context_analysis": "Both document and discovery context focus on merger activities and strategic planning within the specified timeframe (2001). The domains match perfectly - business/legal merger communications requested and business/legal merger communications found.",
      "privilege_reasoning": "Document is clearly marked as 'CONFIDENTIAL ATTORNEY-CLIENT COMMUNICATION' and contains legal advice from counsel to CEO regarding merger strategy. The communication is protected by attorney-client privilege as it involves confidential legal advice.",
      "sensitivity_notes": "Document contains strategic legal analysis that could impact merger negotiations if disclosed. Privilege log entry required."
    }
  }
}
```

#### Example 2: Non-Responsive Classification
**Request:**
```json
POST /classify
{
  "documents": [
    {
      "content": "Subject: Cricket Tournament Results\n\nGreat news! Our company cricket team won the inter-corporate tournament this weekend. The final score was 185-4 against TechCorp United. \n\nMatch highlights:\n- John Smith scored 67 not out\n- Sarah Johnson took 3 wickets\n- Excellent team performance overall\n\nNext tournament is scheduled for September. Let me know if you want to join the team for practice sessions.\n\nCheers,\nSports Committee",
      "meta": {
        "document_id": "sports_001",
        "author": "sports.committee@company.com",
        "date": "2001-05-20"
      }
    }
  ],
  "user_prompt": "Analyze for business relevance and merger responsiveness",
  "discovery_context": "All communications and documents related to merger activities, acquisition discussions, due diligence, and strategic business planning",
  "include_detailed_reasoning": true,
  "response_format": "standard"
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "classification": "Recreational Communication", 
    "confidence": 0.95,
    "responsiveness": "NON-RESPONSIVE",
    "privilege": "Not Privileged",
    "confidentiality": "Internal",
    "document_type": "Email",
    "business_relevance": "None",
    "contains_pii": "No",
    "reasoning": {
      "responsiveness_reasoning": "Document content focuses on sports/recreational activities (cricket tournament) while discovery context seeks business/legal merger-related communications. These are from different domains - sports vs business/legal. Per domain separation rules, sports documents are NON-RESPONSIVE to business/legal discovery requests.",
      "context_analysis": "Clear domain mismatch: Document discusses recreational cricket activities while discovery seeks merger/acquisition business communications. No connection between cricket tournament results and merger activities, due diligence, or strategic business planning."
    }
  }
}
```

### NER Extraction Examples

#### Example 1: Comprehensive Entity Extraction
**Request:**
```json
POST /ner/extract
{
  "text": "Please contact John Smith, Senior Vice President at ACME Corporation, regarding the quarterly meeting scheduled for January 15th, 2024 at 2:00 PM EST. His direct email is john.smith@acme.com and office phone is (555) 123-4567. The meeting will be held at our Chicago office, located at 123 Business Plaza, Chicago, IL 60601. We'll be discussing the $2.5 million budget allocation for the Q1 marketing campaign.",
  "entity_types": ["PERSON", "ORGANIZATION", "LOCATION", "EMAIL", "PHONE", "DATE", "TIME", "MONEY", "ADDRESS"],
  "min_score": 0.7,
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
        {
          "text": "John Smith",
          "confidence": 0.99,
          "start": 15,
          "end": 25
        }
      ],
      "ORGANIZATION": [
        {
          "text": "ACME Corporation",
          "confidence": 0.97,
          "start": 50,
          "end": 66
        }
      ],
      "LOCATION": [
        {
          "text": "Chicago",
          "confidence": 0.95,
          "start": 285,
          "end": 292
        },
        {
          "text": "Chicago, IL 60601",
          "confidence": 0.92,
          "start": 330,
          "end": 347
        }
      ],
      "EMAIL": [
        {
          "text": "john.smith@acme.com",
          "confidence": 0.99,
          "start": 190,
          "end": 209
        }
      ],
      "PHONE": [
        {
          "text": "(555) 123-4567",
          "confidence": 0.98,
          "start": 230,
          "end": 244
        }
      ],
      "DATE": [
        {
          "text": "January 15th, 2024",
          "confidence": 0.96,
          "start": 130,
          "end": 148
        }
      ],
      "MONEY": [
        {
          "text": "$2.5 million",
          "confidence": 0.94,
          "start": 380,
          "end": 392
        }
      ]
    },
    "statistics": {
      "total_entities": 7,
      "entity_types": 6,
      "avg_confidence": 0.96
    },
    "method": "bert_model",
    "position_data": [
      {
        "text": "John Smith",
        "label": "PERSON",
        "start": 15,
        "end": 25,
        "confidence": 0.99
      }
    ]
  }
}
```

#### Example 2: PII-Focused Extraction
**Request:**
```json
POST /ner/extract
{
  "text": "Employee Record: Sarah Johnson (SSN: 123-45-6789) works in the Legal Department. Her employee ID is EMP-2001-0547. Contact information: sarah.johnson@company.com, mobile: (555) 987-6543. Emergency contact: Michael Johnson at (555) 987-1234. Address: 456 Oak Street, Springfield, IL 62701. Date of birth: March 12, 1975.",
  "entity_types": ["PERSON", "EMAIL", "PHONE", "SSN", "EMPLOYEE_ID", "ADDRESS", "DATE"],
  "include_pii": true,
  "min_score": 0.8,
  "method": "bert"
}
```

### Summarization Examples

#### Example 1: Business Focus Summarization
**Request:**
```json
POST /summarize
{
  "text": "The Q2 2001 board meeting covered several critical business initiatives and strategic decisions. Revenue performance exceeded expectations with a 22% increase compared to Q1, driven primarily by strong sales in the technology and healthcare sectors. The sales team reported successful contract negotiations with three major clients: TechStart Industries ($1.2M), HealthCorp Solutions ($800K), and Innovation Partners ($950K). Marketing initiatives have generated a 35% increase in qualified leads, with conversion rates improving from 12% to 18%. The operations team highlighted successful implementation of the new customer relationship management system, which has reduced response times by 40% and improved customer satisfaction scores from 7.2 to 8.6 out of 10. Looking ahead to Q3, the company plans to expand into the European market, with initial focus on Germany and France. The board approved a $3.5M investment in international expansion, including hiring 15 new sales and support staff. Technology infrastructure upgrades are scheduled for August, including server capacity expansion and security enhancements. The finance team reported strong cash flow position and recommended increasing the quarterly dividend by $0.05 per share.",
  "length": "medium",
  "focus": "business",
  "extract_keywords": true
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "summary": "The Q2 2001 board meeting highlighted strong business performance with 22% revenue growth driven by technology and healthcare sectors. Key achievements include successful major client contracts totaling $2.95M, improved lead conversion rates from 12% to 18%, and enhanced customer satisfaction through new CRM implementation. The board approved $3.5M for European expansion targeting Germany and France, along with hiring 15 new staff and planned August technology upgrades.",
    "insights": "Analyzed 12 sentences with business focus, emphasizing financial performance, operational improvements, and strategic expansion plans",
    "keywords": ["revenue", "growth", "clients", "expansion", "technology", "healthcare", "marketing", "investment", "European", "quarterly"]
  }
}
```

#### Example 2: BART-Only Summarization
**Request:**
```json
POST /summarize/bart-only
{
  "email_text": "Subject: Urgent Contract Review Required\n\nTeam,\n\nI need your immediate attention on the TechCorp contract review. The legal department has identified several clauses that require modification before we can proceed with signing. Specifically:\n\n1. Liability limitations need to be expanded to include indirect damages\n2. Termination clause requires 90-day notice instead of 30-day\n3. Intellectual property rights section needs clarification on derivative works\n4. Payment terms should be adjusted to net-45 instead of net-30\n\nThe client is expecting our response by Friday, so we need to move quickly. I've scheduled a conference call for tomorrow at 2 PM EST to discuss these changes. Please review the attached marked-up contract and come prepared with your recommendations.\n\nThis contract is worth $2.3M annually and is critical for our Q3 revenue targets.\n\nBest regards,\nContract Management Team",
  "summary_type": "business",
  "max_length": 130,
  "min_length": 50
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "summary": "Legal team requires urgent contract modifications for $2.3M TechCorp deal including expanded liability terms, extended termination notice, IP clarification, and adjusted payment terms. Conference call scheduled for tomorrow 2 PM EST with Friday deadline for client response.",
    "business_facts": {
      "key_points": [
        "Contract value: $2.3M annually",
        "Critical for Q3 revenue targets",
        "Four specific legal modifications required",
        "Friday deadline for client response"
      ],
      "action_items": [
        "Attend conference call tomorrow 2 PM EST",
        "Review marked-up contract attachment",
        "Prepare modification recommendations"
      ]
    },
    "model_used": "BART-only (facebook/bart-large-cnn)",
    "processing_time": 0.8,
    "confidence_score": 0.91,
    "summary_type": "business",
    "comparison_note": "This endpoint uses BART only - no Ollama integration for pure model comparison"
  }
}
```

### Document Search Examples

#### Example 1: Complex Filter Search
**Request:**
```json
POST /search-direct-index
{
  "query": "contract negotiation",
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
    "participants": {
      "wildcard": "*legal@company.com*"
    },
    "priority": ["high", "urgent"],
    "contains_financial_terms": true
  },
  "fuzzy": true,
  "top_k": 25
}
```

## Python Integration Examples

### Complete Family QA Integration
```python
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional

class FamilyQAClient:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        
    def family_qa_with_thread_filter(self, 
                                   query: str,
                                   thread_id: str = None,
                                   author_filter: List[str] = None,
                                   date_range: Dict[str, str] = None) -> Dict[str, Any]:
        """Family QA with thread and participant filtering"""
        
        filters = {}
        if thread_id:
            filters["thread_id"] = thread_id
        if author_filter:
            filters["author"] = author_filter
        if date_range:
            filters["created_date"] = {"range": date_range}
            
        data = {
            "query": query,
            "index_name": "deephousedeephouse_ediscovery_docs_chunks",
            "filters": filters,
            "top_k": 15
        }
        
        response = requests.post(f"{self.base_url}/qa/family", json=data)
        return response.json()
    
    def analyze_email_thread(self, thread_id: str, analysis_focus: str = None) -> Dict[str, Any]:
        """Comprehensive analysis of an email thread"""
        
        # Base query for thread analysis
        if analysis_focus:
            query = f"Analyze this email thread focusing on {analysis_focus}. What are the key points, decisions, and outcomes?"
        else:
            query = "What are the main discussion points, decisions made, and key outcomes in this email thread?"
        
        filters = {"thread_id": thread_id}
        
        data = {
            "query": query,
            "index_name": "deephousedeephouse_ediscovery_docs_chunks", 
            "filters": filters,
            "top_k": 20
        }
        
        response = requests.post(f"{self.base_url}/qa/family", json=data)
        return response.json()
    
    def legal_discovery_family_qa(self,
                                 query: str,
                                 privilege_filter: bool = True,
                                 date_range: Dict[str, str] = None,
                                 participants: List[str] = None) -> Dict[str, Any]:
        """Family QA optimized for legal discovery"""
        
        filters = {}
        
        # Filter out privileged communications if requested
        if privilege_filter:
            filters["contains_privileged"] = False
            filters["privilege_status"] = "Not Privileged"
        
        # Add date range for discovery period
        if date_range:
            filters["created_date"] = {"range": date_range}
        
        # Filter by participants
        if participants:
            filters["participants"] = participants
            
        # Add discovery-specific filters
        filters["document_type"] = ["email", "document", "attachment"]
        
        data = {
            "query": f"In the context of legal discovery: {query}",
            "index_name": "deephousedeephouse_ediscovery_docs_chunks",
            "filters": filters,
            "top_k": 25
        }
        
        response = requests.post(f"{self.base_url}/qa/family", json=data)
        result = response.json()
        
        # Add discovery metadata
        if result.get("success"):
            result["discovery_metadata"] = {
                "privilege_filtered": privilege_filter,
                "date_range": date_range,
                "participants_filtered": bool(participants),
                "discovery_optimized": True
            }
        
        return result

# Usage Examples
client = FamilyQAClient()

# Example 1: Analyze specific email thread
thread_analysis = client.analyze_email_thread(
    thread_id="merger_discussion_thread_001",
    analysis_focus="merger timeline and key decisions"
)

# Example 2: Legal discovery with privilege filtering
discovery_result = client.legal_discovery_family_qa(
    query="What information was shared about the acquisition timeline?",
    privilege_filter=True,
    date_range={
        "gte": "2001-01-01T00:00:00",
        "lte": "2001-06-30T23:59:59"
    },
    participants=["business.dev@company.com", "ceo@company.com"]
)

# Example 3: Family QA with specific participants
participant_discussion = client.family_qa_with_thread_filter(
    query="What concerns were raised about the project budget?",
    author_filter=["finance@company.com", "project.manager@company.com"],
    date_range={
        "gte": "2001-03-01T00:00:00", 
        "lte": "2001-03-31T23:59:59"
    }
)

print("Thread Analysis Result:", thread_analysis.get("result", {}).get("answer", ""))
print("Discovery Result:", discovery_result.get("result", {}).get("answer", ""))
print("Participant Discussion:", participant_discussion.get("result", {}).get("answer", ""))
```

## JavaScript Family QA Integration

```javascript
class FamilyQAClient {
    constructor(baseUrl = 'http://localhost:8001') {
        this.baseUrl = baseUrl;
    }

    async familyQAWithFilters(query, options = {}) {
        const {
            threadId = null,
            authorFilter = null,
            dateRange = null,
            documentTypes = ['email', 'document'],
            topK = 15
        } = options;

        const filters = {};
        if (threadId) filters.thread_id = threadId;
        if (authorFilter) filters.author = authorFilter;
        if (dateRange) filters.created_date = { range: dateRange };
        filters.document_type = documentTypes;

        const data = {
            query,
            index_name: 'deephousedeephouse_ediscovery_docs_chunks',
            filters,
            top_k: topK
        };

        try {
            const response = await fetch(`${this.baseUrl}/qa/family`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            return await response.json();
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    async analyzeEmailChain(threadId, analysisType = 'comprehensive') {
        const queryMap = {
            'comprehensive': 'Provide a comprehensive analysis of this email chain including key topics, decisions, participants, and outcomes',
            'decisions': 'What decisions were made in this email chain and who made them?',
            'timeline': 'What is the timeline of events and discussions in this email chain?',
            'concerns': 'What concerns, issues, or problems were raised in this email chain?'
        };

        const query = queryMap[analysisType] || queryMap['comprehensive'];

        return await this.familyQAWithFilters(query, {
            threadId,
            topK: 20
        });
    }
}

// Usage Examples
const client = new FamilyQAClient();

// Example 1: Comprehensive email chain analysis
const chainAnalysis = await client.analyzeEmailChain(
    'budget_planning_chain_2001', 
    'comprehensive'
);

// Example 2: Decision-focused analysis
const decisionAnalysis = await client.analyzeEmailChain(
    'merger_negotiation_thread',
    'decisions'
);

// Example 3: Custom filtered family QA
const customAnalysis = await client.familyQAWithFilters(
    'What technical requirements were discussed?',
    {
        authorFilter: ['tech.lead@company.com', 'engineering@company.com'],
        dateRange: {
            gte: '2001-04-01T00:00:00',
            lte: '2001-04-30T23:59:59'
        },
        documentTypes: ['email', 'specification', 'document'],
        topK: 25
    }
);

console.log('Chain Analysis:', chainAnalysis);
console.log('Decision Analysis:', decisionAnalysis);
console.log('Custom Analysis:', customAnalysis);
```

These examples demonstrate the full capabilities of the Family QA system and how it can be integrated for various use cases including legal discovery, business analysis, and technical discussions.
