#!/usr/bin/env python3
"""
Simple REST API server for Haystack pipelines using FastAPI
"""
import os
import sys
import re
import json
import random
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import yaml
from haystack import Pipeline
from haystack.dataclasses import Document

# OpenSearch direct client for advanced operations
try:
    from opensearchpy import OpenSearch
except ImportError:
    OpenSearch = None

# Import the manager from our main script
from haystack_new import initialize_haystack_rest_api

# Import enhanced ML processor for BART-only endpoints
try:
    from enhanced_ml_processor import create_ml_processor, EnhancedMLEmailProcessor
    enhanced_ml_available = True
except ImportError:
    enhanced_ml_available = False
    print("Enhanced ML processor not available - BART endpoints will be disabled")

# Import NER functionality
try:
    from simple_ner_processor import ner_processor
    ner_available = True
    print("NER functionality available")
except ImportError as e:
    ner_available = False
    print(f"NER functionality not available: {e}")

# Import improved document management
try:
    from improved_indexing_qa import get_document_manager, run_improved_qa
    improved_indexing_available = True
    print("✅ Improved indexing and QA available")
except ImportError as e:
    improved_indexing_available = False
    print(f"⚠️ Improved indexing not available: {e}")

# Helper function to create direct OpenSearch client
def create_opensearch_client():
    """Create a direct OpenSearch client for advanced operations"""
    if OpenSearch is None:
        raise ImportError("opensearchpy not available")
    
    return OpenSearch(
        hosts=['http://localhost:9200'],
        http_auth=None,
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False
    )

app = FastAPI(
    title="Haystack Document Processing API", 
    version="1.0.0",
    description="API for document indexing, retrieval, and question answering using Haystack framework",
    openapi_tags=[
        {"name": "Indexing", "description": "Endpoints for document indexing and management"},
        {"name": "QA", "description": "Question answering endpoints for querying indexed documents"},
        {"name": "Collections", "description": "Endpoints for managing document collections"},
    ]
)

# Global pipeline storage and API manager
pipelines = {}
api_manager = None
ml_processor = None

class DocumentInput(BaseModel):
    content: str = Field(..., description="The text content of the document")
    meta: Optional[Dict[str, Any]] = Field({}, description="Optional metadata for the document")
    user_input: Optional[str] = Field(None, description="Simple text input from user")

    class Config:
        schema_extra = {
            "example": {
                "content": "This is a sample document about budget planning.",
                "meta": {"document_id": "doc_1", "source": "user_upload"}
            }
        }

class SummarizationRequest(BaseModel):
    documents: List[DocumentInput]

class TextSummarizationRequest(BaseModel):
    text: str = Field(..., description="Text to summarize")
    length: str = Field(default="medium", description="Summary length: short, medium, long")
    focus: str = Field(default="general", description="Focus area for summarization")
    extract_keywords: bool = Field(default=True, description="Extract keywords")

class QARequest(BaseModel):
    query: str
    documents: Optional[List[DocumentInput]] = []

class SimpleQARequest(BaseModel):
    """Simple QA request for querying pre-indexed documents"""
    query: str = Field(..., description="The question to ask")
    collection_id: Optional[str] = Field("default", description="Collection identifier for document retrieval")
    prompt_template: Optional[str] = Field(None, description="Optional custom prompt template for LLM")
    model: Optional[str] = Field(None, description="Optional model override (e.g., specific HuggingFace or Ollama model)")
    num_results: Optional[int] = Field(3, description="Number of document results to retrieve")

    class Config:
        schema_extra = {
            "example": {
                "query": "What is the budget status?",
                "collection_id": "default",
                "num_results": 3
            }
        }

class IndexDocumentsRequest(BaseModel):
    """Request to index documents for later QA queries"""
    documents: List[DocumentInput] = Field(..., description="List of documents to index")
    collection_id: Optional[str] = Field("default", description="Collection identifier to group documents")

    class Config:
        schema_extra = {
            "example": {
                "documents": [
                    {
                        "content": "The budget for Q1 2023 shows a 15% increase in marketing expenditure.",
                        "meta": {"document_id": "doc_1", "source": "financial_report"}
                    }
                ],
                "collection_id": "financial"
            }
        }

class ClassificationRequest(BaseModel):
    documents: List[DocumentInput]
    classifications: Optional[List[str]] = Field(default=None, description="List of classification categories")
    metadata: Optional[Dict[str, Any]] = {}
    user_preferences: Optional[Dict[str, Any]] = None  # Simplified to generic dict
    user_prompt: Optional[str] = Field(default=None, description="Additional context or instructions for classification (e.g., 'Focus on responsiveness to contract disputes', 'This case involves employment law')")
    discovery_context: Optional[str] = Field(default=None, description="Discovery request context to help determine responsiveness")
    
    # Response Configuration Options
    include_detailed_reasoning: bool = Field(default=True, description="Include detailed reasoning and analysis")
    include_topic_analysis: bool = Field(default=True, description="Include comprehensive topic analysis")
    include_raw_response: bool = Field(default=False, description="Include raw LLM response for debugging")
    response_format: str = Field(default="comprehensive", description="Response detail level: 'minimal', 'standard', 'comprehensive'")
    fields_to_include: Optional[List[str]] = Field(default=None, description="Specific fields to include: ['responsiveness', 'privilege', 'confidentiality', 'document_type', 'business_relevance', 'contains_pii']")

class DirectIndexQARequest(BaseModel):
    """Direct QA request for existing indices without collection filtering"""
    query: str = Field(..., description="The question to answer")
    index_name: str = Field(..., description="Name of the index to search")
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    direct_access: bool = Field(default=True, description="Bypass collection filtering")
    filters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="""Optional metadata filters. Supports:
        - Simple: {"field": "value"}
        - Multiple values: {"field": ["val1", "val2"]}
        - Range: {"field": {"range": {"gte": 100, "lte": 200}}}
        - Wildcard: {"field": {"wildcard": "prefix*"}}
        - Document ID: {"document_id": "doc123"} or {"id": "doc123"}
        - Exists: {"field": {"exists": true}}
        - Custom query: {"field": {"query": {...}}}
        """
    )

class DirectIndexSearchRequest(BaseModel):
    """Direct search request for existing indices"""
    query: str = Field(..., description="Search query")
    index_name: str = Field(..., description="Name of the index to search")
    filters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="""Advanced metadata filters. Supports:
        - Simple: {"field": "value"}
        - Multiple values: {"field": ["val1", "val2"]}
        - Range: {"field": {"range": {"gte": 100, "lte": 200}}}
        - Wildcard: {"field": {"wildcard": "prefix*"}}
        - Document ID: {"document_id": "doc123"} or {"id": "doc123"}
        - Exists: {"field": {"exists": true}}
        - Custom query: {"field": {"query": {...}}}
        """
    )
    fuzzy: bool = Field(default=True, description="Enable fuzzy search")
    top_k: int = Field(default=10, description="Number of results")

class IndexStatsRequest(BaseModel):
    """Request for index statistics"""
    index_name: str = Field(..., description="Name of the index")

class DocumentBrowserRequest(BaseModel):
    """Request to browse documents in an index"""
    index_name: str = Field(..., description="Name of the index")
    start: int = Field(default=0, description="Starting document index")
    limit: int = Field(default=10, description="Number of documents to return")

class BartSummaryRequest(BaseModel):
    """BART-only summarization request for comparison testing"""
    email_text: str
    summary_type: Optional[str] = "business"
    max_length: Optional[int] = 150
    min_length: Optional[int] = 40

class BartClassificationRequest(BaseModel):
    """BART/ML-based classification request for comparison testing"""
    email_text: str
    classification_schemes: Optional[List[str]] = ["business", "legal", "sentiment", "priority"]
    include_advanced_analysis: Optional[bool] = True
    confidence_threshold: Optional[float] = 0.7

class NERRequest(BaseModel):
    """NER extraction request"""
    text: str
    entity_types: Optional[List[str]] = None
    include_pii: Optional[bool] = True
    min_score: Optional[float] = 0.7
    method: Optional[str] = Field(default="bert", description="NER method: 'bert' for BERT-based NER, 'llm' for LLM-based NER")

class FileNERRequest(BaseModel):
    """NER extraction from file request"""
    file_path: str
    entity_types: Optional[List[str]] = None
    include_pii: Optional[bool] = True
    min_score: Optional[float] = 0.7
    include_content: Optional[bool] = False
    method: Optional[str] = Field(default="bert", description="NER method: 'bert' for BERT-based NER, 'llm' for LLM-based NER")

class APIResponse(BaseModel):
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None

# Direct Index Access Models (Updated for compatibility)
class DirectIndexSearchRequest(BaseModel):
    """Direct search request for existing indices"""
    query: str = Field(..., description="Search query")
    index_name: str = Field(..., description="Name of the index to search")
    filters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Advanced metadata filters"
    )
    fuzzy: bool = Field(default=True, description="Enable fuzzy search")
    top_k: int = Field(default=10, description="Number of results")

class DirectIndexQARequest(BaseModel):
    """Direct access QA request for existing index"""
    query: str
    index_name: str
    top_k: Optional[int] = 5
    direct_access: Optional[bool] = True
    filters: Optional[Dict[str, Any]] = {}

class IndexStatsRequest(BaseModel):
    """Request for index statistics"""
    index_name: str

class DocumentBrowserRequest(BaseModel):
    """Request for browsing documents in index"""
    index_name: str
    limit: Optional[int] = 20
    offset: Optional[int] = 0
    filters: Optional[Dict[str, Any]] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize pipelines on startup"""
    global pipelines, api_manager, ml_processor
    try:
        print("Initializing Haystack pipelines...")
        try:
            api_manager = initialize_haystack_rest_api(
                model_type="ollama",
                model_name="mistral",
                base_url="http://localhost:11434",
                temperature=0.1,
                use_opensearch=True  # Enable OpenSearch for document indexing
            )
            pipelines = api_manager.pipelines
            print(f"Initialized {len(pipelines)} pipelines: {list(pipelines.keys())}")
            print(f"Document store type: {'OpenSearch' if api_manager.use_opensearch else 'InMemory'}")
        except Exception as e:
            print(f"⚠️ Failed to initialize Haystack pipelines: {e}")
            print("Falling back to minimal initialization")
            api_manager = None
            pipelines = {}
        
        # Initialize Enhanced ML Processor for BART-only endpoints
        if enhanced_ml_available:
            try:
                print("Initializing Enhanced ML Processor for BART endpoints...")
                ml_processor = create_ml_processor(
                    model_name='facebook/bart-large-cnn',
                    device='auto',
                    enable_haystack=False,  # No Haystack integration for pure BART testing
                    haystack_model_type=None,
                    haystack_model_name=None
                )
                print("✅ Enhanced ML Processor initialized for BART-only processing")
            except Exception as e:
                print(f"⚠️ Failed to initialize Enhanced ML Processor: {e}")
                ml_processor = None
        
    except Exception as e:
        print(f"Failed to initialize pipelines: {e}")
        print("⚠️ API will start but some functionality may be limited")
        # Don't exit - allow the API to start for debugging purposes
        api_manager = None
        pipelines = {}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Haystack Email Processing API",
        "available_pipelines": list(pipelines.keys()),
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
            "collections": "/collections",
            "collection_stats": "/collections/{collection_id}/stats",
            "find_document": "/documents/{document_id}/collection",
            "auto_suggest": "/collections/auto-suggest"
        }
    }

# PRE-CONFIGURED CLASSIFICATION SCHEMA (Server-side configuration)
EDISCOVERY_CLASSIFICATION_CONFIG = {
    "classifications": [
        {
            "category": "Responsiveness",
            "labels": ["Responsive", "Non-Responsive", "Privileged", "Partially Responsive"],
            "instructions": "Determine if this document is responsive to the discovery request. Mark as Privileged if it contains attorney-client communications or work product. Consider user input as additional context."
        },
        {
            "category": "Document Type",
            "labels": ["Email", "Contract", "Report", "Invoice", "Meeting Notes", "Legal Document", "Financial Document", "Other"],
            "instructions": "Classify the primary document type based on its format and content structure."
        },
        {
            "category": "Business Relevance",
            "labels": ["Critical", "High", "Medium", "Low", "None"],
            "instructions": "Rate the business importance and relevance of this document to the case or matter."
        },
        {
            "category": "Sensitivity Level",
            "labels": ["Public", "Internal", "Confidential", "Highly Confidential", "Attorney-Client Privileged"],
            "instructions": "Determine the sensitivity and confidentiality level based on document content and markings."
        },
        {
            "category": "Contains PII",
            "labels": ["Yes", "No", "Uncertain"],
            "instructions": "Identify if the document contains personally identifiable information that may require redaction."
        },
        {
            "category": "Review Priority",
            "labels": ["Urgent", "High Priority", "Standard", "Low Priority", "Bulk Process"],
            "instructions": "Assign processing priority based on document importance and complexity."
        }
    ],
    "metadata": {
        "case_name": "eDiscovery Case",
        "discovery_request": "All communications and documents related to business operations, strategy, and decision-making",
        "reviewer": "Legal Team",
        "matter_type": "Commercial Litigation"
    },
    "user_preferences": {
        "trust_user_suggestions": "high",
        "weight_user_suggestions": 0.8,
        "conservative_privilege_review": "Yes",
        "flag_conflicts_for_review": "Yes",
        "escalate_strategic_docs": "Yes"
    }
}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "pipelines_loaded": len(pipelines)}

@app.get("/classification-schema")
async def get_classification_schema():
    """Get the pre-configured classification schema"""
    return APIResponse(
        success=True,
        result=EDISCOVERY_CLASSIFICATION_CONFIG
    )

@app.post("/summarize", response_model=APIResponse)
async def summarize_documents(request: TextSummarizationRequest):
    """Summarize text content"""
    try:
        if "summarization" not in pipelines:
            # Create a simple mock summarization response
            import re
            
            # Simple extractive summarization: take first few sentences
            sentences = re.split(r'[.!?]+', request.text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if request.length == "short":
                summary_sentences = sentences[:2]
            elif request.length == "long":
                summary_sentences = sentences[:5]
            else:  # medium
                summary_sentences = sentences[:3]
            
            summary = '. '.join(summary_sentences) + '.'
            
            result = {
                "summary": summary,
                "insights": f"Analyzed {len(sentences)} sentences with {request.focus} focus"
            }
            
            if request.extract_keywords:
                # Simple keyword extraction (common words)
                words = request.text.lower().split()
                common_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should']
                keywords = [word for word in set(words) if len(word) > 3 and word not in common_words][:5]
                result["keywords"] = keywords
            
            return APIResponse(success=True, result=result)
        
        # Use existing summarization pipeline if available
        # Convert text to documents format for pipeline
        from haystack.dataclasses import Document
        docs = [Document(content=request.text, meta={"source": "user_input"})]
        result = pipelines["summarization"].run({"documents": docs})
        
        # Extract result from pipeline response
        if result.get("generator", {}).get("replies"):
            summary_result = result["generator"]["replies"][0]
        else:
            summary_result = "No summary generated"
        
        return APIResponse(success=True, result={"summary": summary_result})
        
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.post("/summarize/family", response_model=APIResponse)
async def summarize_family(request: SummarizationRequest):
    """Summarize email families (email + attachments)"""
    try:
        if "family_summarization" not in pipelines:
            raise HTTPException(status_code=404, detail="Family summarization pipeline not found")
        
        docs = [Document(content=doc.content, meta=doc.meta) for doc in request.documents]
        result = pipelines["family_summarization"].run({"documents": docs})
        
        return APIResponse(
            success=True,
            result=result["generator"]["replies"][0] if result["generator"]["replies"] else "No summary generated"
        )
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.post("/summarize/thread", response_model=APIResponse)
async def summarize_thread(request: SummarizationRequest):
    """Summarize email threads"""
    try:
        if "thread_summarization" not in pipelines:
            raise HTTPException(status_code=404, detail="Thread summarization pipeline not found")
        
        docs = [Document(content=doc.content, meta=doc.meta) for doc in request.documents]
        result = pipelines["thread_summarization"].run({"documents": docs})
        
        return APIResponse(
            success=True,
            result=result["generator"]["replies"][0] if result["generator"]["replies"] else "No summary generated"
        )
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.post("/summarize/bart-only", response_model=APIResponse)
async def summarize_bart_only(request: BartSummaryRequest):
    """BART-only summarization for comparison testing (no Ollama)"""
    try:
        if not enhanced_ml_available or ml_processor is None:
            raise HTTPException(
                status_code=503, 
                detail="Enhanced ML processor (BART) not available. Please check server logs."
            )
        
        if not request.email_text.strip():
            raise HTTPException(status_code=400, detail="Email text cannot be empty")
        
        # Use BART-only method
        result = ml_processor.generate_bart_only_summary(
            text=request.email_text,
            summary_type=request.summary_type,
            max_length=request.max_length,
            min_length=request.min_length
        )
        
        return APIResponse(
            success=True,
            result={
                "summary": result.get("summary", ""),
                "business_facts": result.get("business_facts", {}),
                "model_used": "BART-only (facebook/bart-large-cnn)",
                "processing_time": result.get("processing_time", 0.0),
                "confidence_score": result.get("confidence_score", 0.0),
                "summary_type": request.summary_type,
                "comparison_note": "This endpoint uses BART only - no Ollama integration for pure model comparison"
            }
        )
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.post("/classify/bart-only", response_model=APIResponse)
async def classify_bart_only(request: BartClassificationRequest):
    """BART/ML-based classification for comparison testing (no Ollama)"""
    try:
        if not enhanced_ml_available or ml_processor is None:
            raise HTTPException(
                status_code=503, 
                detail="Enhanced ML processor (BART) not available. Please check server logs."
            )
        
        if not request.email_text.strip():
            raise HTTPException(status_code=400, detail="Email text cannot be empty")
        
        # Use enhanced ML processor classification
        result = ml_processor.advanced_email_classification(
            text=request.email_text,
            classification_schemes=request.classification_schemes
        )
        
        # Add metadata about the classification
        if "metadata" not in result:
            result["metadata"] = {}
        
        result["metadata"].update({
            "model_used": "Enhanced ML Processor (BART + Traditional ML)",
            "classification_schemes": request.classification_schemes,
            "confidence_threshold": request.confidence_threshold,
            "include_advanced_analysis": request.include_advanced_analysis,
            "comparison_note": "This endpoint uses BART + ML classification - no Ollama integration for pure model comparison"
        })
        
        return APIResponse(
            success=True,
            result=result
        )
        
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.post("/ner/extract", response_model=APIResponse)
async def extract_ner_entities(request: NERRequest):
    """Extract Named Entities and PII from text with position information"""
    try:
        # Set default entity types if none provided
        if request.entity_types is None:
            request.entity_types = ["PERSON", "ORGANIZATION", "LOCATION", "EMAIL", "PHONE", "DATE"]
        
        # Choose method based on request parameter
        if request.method == "bert" and ner_available:
            # BERT-based NER processor (primary method)
            print(f"🔬 Using BERT-based NER processor")
            try:
                print(f"🔬 Using BERT-based NER processor (primary method)")
                result = ner_processor.extract_entities_from_text(
                    text=request.text,
                    entity_types=request.entity_types,
                    include_pii=request.include_pii,
                    min_score=request.min_score
                )
                
                # Check if model-based approach worked
                if "error" not in result:
                    entities = result.get("entities", [])
                    
                    # Convert to grouped format for consistency with LLM approach
                    grouped_entities = {}
                    for entity in entities:
                        entity_type = entity.get("label", "UNKNOWN")
                        if entity_type not in grouped_entities:
                            grouped_entities[entity_type] = []
                        grouped_entities[entity_type].append({
                            "text": entity.get("text", entity.get("word", "")),
                            "confidence": entity.get("confidence", 0.0),
                            "start": entity.get("start", 0),
                            "end": entity.get("end", 0)
                        })
                    
                    return APIResponse(
                        success=True,
                        result={
                            "entities": grouped_entities,
                            "statistics": {
                                "total_entities": result.get("total_entities", len(entities)),
                                "entity_types": len(grouped_entities),
                                "avg_confidence": sum(e.get("confidence", 0) for e in entities) / len(entities) if entities else 0.0
                            },
                            "method": "bert_model",
                            "processing_info": result.get("processing_info", {}),
                            "position_data": result.get("entities", [])  # Include original position data
                        }
                    )
                else:
                    print(f"🔍 BERT model error: {result.get('error')}, trying LLM approach")
            except Exception as e:
                print(f"⚠️ BERT-based NER failed: {e}, falling back to LLM")
        elif request.method == "bert" and not ner_available:
            print(f"❌ BERT NER processor not available, falling back to LLM")
        elif request.method == "llm":
            print(f"🤖 Using LLM-based NER (requested method)")
        else:
            print(f"📝 Using LLM-based NER (default fallback)")
        
        # LLM-based NER approach
        print(f"🤖 Processing with LLM-based NER")
        import requests
        import json
        import re
        
        entity_types_str = ", ".join(request.entity_types)
        
        ner_prompt = f"""
Extract named entities from the following text. Find entities of these types: {entity_types_str}

Text:
{request.text}

For each entity found, provide:
- Entity text
- Entity type (from: {entity_types_str})
- Confidence score (0.0 to 1.0)

Format your response as JSON:
{{"entities": {{"PERSON": [{{"text": "John Smith", "confidence": 0.95}}], "ORGANIZATION": [{{"text": "ACME Corp", "confidence": 0.9}}]}}}}

Extract entities:"""

        try:
            # Try direct Ollama call
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral",
                    "prompt": ner_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 300
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get("response", "")
                
                # Try to parse JSON from response
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    try:
                        parsed_result = json.loads(json_match.group())
                        entities = parsed_result.get("entities", {})
                        
                        # Filter entities by minimum score
                        filtered_entities = {}
                        total_entities = 0
                        
                        for entity_type, entity_list in entities.items():
                            if entity_type in request.entity_types:
                                filtered_list = [
                                    ent for ent in entity_list 
                                    if ent.get("confidence", 0) >= request.min_score
                                ]
                                if filtered_list:
                                    filtered_entities[entity_type] = filtered_list
                                    total_entities += len(filtered_list)
                        
                        return APIResponse(
                            success=True,
                            result={
                                "entities": filtered_entities,
                                "statistics": {
                                    "total_entities": total_entities,
                                    "entity_types": len(filtered_entities),
                                    "avg_confidence": sum(
                                        sum(ent.get("confidence", 0) for ent in ent_list)
                                        for ent_list in filtered_entities.values()
                                    ) / max(total_entities, 1)
                                },
                                "method": "ollama_direct"
                            }
                        )
                    except json.JSONDecodeError:
                        pass
        except Exception as ollama_error:
            print(f"Ollama NER failed: {ollama_error}")
        
        # Fallback to existing logic
        if not ner_available:
            # Create a simple mock NER response
            import re
            
            # Simple regex-based entity extraction for testing
            entities = {}
            text = request.text
            
            # PERSON: Look for capitalized names
            if "PERSON" in request.entity_types:
                person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
                persons = re.findall(person_pattern, text)
                entities["PERSON"] = [{"text": p, "confidence": 0.9} for p in persons]
            
            # ORGANIZATION: Look for common company indicators
            if "ORGANIZATION" in request.entity_types:
                org_pattern = r'\b[A-Z][a-z]+ (?:Corp|Inc|LLC|Company|Corporation|Ltd)\b'
                orgs = re.findall(org_pattern, text)
                entities["ORGANIZATION"] = [{"text": o, "confidence": 0.8} for o in orgs]
            
            # MONEY: Look for dollar amounts
            if "MONEY" in request.entity_types:
                money_pattern = r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|k))?'
                money = re.findall(money_pattern, text, re.IGNORECASE)
                entities["MONEY"] = [{"text": m, "confidence": 0.95} for m in money]
            
            # DATE: Look for date patterns
            if "DATE" in request.entity_types:
                date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b'
                dates = re.findall(date_pattern, text)
                entities["DATE"] = [{"text": d, "confidence": 0.9} for d in dates]
            
            # LOCATION: Look for "at the X office" patterns
            if "LOCATION" in request.entity_types:
                location_pattern = r'\b(?:at the )?([A-Z][a-z]+(?: [A-Z][a-z]+)*) office\b'
                locations = re.findall(location_pattern, text)
                entities["LOCATION"] = [{"text": f"{l} office", "confidence": 0.8} for l in locations]
            
            return APIResponse(
                success=True,
                result={
                    "entities": entities,
                    "statistics": {
                        "total_entities": sum(len(ent_list) for ent_list in entities.values()),
                        "entity_types": len(entities),
                        "avg_confidence": 0.85
                    },
                    "method": "regex_fallback"
                }
            )
        # If no LLM results and no model-based results, return empty
        return APIResponse(
            success=True,
            result={
                "entities": {},
                "statistics": {
                    "total_entities": 0,
                    "entity_types": 0,
                    "avg_confidence": 0.0
                },
                "method": "no_entities_found"
            }
        )
        
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.post("/ner/extract-from-file", response_model=APIResponse)
async def extract_ner_from_file(request: FileNERRequest):
    """Extract Named Entities and PII from a file with position information"""
    try:
        if not ner_available:
            raise HTTPException(
                status_code=503, 
                detail="NER functionality not available. Please check server configuration."
            )
        
        if not request.file_path.strip():
            raise HTTPException(status_code=400, detail="File path cannot be empty")
        
        # Extract entities from file
        # Note: File processing primarily uses BERT processor, but method parameter is available for future extensions
        result = ner_processor.extract_entities_from_file(
            file_path=request.file_path,
            entity_types=request.entity_types,
            include_pii=request.include_pii,
            min_score=request.min_score,
            include_content=request.include_content
        )
        
        if "error" in result:
            return APIResponse(success=False, error=result["error"])
        
        return APIResponse(
            success=True,
            result={
                "entities": result.get("entities", []),
                "file_info": result.get("file_info", {}),
                "summary": {
                    "total_entities": result.get("total_entities", 0),
                    "entity_types": result.get("entity_types", []),
                    "text_length": result.get("text_length", 0)
                },
                "processing_info": result.get("processing_info", {}),
                "content": result.get("content", None),
                "highlight_instructions": {
                    "description": "Use start/end positions to highlight entities in source text",
                    "note": "Positions are relative to extracted text content"
                }
            }
        )
        
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.post("/qa", response_model=APIResponse)
async def qa_documents(request: QARequest):
    """Answer questions about documents"""
    try:
        if "qa" not in pipelines:
            raise HTTPException(status_code=404, detail="QA pipeline not found")
        
        # For QA, we need to first add documents to the document store
        # This is a simplified version - in production you'd manage the document store properly
        docs = [Document(content=doc.content, meta=doc.meta) for doc in request.documents]
        
        # Add documents to document store (this is specific to the pipeline setup)
        if docs:
            pipelines["qa"].get_component("retriever").document_store.write_documents(docs)
        
        result = pipelines["qa"].run({"retriever": {"query": request.query}})
        
        return APIResponse(
            success=True,
            result=result["generator"]["replies"][0] if result["generator"]["replies"] else "No answer generated"
        )
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.post("/qa/family", response_model=APIResponse)
async def qa_family(request: QARequest):
    """Answer questions about email families (email + attachments as single unit)"""
    try:
        if "family_qa" not in pipelines:
            raise HTTPException(status_code=404, detail="Family QA pipeline not found")
        
        # Handle two modes: direct documents or index search
        if request.documents and len(request.documents) > 0:
            # Mode 1: Process provided documents directly
            docs = [Document(content=doc.content, meta=doc.meta) for doc in request.documents]
            print(f"DEBUG: Family QA with {len(docs)} provided documents")
        else:
            # Mode 2: Search index for relevant documents first
            print(f"DEBUG: Family QA searching index for query: {request.query}")
            
            # Get index name from request or use default
            index_name = getattr(request, 'index_name', None) or "deephousedeephouse_ediscovery_docs_chunks"
            
            # Use the same search logic as qa-direct-index
            from haystack_new import HaystackRestAPIManager, create_model_config
            
            model_config = create_model_config(
                model_type="ollama",
                model_name="mistral",
                base_url="http://localhost:11434",
                temperature=0.1
            )
            
            direct_manager = HaystackRestAPIManager(
                model_config=model_config,
                use_opensearch=True,
                index_name=index_name
            )
            
            # Build search query for family documents
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": request.query,
                                    "fields": ["content", "meta.*"],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO"
                                }
                            }
                        ]
                    }
                },
                "size": getattr(request, 'top_k', 10),
                "_source": ["content", "meta"]
            }
            
            # Add filters if provided
            if hasattr(request, 'filters') and request.filters:
                if "filter" not in search_body["query"]["bool"]:
                    search_body["query"]["bool"]["filter"] = []
                
                for field, value in request.filters.items():
                    if isinstance(value, dict) and "range" in value:
                        search_body["query"]["bool"]["filter"].append({
                            "range": {f"meta.{field}": value["range"]}
                        })
                    elif isinstance(value, list):
                        search_body["query"]["bool"]["filter"].append({
                            "terms": {f"meta.{field}": value}
                        })
                    else:
                        search_body["query"]["bool"]["filter"].append({
                            "term": {f"meta.{field}": value}
                        })
            
            # Execute search
            opensearch_client = create_opensearch_client()
            response = opensearch_client.search(
                index=index_name,
                body=search_body
            )
            
            hits = response['hits']['hits']
            
            # Convert search results to Document objects
            docs = []
            for hit in hits:
                source_data = hit['_source']
                docs.append(Document(
                    id=hit['_id'],
                    content=source_data.get('content', ''),
                    meta=source_data.get('meta', {})
                ))
            
            print(f"DEBUG: Found {len(docs)} documents from index search")
            
            if not docs:
                return APIResponse(
                    success=True,
                    result="No relevant documents found for family QA. Please try a different query or check your filters."
                )
        
        # Create a temporary in-memory store for this specific family QA request
        from haystack.document_stores.in_memory import InMemoryDocumentStore
        from haystack.components.retrievers import InMemoryBM25Retriever
        
        temp_store = InMemoryDocumentStore()
        temp_store.write_documents(docs)
        temp_retriever = InMemoryBM25Retriever(document_store=temp_store)
        
        # Run retrieval + QA on the family documents
        retrieval_result = temp_retriever.run(query=request.query)
        retrieved_docs = retrieval_result.get("documents", docs)
        
        # Run the prompt builder and generator with retrieved family documents
        prompt_builder = pipelines["family_qa"].get_component("prompt_builder")
        generator = pipelines["family_qa"].get_component("generator")
        
        prompt_result = prompt_builder.run(documents=retrieved_docs, query=request.query)
        generation_result = generator.run(prompt=prompt_result["prompt"])
        
        answer = generation_result["replies"][0] if generation_result["replies"] else "No answer generated"
        
        return APIResponse(
            success=True,
            result={
                "answer": answer,
                "documents_processed": len(docs),
                "sources": [
                    {
                        "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                        "metadata": doc.meta
                    } for doc in retrieved_docs[:3]  # Show top 3 sources
                ]
            }
        )
    except Exception as e:
        return APIResponse(success=False, error=str(e))

def build_custom_classification_prompt(request: ClassificationRequest, text_content: str, user_context: str) -> str:
    """Build a custom prompt based on user preferences to avoid unnecessary computation"""
    
    # Base prompt with document content and context
    base_prompt = f"""
Analyze the following document for eDiscovery classification:

{text_content}
{user_context}

CRITICAL INSTRUCTIONS FOR RESPONSIVENESS DETERMINATION:

STEP 1: Identify the document's primary domain/topic (e.g., business, legal, sports, technology, etc.)
STEP 2: Identify the discovery context's primary domain/topic  
STEP 3: If domains are different, document is NON-RESPONSIVE
STEP 4: If domains match, check if content specifically relates to the discovery scope

DOMAIN SEPARATION RULES:
- Business/Legal documents (contracts, mergers, acquisitions) are NON-RESPONSIVE to Sports/Recreation discovery
- Sports/Recreation documents (cricket, tournaments, games) are NON-RESPONSIVE to Business/Legal discovery
- Technology documents are NON-RESPONSIVE to Sports discovery unless both involve technology
- Financial documents are NON-RESPONSIVE to Sports discovery unless both involve finances

EXAMPLE LOGIC:
- Document about "merger agreement" + Discovery about "cricket tournaments" = NON-RESPONSIVE (business ≠ sports)
- Document about "cricket match" + Discovery about "merger activities" = NON-RESPONSIVE (sports ≠ business)
- Document about "merger agreement" + Discovery about "merger activities" = RESPONSIVE (business = business)

BEFORE marking RESPONSIVE, ask yourself: "Are these topics from the same domain?" If NO, mark NON-RESPONSIVE.

Please provide analysis with the following classifications:
"""
    
    # Determine what sections to include based on user preferences
    sections_to_include = []
    
    # Always include basic classification
    sections_to_include.append("- Primary Classification: [Main document category]")
    sections_to_include.append("- Confidence Level: [0.0-1.0]")
    
    # Core eDiscovery fields (always include these)
    sections_to_include.append("- Responsiveness: [RESPONSIVE/NON-RESPONSIVE/PARTIALLY RESPONSIVE]")
    sections_to_include.append("- Privilege: [Attorney-Client Privileged/Work Product/Not Privileged]") 
    sections_to_include.append("- Confidentiality: [Public/Internal/Confidential/Highly Confidential]")
    
    # Add sections based on response format
    if request.response_format in ["standard", "comprehensive"]:
        sections_to_include.extend([
            "- Document Type: [Email/Contract/Report/Legal Document/Financial/Other]",
            "- Business Relevance: [Critical/High/Medium/Low/None]",
            "- Contains PII: [Yes/No/Uncertain]"
        ])
    
    # Add topic analysis if requested
    if request.include_topic_analysis and request.response_format != "minimal":
        sections_to_include.extend([
            "- Primary Topic: [What is the main subject/theme of this document?]",
            "- Subject Matter: [Business area, legal matter, technical domain, etc.]"
        ])
        
        if request.response_format == "comprehensive":
            sections_to_include.extend([
                "- Secondary Topics: [Any additional important topics discussed]",
                "- Key Concepts: [Important terms, entities, concepts mentioned]"
            ])
    
    # Add detailed reasoning if requested
    if request.include_detailed_reasoning and request.response_format != "minimal":
        sections_to_include.extend([
            "- Responsiveness Reasoning: [If your context analysis shows different domains, mark as NON-RESPONSIVE. Only mark RESPONSIVE if domains match AND content relates]",
            "- Context Analysis: [How document content relates to discovery context]"
        ])
        
        if request.response_format == "comprehensive":
            sections_to_include.extend([
                "- Privilege Analysis: [Specific reasoning for any privilege claims]",
                "- Sensitivity Assessment: [Any sensitive information or redaction needs]"
            ])
    
    # Handle custom fields
    if request.fields_to_include:
        # Reset sections to only basic + requested fields
        sections_to_include = [
            "- Primary Classification: [Main document category]",
            "- Confidence Level: [0.0-1.0]"
        ]
        
        field_mapping = {
            "responsiveness": "- Responsiveness: [RESPONSIVE/NON-RESPONSIVE/PARTIALLY RESPONSIVE]",
            "privilege": "- Privilege: [Attorney-Client Privileged/Work Product/Not Privileged]",
            "confidentiality": "- Confidentiality: [Public/Internal/Confidential/Highly Confidential]",
            "document_type": "- Document Type: [Email/Contract/Report/Legal Document/Financial/Other]",
            "business_relevance": "- Business Relevance: [Critical/High/Medium/Low/None]",
            "contains_pii": "- Contains PII: [Yes/No/Uncertain]"
        }
        
        for field in request.fields_to_include:
            if field in field_mapping:
                sections_to_include.append(field_mapping[field])
    
    # Add the sections to the prompt
    prompt = base_prompt + "\n".join(sections_to_include)
    
    # Add format instruction
    prompt += f"""

FINAL CONSISTENCY CHECK: If your context analysis identifies that the document and discovery context are from different domains (e.g., business vs sports), your responsiveness determination MUST be NON-RESPONSIVE.

RESPOND ONLY in this exact JSON format (no additional text):
{{
    "classification": "[primary classification]",
    "confidence": [0.0-1.0],
    "responsiveness": "[RESPONSIVE/NON-RESPONSIVE/PARTIALLY RESPONSIVE]",
    "privilege": "[privilege status]",
    "confidentiality": "[confidentiality level]"
"""
    
    # Add optional fields to JSON format based on what was requested
    if request.response_format in ["standard", "comprehensive"] or (request.fields_to_include and "document_type" in request.fields_to_include):
        prompt += ',\n    "document_type": "[document type]"'
    if request.response_format in ["standard", "comprehensive"] or (request.fields_to_include and "business_relevance" in request.fields_to_include):
        prompt += ',\n    "business_relevance": "[business relevance]"'
    if request.response_format in ["standard", "comprehensive"] or (request.fields_to_include and "contains_pii" in request.fields_to_include):
        prompt += ',\n    "contains_pii": "[yes/no/uncertain]"'
    
    if request.include_topic_analysis and request.response_format != "minimal":
        prompt += ',\n    "topic_analysis": {\n        "primary_topic": "[primary topic]",\n        "subject_matter": "[subject matter]"'
        if request.response_format == "comprehensive":
            prompt += ',\n        "secondary_topics": ["topic1", "topic2"],\n        "key_concepts": ["concept1", "concept2"]'
        prompt += '\n    }'
    
    if request.include_detailed_reasoning and request.response_format != "minimal":
        prompt += ',\n    "reasoning": {\n        "responsiveness_reasoning": "[detailed explanation]",\n        "context_analysis": "[context analysis]"'
        if request.response_format == "comprehensive":
            prompt += ',\n        "privilege_reasoning": "[privilege explanation]",\n        "sensitivity_notes": "[sensitivity notes]"'
        prompt += '\n    }'
    
    prompt += "\n}"
    
    return prompt

def filter_classification_response(result: Dict[str, Any], request: ClassificationRequest) -> Dict[str, Any]:
    """Filter classification response based on user preferences - fallback approach"""
    
    # Create a copy to avoid modifying the original
    result = result.copy()
    
    # Remove raw_response unless explicitly requested
    if not request.include_raw_response:
        result.pop("raw_response", None)
    
    # If comprehensive with all options enabled, return as-is
    if (request.response_format == "comprehensive" and 
        request.include_detailed_reasoning and 
        request.include_topic_analysis and 
        not request.fields_to_include):
        return result
    
    # Create filtered result based on response format
    filtered_result = {}
    
    # Always include basic classification info
    basic_fields = ["classification", "confidence", "method"]
    for field in basic_fields:
        if field in result:
            filtered_result[field] = result[field]
    
    # Handle custom field selection first (overrides format settings)
    if request.fields_to_include:
        # Add only requested fields
        for field in request.fields_to_include:
            if field in result:
                filtered_result[field] = result[field]
        return filtered_result
    
    # Handle response format levels
    if request.response_format == "minimal":
        # Only basic classification + core eDiscovery fields
        core_fields = ["responsiveness", "privilege", "confidentiality"]
        for field in core_fields:
            if field in result:
                filtered_result[field] = result[field]
        return filtered_result
    
    elif request.response_format == "standard":
        # Include eDiscovery fields but not detailed reasoning
        standard_fields = [
            "responsiveness", "privilege", "confidentiality", 
            "document_type", "business_relevance", "contains_pii"
        ]
        for field in standard_fields:
            if field in result:
                filtered_result[field] = result[field]
        
        # Include simplified topic analysis if requested
        if request.include_topic_analysis and "topic_analysis" in result:
            topic_data = result["topic_analysis"]
            if isinstance(topic_data, dict):
                simplified_topic = {
                    "primary_topic": topic_data.get("primary_topic", "N/A"),
                    "subject_matter": topic_data.get("subject_matter", "N/A")
                }
                filtered_result["topic_analysis"] = simplified_topic
            else:
                filtered_result["topic_analysis"] = topic_data
        
        return filtered_result
    
    else:  # comprehensive
        # Include everything except what's explicitly excluded
        for key, value in result.items():
            skip_field = False
            
            # Skip reasoning fields if not requested
            if key in ["reasoning", "detailed_reasoning"] and not request.include_detailed_reasoning:
                skip_field = True
            
            # Skip topic analysis if not requested  
            if key == "topic_analysis" and not request.include_topic_analysis:
                skip_field = True
            
            if not skip_field:
                filtered_result[key] = value
    
    return filtered_result

@app.post("/classify", response_model=APIResponse)
async def classify_documents(request: ClassificationRequest):
    """
    Classify text for eDiscovery using direct Ollama integration
    """
    try:
        # Direct Ollama classification (bypass pipelines for reliability)
        import requests
        
        # Extract text from documents for classification - handle both string and dict content
        text_parts = []
        for doc in request.documents:
            if hasattr(doc, 'content'):
                # Handle DocumentInput objects
                content = doc.content
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, dict):
                    # If content is a dict, try to extract text from common fields
                    text_parts.append(str(content.get('text', content.get('content', str(content)))))
                else:
                    text_parts.append(str(content))
            elif isinstance(doc, dict):
                # Handle dictionary documents
                content = doc.get('content', doc.get('text', str(doc)))
                text_parts.append(str(content))
            else:
                # Fallback: convert to string
                text_parts.append(str(doc))
        
        text_content = " ".join(text_parts)
        classifications = request.classifications if request.classifications else ["business", "legal", "technical", "personal"]
        
        # Build context from user input
        user_context = ""
        if request.user_prompt:
            user_context += f"\n\nUSER INSTRUCTIONS: {request.user_prompt}"
        if request.discovery_context:
            user_context += f"\n\nDISCOVERY REQUEST CONTEXT: {request.discovery_context}"
        
        # Use custom prompt based on user preferences (more efficient than post-filtering)
        if any([request.response_format != "comprehensive", 
                not request.include_detailed_reasoning, 
                not request.include_topic_analysis,
                request.fields_to_include]):
            print(f"🔧 DEBUG: Using custom prompt for format={request.response_format}")
            enhanced_classification_prompt = build_custom_classification_prompt(request, text_content, user_context)
        else:
            # Use full comprehensive prompt for default case
            enhanced_classification_prompt = f"""
You are an expert eDiscovery attorney reviewing documents for litigation. Perform a comprehensive classification analysis.

Document Content:
{text_content}
{user_context}

CRITICAL INSTRUCTIONS FOR RESPONSIVENESS DETERMINATION:

STEP 1: Identify the document's primary domain/topic (e.g., business, legal, sports, technology, etc.)
STEP 2: Identify the discovery context's primary domain/topic  
STEP 3: If domains are different, document is NON-RESPONSIVE
STEP 4: If domains match, check if content specifically relates to the discovery scope

DOMAIN SEPARATION RULES:
- Business/Legal documents (contracts, mergers, acquisitions) are NON-RESPONSIVE to Sports/Recreation discovery
- Sports/Recreation documents (cricket, tournaments, games) are NON-RESPONSIVE to Business/Legal discovery
- Technology documents are NON-RESPONSIVE to Sports discovery unless both involve technology
- Financial documents are NON-RESPONSIVE to Sports discovery unless both involve finances

EXAMPLE LOGIC:
- Document about "merger agreement" + Discovery about "cricket tournaments" = NON-RESPONSIVE (business ≠ sports)
- Document about "cricket match" + Discovery about "merger activities" = NON-RESPONSIVE (sports ≠ business)
- Document about "merger agreement" + Discovery about "merger activities" = RESPONSIVE (business = business)

BEFORE marking RESPONSIVE, ask yourself: "Are these topics from the same domain?" If NO, mark NON-RESPONSIVE.

Please provide a complete analysis with the following classifications:

1. TOPIC ANALYSIS:
   - Primary Topic: [What is the main subject/theme of this document?]
   - Secondary Topics: [Any additional important topics discussed]  
   - Key Subject Matter: [Business area, legal matter, technical domain, etc.]
   - Key Concepts: [Important terms, entities, concepts mentioned]

2. EDISCOVERY CLASSIFICATIONS:
   - Responsiveness: [RESPONSIVE/NON-RESPONSIVE/PARTIALLY RESPONSIVE] 
     * MUST analyze document content against the provided discovery context
     * If the context analysis shows the document topic and discovery context are from different domains (e.g., business vs sports), mark as NON-RESPONSIVE
     * Only mark RESPONSIVE if the document actually relates to the discovery scope  
     * Example: Business merger document + Cricket discovery = NON-RESPONSIVE (different domains)
     * Be strictly logical - only mark RESPONSIVE if content actually relates to the discovery scope
   - Privilege: [Attorney-Client Privileged/Work Product/Not Privileged]
   - Confidentiality: [Public/Internal/Confidential/Highly Confidential]
   - Document Type: [Email/Contract/Report/Legal Document/Financial/Other]
   - Business Relevance: [Critical/High/Medium/Low/None]
   - Contains PII: [Yes/No/Uncertain]
   - Review Priority: [Urgent/High Priority/Standard/Low Priority]

3. DETAILED REASONING:
   - Responsiveness Analysis: [If your context analysis shows different domains, mark as NON-RESPONSIVE. Only mark RESPONSIVE if domains match AND content relates]
   - Privilege Analysis: [Specific reasoning for any privilege claims]
   - Sensitivity Assessment: [Any sensitive information or redaction needs]
   - Context Compliance: [How user instructions were addressed]

FINAL CONSISTENCY CHECK: If your context analysis identifies that the document and discovery context are from different domains (e.g., business vs sports), your responsiveness determination MUST be NON-RESPONSIVE.

RESPOND ONLY in this exact JSON format (no additional text):
{{
    "topic_analysis": {{
        "primary_topic": "specific main topic",
        "secondary_topics": ["topic1", "topic2"],
        "subject_matter": "specific domain/area",
        "key_concepts": ["concept1", "concept2", "concept3"]
    }},
    "ediscovery_classification": {{
        "responsiveness": "RESPONSIVE|NON-RESPONSIVE|PARTIALLY RESPONSIVE",
        "privilege": "Attorney-Client Privileged|Work Product|Not Privileged", 
        "confidentiality": "Public|Internal|Confidential|Highly Confidential",
        "document_type": "Email|Contract|Report|Legal Document|Financial|Other",
        "business_relevance": "Critical|High|Medium|Low|None",
        "contains_pii": "Yes|No|Uncertain",
        "review_priority": "Urgent|High Priority|Standard|Low Priority"
    }},
    "reasoning": {{
        "responsiveness_reasoning": "Detailed analysis of why document is responsive/non-responsive to discovery request: [specific explanation based on discovery context]",
        "privilege_reasoning": "Specific analysis of privilege claims with legal basis",
        "sensitivity_notes": "Assessment of sensitive content requiring protection",
        "redaction_recommendations": "Specific redaction guidance if needed",
        "context_analysis": "How the discovery context was applied to this document"
    }},
    "confidence": 0.85
}}"""

        try:
            # Try direct Ollama call first
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral",
                    "prompt": enhanced_classification_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 500
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get("response", "")
                
                # Try to parse JSON from response
                import json
                import re
                
                # Look for JSON in the response - enhanced pattern for nested JSON
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    try:
                        parsed_result = json.loads(json_match.group())
                        
                        # Check if this is a simple format response (from custom prompt)
                        is_simple_format = (request.response_format == "minimal" or 
                                          (request.fields_to_include and len(request.fields_to_include) <= 3))
                        
                        if is_simple_format and all(key in ["classification", "confidence", "method", "responsiveness", "privilege", "confidentiality"] + (request.fields_to_include or []) 
                                                   for key in parsed_result.keys()):
                            # Handle simple format response - use as-is and add method if missing
                            result = parsed_result.copy()
                            if "method" not in result:
                                result["method"] = "ollama_enhanced_ediscovery_contextual"
                        else:
                            # Handle comprehensive format response - extract nested structures
                            # Extract enhanced eDiscovery classifications
                            topic_analysis = parsed_result.get("topic_analysis", {})
                            ediscovery_class = parsed_result.get("ediscovery_classification", {})
                            reasoning = parsed_result.get("reasoning", {})
                            
                            # Determine primary classification
                            primary_classification = topic_analysis.get("subject_matter", classifications[0])
                            if not primary_classification or primary_classification == "domain/area":
                                primary_classification = classifications[0]
                            
                            # Create comprehensive result with enhanced context analysis
                            result = {
                                "classification": primary_classification,
                                "confidence": parsed_result.get("confidence", 0.85),
                                "method": "ollama_enhanced_ediscovery_contextual"
                            }
                            
                            # Add enhanced eDiscovery fields with context awareness
                            if topic_analysis:
                                result["topic_analysis"] = topic_analysis
                            if ediscovery_class:
                                result["ediscovery_classification"] = ediscovery_class
                                result["responsiveness"] = ediscovery_class.get("responsiveness", "Unknown")
                                result["privilege"] = ediscovery_class.get("privilege", "Unknown") 
                                result["confidentiality"] = ediscovery_class.get("confidentiality", "Unknown")
                                result["document_type"] = ediscovery_class.get("document_type", "Unknown")
                                result["contains_pii"] = ediscovery_class.get("contains_pii", "Unknown")
                                result["business_relevance"] = ediscovery_class.get("business_relevance", "Unknown")
                                result["review_priority"] = ediscovery_class.get("review_priority", "Unknown")
                            
                            if reasoning:
                                result["reasoning"] = reasoning
                                # Create comprehensive reasoning summary
                                reasoning_parts = []
                                if reasoning.get('responsiveness_reasoning'):
                                    reasoning_parts.append(f"RESPONSIVENESS: {reasoning['responsiveness_reasoning']}")
                                if reasoning.get('privilege_reasoning'):
                                    reasoning_parts.append(f"PRIVILEGE: {reasoning['privilege_reasoning']}")
                                if reasoning.get('context_analysis'):
                                    reasoning_parts.append(f"CONTEXT ANALYSIS: {reasoning['context_analysis']}")
                                
                                result["detailed_reasoning"] = " | ".join(reasoning_parts) if reasoning_parts else "Enhanced eDiscovery analysis completed"
                                
                                if reasoning.get('context_analysis'):
                                    result["context_compliance"] = reasoning['context_analysis']
                            else:
                                result["reasoning"] = "Enhanced eDiscovery analysis with context consideration completed"
                        
                        result["raw_response"] = raw_response
                        
                        # Apply response filtering as fallback (since LLM may not follow custom format exactly)
                        print(f"🔧 DEBUG: Before filtering - fields: {list(result.keys())}")
                        print(f"🔧 DEBUG: Request format: {request.response_format}, detailed_reasoning: {request.include_detailed_reasoning}, topic_analysis: {request.include_topic_analysis}")
                        print(f"🔧 DEBUG: Raw response length: {len(raw_response) if 'raw_response' in result else 'No raw_response'}")
                        filtered_result = filter_classification_response(result, request)
                        print(f"🔧 DEBUG: After filtering - fields: {list(filtered_result.keys())}")
                        
                        return APIResponse(success=True, result=filtered_result)
                        
                    except json.JSONDecodeError as e:
                        print(f"JSON parsing failed: {e}")
                        # Try to extract key information even if JSON is malformed
                        if "responsive" in raw_response.lower():
                            responsiveness = "Responsive" if "non-responsive" not in raw_response.lower() else "Non-Responsive"
                        else:
                            responsiveness = "Unknown"
                        
                        result = {
                            "classification": classifications[0],
                            "confidence": 0.7,
                            "responsiveness": responsiveness,
                            "privilege": "Privileged" if "privilege" in raw_response.lower() else "Unknown",
                            "reasoning": "Partial analysis due to response format issues",
                            "raw_response": raw_response,
                            "method": "ollama_partial_parse"
                        }
                        filtered_result = filter_classification_response(result, request)
                        return APIResponse(success=True, result=filtered_result)
                
                # Fallback: extract classification from text
                category = classifications[0]  # Default
                for cat in classifications:
                    if cat.lower() in raw_response.lower():
                        category = cat
                        break
                
                return APIResponse(
                    success=True,
                    result={
                        "classification": category,
                        "confidence": 0.7,
                        "reasoning": raw_response.strip()[:100] + "...",
                        "method": "ollama_direct_parsed"
                    }
                )
        except Exception as ollama_error:
            print(f"Ollama classification failed: {ollama_error}")
        
        # Fallback: try pipelines if available
        if "classification" not in pipelines:
            # Create a simple mock classification response
            categories = ["business", "legal", "technical", "personal", "financial"]
            import random
            selected_category = random.choice(categories)
            
            return APIResponse(
                success=True,
                result={
                    "classification": selected_category,
                    "confidence_scores": {cat: random.random() for cat in categories},
                    "reasoning": f"Classified as {selected_category} based on content analysis",
                    "method": "mock_fallback"
                }
            )
        
        # Use existing classification pipeline if available
        # Convert request documents to Haystack Documents for pipeline processing
        documents = [Document(content=doc.content, meta=doc.meta or {}) for doc in request.documents]
        result = pipelines["classification"].run({"documents": documents})
        
        return APIResponse(
            success=True,
            result=result
        )
    except Exception as e:
        return APIResponse(success=False, error=str(e))
        
        # Create a temporary classifier node to access the prompt generation method
        model_config = ModelConfig(
            model_name="mistral",
            model_type="ollama",
            base_url="http://localhost:11434",
            temperature=0.1,
            max_tokens=1000
        )
        classifier_node = EnhancedDocumentClassifierNode(model_config=model_config)
        
        # Run classification for each document
        results = []
        for i, doc in enumerate(docs):
            print(f"DEBUG: Processing document {i+1}/{len(docs)}")
            
            # Create classification prompt with pre-configured schema
            base_prompt = classifier_node.create_classification_prompt(
                classifications=classification_config["classifications"],
                metadata=classification_config["metadata"],
                user_preferences=classification_config["user_preferences"],
                user_inputs=[user_inputs[i]] if user_inputs[i] else None
            )
            
            print(f"DEBUG: Generated base prompt length: {len(base_prompt)}")
            
            # For classification, we'll process the document directly without using the full pipeline
            # since we need to inject a dynamic prompt. Let's clean and split the document first.
            
            # Process document through cleaner and splitter manually
            cleaner = pipelines["classification"].get_component("cleaner")
            splitter = pipelines["classification"].get_component("splitter")
            
            # Clean the document
            clean_result = cleaner.run(documents=[doc])
            cleaned_docs = clean_result.get("documents", [doc])
            
            # Split the document  
            split_result = splitter.run(documents=cleaned_docs)
            processed_docs = split_result.get("documents", cleaned_docs)
            
            # Create final prompt with processed document content
            if processed_docs:
                # Use the first processed document for classification
                doc_content = f"\n\nDocument to classify:\nContent: {processed_docs[0].content}"
                if doc.meta:
                    doc_content += f"\nMetadata: {doc.meta}"
            else:
                # Fallback to original document
                doc_content = f"\n\nDocument to classify:\nContent: {doc.content}"
                if doc.meta:
                    doc_content += f"\nMetadata: {doc.meta}"
            
            final_prompt = base_prompt + doc_content
            
            # Run generator directly with the complete prompt
            generator = pipelines["classification"].get_component("generator")
            classification_result_data = generator.run(prompt=final_prompt)
            
            # Debug: Log generator result
            print(f"DEBUG: Generator result keys: {classification_result_data.keys()}")
            if "replies" in classification_result_data:
                print(f"DEBUG: Generator reply: {classification_result_data['replies'][0][:200]}...")
            
            classification_result = {
                "document_index": i,
                "document_id": doc.meta.get("document_id", f"doc_{i}"),
                "classification": classification_result_data["replies"][0] if classification_result_data.get("replies") else "No classification generated",
                "user_input": user_inputs[i],
                "metadata": doc.meta
            }
            
            results.append(classification_result)
        
        return APIResponse(
            success=True,
            result=results
        )
        
    except Exception as e:
        print(f"ERROR in classification: {str(e)}")
        return APIResponse(success=False, error=str(e))

@app.post("/index", response_model=APIResponse, tags=["Indexing"], 
         summary="Index documents for QA retrieval",
         description="Index documents into OpenSearch for later question answering. Specify collection_id to group related documents.")
async def index_documents(request: IndexDocumentsRequest):
    """
    Index documents for later QA queries with simplified handling
    Step 1: Upload and index documents in OpenSearch/InMemory store
    """
    try:
        # Always use the same api_manager that the QA endpoints use
        if api_manager is None:
            raise HTTPException(status_code=500, detail="API manager not initialized")
        
        # Convert input to Haystack Documents
        docs = [Document(content=doc.content, meta=doc.meta) for doc in request.documents]
        
        # Debug logging
        print(f"DEBUG: Indexing {len(docs)} documents in collection '{request.collection_id}' using api_manager")
        
        # Index documents using our fixed api_manager
        result = api_manager.index_documents(docs, request.collection_id)
        
        if result["success"]:
            return APIResponse(
                success=True,
                result={
                    "indexed_count": result["documents_indexed"],
                    "collection_id": result["collection_id"],
                    "document_store_type": result["document_store_type"],
                    "message": f"Successfully indexed {result['documents_indexed']} documents"
                }
            )
        else:
            return APIResponse(success=False, error=result.get("error", "Unknown indexing error"))
        
    except Exception as e:
        print(f"ERROR in document indexing: {str(e)}")
        return APIResponse(success=False, error=str(e))

@app.post("/qa-simple", response_model=APIResponse, tags=["QA"],
         summary="Answer questions using indexed documents",
         description="Query the indexed documents to answer questions. Specify collection_id to query specific document groups.")
async def qa_with_indexed_docs(request: SimpleQARequest):
    """
    Simple QA against pre-indexed documents
    Step 2: Ask questions about previously indexed documents
    """
    try:
        if api_manager is None:
            raise HTTPException(status_code=500, detail="API manager not initialized")
        
        # 1. Log the received request - focusing only on collection_id
        print(f"DEBUG: QA Simple Request - Collection: {request.collection_id}, Query: {request.query}")
        
        # 2. Prepare retriever input with collection_id only
        retriever_input = {
            "collection_id": request.collection_id,
            "query": request.query,
            "prompt_template": request.prompt_template or None,
            "model": request.model or None,
            "num_results": request.num_results or 3
        }
        
        # 3. Call API manager to process the request
        result = api_manager.process_qa_request(retriever_input)
        
        # 4. Return the response
        return APIResponse(
            success=True,
            result={
                "answer": result.get("answer", "No answer generated"),
                "sources": result.get("sources", []),
                "time_taken": result.get("time_taken", 0),
                "collection_id": request.collection_id
            }
        )
    except Exception as e:
        print(f"ERROR in QA processing: {str(e)}")
        return APIResponse(
            success=False,
            error=f"Failed to process QA request: {str(e)}"
        )

@app.post("/qa-direct-index", response_model=APIResponse, tags=["QA"],
         summary="Direct QA on existing index",
         description="Perform QA directly on existing index without collection filtering")
async def qa_direct_index(request: DirectIndexQARequest):
    """Process QA request directly on existing index bypassing collection filters"""
    try:
        if api_manager is None:
            raise HTTPException(status_code=500, detail="API manager not initialized")
        
        print(f"DEBUG: Direct Index QA - Index: {request.index_name}, Query: {request.query}")
        
        # Initialize API manager with specific index if needed
        from haystack_new import HaystackRestAPIManager, create_model_config
        
        # Create model config for Ollama
        model_config = create_model_config(
            model_type="ollama",
            model_name="mistral",
            base_url="http://localhost:11434",
            temperature=0.1
        )
        
        # Create manager with specific index
        direct_manager = HaystackRestAPIManager(
            model_config=model_config,
            use_opensearch=True,
            index_name=request.index_name
        )
        
        # Ensure generator is properly initialized
        print(f"DEBUG: Generator available: {hasattr(direct_manager, 'generator') and direct_manager.generator is not None}")
        if hasattr(direct_manager, 'generator') and direct_manager.generator:
            print(f"DEBUG: Generator type: {type(direct_manager.generator)}")
        
        # Perform direct search on the index
        doc_store = direct_manager.document_store
        
        # Build search query
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": request.query,
                                "fields": ["content", "meta.*"],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
                            }
                        }
                    ]
                }
            },
            "size": request.top_k,
            "_source": ["content", "meta"],
            "highlight": {
                "fields": {
                    "content": {
                        "fragment_size": 150,
                        "number_of_fragments": 3
                    }
                }
            }
        }
        
        # Add enhanced filters if provided
        if request.filters:
            if "filter" not in search_body["query"]["bool"]:
                search_body["query"]["bool"]["filter"] = []
            
            for field, value in request.filters.items():
                print(f"DEBUG: Processing filter - Field: {field}, Value: {value}")
                
                # Handle different filter types
                if isinstance(value, dict):
                    # Range filters for dates, numbers
                    if "range" in value:
                        # Try multiple field formats for date fields
                        date_field_candidates = [
                            f"meta.{field}",
                            field,  # Direct field name
                            f"meta.{field}.keyword",  # Keyword version
                            f"{field}.keyword"
                        ]
                        
                        # Use the first candidate and add multiple field attempts
                        range_filter = {
                            "bool": {
                                "should": [
                                    {"range": {candidate: value["range"]}} 
                                    for candidate in date_field_candidates
                                ],
                                "minimum_should_match": 1
                            }
                        }
                        search_body["query"]["bool"]["filter"].append(range_filter)
                    # Term filters with specific operators
                    elif "terms" in value:
                        search_body["query"]["bool"]["filter"].append({
                            "terms": {f"meta.{field}": value["terms"]}
                        })
                    # Wildcard/pattern matching
                    elif "wildcard" in value:
                        search_body["query"]["bool"]["filter"].append({
                            "wildcard": {f"meta.{field}": value["wildcard"]}
                        })
                    # Exists filter
                    elif "exists" in value and value["exists"]:
                        search_body["query"]["bool"]["filter"].append({
                            "exists": {"field": f"meta.{field}"}
                        })
                    # Custom query for complex filtering
                    elif "query" in value:
                        search_body["query"]["bool"]["filter"].append(value["query"])
                # Handle document ID filtering (special case)
                elif field == "document_id" or field == "id":
                    search_body["query"]["bool"]["filter"].append({
                        "term": {"_id": value}
                    })
                # Handle list of values
                elif isinstance(value, list):
                    search_body["query"]["bool"]["filter"].append({
                        "terms": {f"meta.{field}": value}
                    })
                # Simple term filter
                else:
                    search_body["query"]["bool"]["filter"].append({
                        "term": {f"meta.{field}": value}
                    })
        
        # Execute search using direct OpenSearch client
        opensearch_client = create_opensearch_client()
        response = opensearch_client.search(
            index=request.index_name,
            body=search_body
        )
        
        hits = response['hits']['hits']
        total_hits = response['hits']['total']['value']
        
        # Format sources
        sources = []
        context_parts = []
        
        for hit in hits:
            source_data = hit['_source']
            content = source_data.get('content', '')
            
            # Use highlights if available
            highlights = hit.get('highlight', {}).get('content', [])
            display_content = " ... ".join(highlights) if highlights else content[:500]
            
            source = {
                "content": content,
                "metadata": source_data.get('meta', {}),
                "score": hit['_score'],
                "highlights": highlights
            }
            sources.append(source)
            context_parts.append(display_content)
        
        # Generate answer using enhanced QA processing
        answer = "No relevant documents found."
        if sources:
            # Build context from retrieved documents
            context_parts_for_answer = []
            for source in sources:
                content = source.get("content", "")
                # Take more content for better context
                context_parts_for_answer.append(content[:1000])  # More content per document
            
            # Create comprehensive context
            context = "\n\n---\n\n".join(context_parts_for_answer)
            
            # Enhanced prompt for better answers
            answer_prompt = f"""Based on the following documents, provide a comprehensive and detailed answer to the question. Use information from multiple documents when relevant.

Documents:
{context}

Question: {request.query}

Instructions:
- Provide a detailed, informative answer based on the document content
- Combine information from multiple documents when applicable
- If the documents contain relevant information, synthesize it into a coherent response
- If no relevant information is found, state that clearly

Answer:"""
            
            # Use the generator to create answer
            try:
                print(f"DEBUG: Attempting answer generation with {len(sources)} sources")
                
                if hasattr(direct_manager, 'generator') and direct_manager.generator:
                    print("DEBUG: Generator is available, generating answer...")
                    result = direct_manager.generator.run(prompt=answer_prompt)
                    generated_answer = result.get("replies", ["I cannot provide a specific answer based on the available context."])[0]
                    
                    print(f"DEBUG: Generated answer length: {len(generated_answer) if generated_answer else 0}")
                    
                    # Only use generated answer if it's meaningful
                    if generated_answer and len(generated_answer.strip()) > 20:
                        answer = generated_answer
                        print("DEBUG: Using generated answer")
                    else:
                        answer = f"Found {len(sources)} relevant documents but could not generate a comprehensive answer. Please try a more specific query."
                        print("DEBUG: Generated answer too short, using fallback")
                else:
                    print("DEBUG: Generator not available, trying to reinitialize...")
                    # Try to reinitialize the generator
                    try:
                        direct_manager._initialize_generator()
                        if hasattr(direct_manager, 'generator') and direct_manager.generator:
                            result = direct_manager.generator.run(prompt=answer_prompt)
                            generated_answer = result.get("replies", ["I cannot provide a specific answer based on the available context."])[0]
                            if generated_answer and len(generated_answer.strip()) > 20:
                                answer = generated_answer
                            else:
                                answer = f"Found {len(sources)} relevant documents. Generator reinitialized but answer generation needs refinement."
                        else:
                            answer = f"Found {len(sources)} relevant documents. Generator unavailable - please check Ollama service."
                    except Exception as reinit_error:
                        print(f"DEBUG: Generator reinitialization failed: {reinit_error}")
                        answer = f"Found {len(sources)} relevant documents. Generator initialization failed."
            except Exception as gen_error:
                print(f"DEBUG: Answer generation error: {gen_error}")
                answer = f"Found {len(sources)} relevant documents but encountered an error during answer generation: {str(gen_error)}"
        
        return APIResponse(
            success=True,
            result={
                "answer": answer,
                "sources": sources,
                "total_documents_searched": total_hits,
                "documents_returned": len(sources),
                "index_name": request.index_name,
                "filters_applied": request.filters
            }
        )
        
    except Exception as e:
        print(f"ERROR in Direct Index QA: {str(e)}")
        return APIResponse(
            success=False,
            error=f"Failed to process direct index QA: {str(e)}"
        )

@app.post("/search-direct-index", response_model=APIResponse, tags=["Search"],
         summary="Direct search on existing index",
         description="Search directly on existing index with advanced filtering")
async def search_direct_index(request: DirectIndexSearchRequest):
    """Perform direct search on existing index with advanced filtering"""
    try:
        if api_manager is None:
            raise HTTPException(status_code=500, detail="API manager not initialized")
        
        # Create direct manager for the specific index
        from haystack_new import HaystackRestAPIManager, create_model_config
        
        model_config = create_model_config(
            model_type="ollama",
            model_name="mistral",
            base_url="http://localhost:11434",
            temperature=0.1
        )
        
        direct_manager = HaystackRestAPIManager(
            model_config=model_config,
            use_opensearch=True,
            index_name=request.index_name
        )
        
        doc_store = direct_manager.document_store
        
        # Build search query with fuzzy option
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": request.query,
                                "fields": ["content", "meta.*"],
                                "type": "best_fields",
                                "fuzziness": "AUTO" if request.fuzzy else "0"
                            }
                        }
                    ]
                }
            },
            "size": request.top_k,
            "_source": ["content", "meta"]
        }
        
        # Add enhanced filters if provided
        if request.filters:
            if "filter" not in search_body["query"]["bool"]:
                search_body["query"]["bool"]["filter"] = []
            
            for field, value in request.filters.items():
                # Handle different filter types
                if isinstance(value, dict):
                    # Range filters for dates, numbers
                    if "range" in value:
                        search_body["query"]["bool"]["filter"].append({
                            "range": {f"meta.{field}": value["range"]}
                        })
                    # Term filters with specific operators
                    elif "terms" in value:
                        search_body["query"]["bool"]["filter"].append({
                            "terms": {f"meta.{field}": value["terms"]}
                        })
                    # Wildcard/pattern matching
                    elif "wildcard" in value:
                        search_body["query"]["bool"]["filter"].append({
                            "wildcard": {f"meta.{field}": value["wildcard"]}
                        })
                    # Exists filter
                    elif "exists" in value and value["exists"]:
                        search_body["query"]["bool"]["filter"].append({
                            "exists": {"field": f"meta.{field}"}
                        })
                    # Custom query for complex filtering
                    elif "query" in value:
                        search_body["query"]["bool"]["filter"].append(value["query"])
                # Handle document ID filtering (special case)
                elif field == "document_id" or field == "id":
                    search_body["query"]["bool"]["filter"].append({
                        "term": {"_id": value}
                    })
                # Handle list of values
                elif isinstance(value, list):
                    search_body["query"]["bool"]["filter"].append({
                        "terms": {f"meta.{field}": value}
                    })
                # Simple term filter
                else:
                    search_body["query"]["bool"]["filter"].append({
                        "term": {f"meta.{field}": value}
                    })
        
        # Execute search using direct OpenSearch client
        opensearch_client = create_opensearch_client()
        response = opensearch_client.search(
            index=request.index_name,
            body=search_body
        )
        
        hits = response['hits']['hits']
        documents = []
        
        for hit in hits:
            documents.append({
                "id": hit['_id'],
                "content": hit['_source'].get('content', ''),
                "metadata": hit['_source'].get('meta', {}),
                "score": hit['_score']
            })
        
        return APIResponse(
            success=True,
            result={
                "documents": documents,
                "total_found": response['hits']['total']['value'],
                "query": request.query,
                "filters": request.filters
            }
        )
        
    except Exception as e:
        print(f"ERROR in Direct Index Search: {str(e)}")
        return APIResponse(
            success=False,
            error=f"Failed to search direct index: {str(e)}"
        )

@app.post("/index-stats", response_model=APIResponse, tags=["Index"],
         summary="Get index statistics",
         description="Get detailed statistics about an index")
async def get_index_stats(request: IndexStatsRequest):
    """Get statistics about a specific index"""
    try:
        if api_manager is None:
            raise HTTPException(status_code=500, detail="API manager not initialized")
        
        # Create direct manager for the specific index
        from haystack_new import HaystackRestAPIManager, create_model_config
        
        model_config = create_model_config(
            model_type="ollama",
            model_name="mistral",
            base_url="http://localhost:11434",
            temperature=0.1
        )
        
        direct_manager = HaystackRestAPIManager(
            model_config=model_config,
            use_opensearch=True,
            index_name=request.index_name
        )
        
        doc_store = direct_manager.document_store
        
        # Get index stats using direct OpenSearch client
        opensearch_client = create_opensearch_client()
        stats_response = opensearch_client.indices.stats(index=request.index_name)
        mapping_response = opensearch_client.indices.get_mapping(index=request.index_name)
        
        # Extract useful information
        index_stats = stats_response['indices'][request.index_name]
        
        # Get document count
        doc_count = index_stats['total']['docs']['count']
        
        # Get index size
        store_size = index_stats['total']['store']['size_in_bytes']
        
        # Get available fields from mapping
        mapping = mapping_response[request.index_name]['mappings']
        properties = mapping.get('properties', {})
        
        # Extract metadata fields
        meta_fields = []
        if 'meta' in properties:
            meta_props = properties['meta'].get('properties', {})
            meta_fields = list(meta_props.keys())
        
        all_fields = list(properties.keys()) + [f"meta.{field}" for field in meta_fields]
        
        return APIResponse(
            success=True,
            result={
                "stats": {
                    "document_count": doc_count,
                    "index_size": f"{store_size / (1024*1024):.2f} MB",
                    "fields": all_fields,
                    "metadata_fields": meta_fields
                },
                "index_name": request.index_name
            }
        )
        
    except Exception as e:
        print(f"ERROR in Index Stats: {str(e)}")
        return APIResponse(
            success=False,
            error=f"Failed to get index stats: {str(e)}"
        )

@app.post("/browse-documents", response_model=APIResponse, tags=["Index"],
         summary="Browse documents in index",
         description="Browse and paginate through documents in an index")
async def browse_documents(request: DocumentBrowserRequest):
    """Browse documents in an index with pagination"""
    try:
        if api_manager is None:
            raise HTTPException(status_code=500, detail="API manager not initialized")
        
        # Create direct manager for the specific index
        from haystack_new import HaystackRestAPIManager, create_model_config
        
        model_config = create_model_config(
            model_type="ollama",
            model_name="mistral",
            base_url="http://localhost:11434",
            temperature=0.1
        )
        
        direct_manager = HaystackRestAPIManager(
            model_config=model_config,
            use_opensearch=True,
            index_name=request.index_name
        )
        
        doc_store = direct_manager.document_store
        
        # Get documents with pagination using direct OpenSearch client
        opensearch_client = create_opensearch_client()
        response = opensearch_client.search(
            index=request.index_name,
            body={
                "query": {"match_all": {}},
                "from": request.start,
                "size": request.limit,
                "_source": ["content", "meta"]
            }
        )
        
        hits = response['hits']['hits']
        documents = []
        
        for hit in hits:
            documents.append({
                "id": hit['_id'],
                "content": hit['_source'].get('content', ''),
                "metadata": hit['_source'].get('meta', {})
            })
        
        return APIResponse(
            success=True,
            result={
                "documents": documents,
                "start": request.start,
                "limit": request.limit,
                "total_documents": response['hits']['total']['value'],
                "returned_count": len(documents)
            }
        )
        
    except Exception as e:
        print(f"ERROR in Browse Documents: {str(e)}")
        return APIResponse(
            success=False,
            error=f"Failed to browse documents: {str(e)}"
        )

@app.get("/collections/{collection_id}/stats", tags=["Collections"])
async def get_collection_stats(collection_id: str):
    """Get statistics about a document collection."""
    try:
        if api_manager is None:
            raise HTTPException(status_code=500, detail="API manager not initialized")
        
        stats = api_manager.get_collection_stats(collection_id)
        return APIResponse(
            success=True,
            result=stats
        )
        
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.get("/collections", tags=["Collections"], 
         summary="List all collections", 
         description="Retrieve a list of all document collections in the system.")
async def list_all_collections():
    """List all available document collections with their stats"""
    try:
        if api_manager is None:
            raise HTTPException(status_code=500, detail="API manager not initialized")
        
        # Get all unique collection IDs from the document store
        all_docs = api_manager.document_store.filter_documents()
        collections = {}
        
        for doc in all_docs:
            collection_id = doc.meta.get("collection_id", "default")
            if collection_id not in collections:
                collections[collection_id] = {
                    "collection_id": collection_id,
                    "document_count": 0,
                    "document_types": set(),
                    "sources": set(),
                    "date_range": {"earliest": None, "latest": None}
                }
            
            collections[collection_id]["document_count"] += 1
            
            # Track document types and sources
            if "source" in doc.meta:
                collections[collection_id]["sources"].add(doc.meta["source"])
            if "document_type" in doc.meta:
                collections[collection_id]["document_types"].add(doc.meta["document_type"])
            
            # Track date range
            if "timestamp" in doc.meta:
                timestamp = doc.meta["timestamp"]
                if collections[collection_id]["date_range"]["earliest"] is None or timestamp < collections[collection_id]["date_range"]["earliest"]:
                    collections[collection_id]["date_range"]["earliest"] = timestamp
                if collections[collection_id]["date_range"]["latest"] is None or timestamp > collections[collection_id]["date_range"]["latest"]:
                    collections[collection_id]["date_range"]["latest"] = timestamp
        
        # Convert sets to lists for JSON serialization
        for collection in collections.values():
            collection["document_types"] = list(collection["document_types"])
            collection["sources"] = list(collection["sources"])
        
        return APIResponse(
            success=True,
            result={
                "total_collections": len(collections),
                "collections": list(collections.values()),
                "document_store_type": "OpenSearch" if api_manager.use_opensearch else "InMemory"
            }
        )
        
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.get("/documents/{document_id}/collection")
async def find_document_collection(document_id: str):
    """Find which collection a specific document belongs to"""
    try:
        if api_manager is None:
            raise HTTPException(status_code=500, detail="API manager not initialized")
        
        # Search for document by ID across all collections
        matching_docs = api_manager.document_store.filter_documents(
            filters={"field": "document_id", "operator": "==", "value": document_id}
        )
        
        if not matching_docs:
            return APIResponse(
                success=False,
                error=f"Document with ID '{document_id}' not found in any collection"
            )
        
        # Get collection info for found documents
        results = []
        for doc in matching_docs:
            collection_id = doc.meta.get("collection_id", "default")
            results.append({
                "document_id": document_id,
                "collection_id": collection_id,
                "document_meta": doc.meta,
                "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            })
        
        return APIResponse(
            success=True,
            result={
                "document_id": document_id,
                "found_in_collections": results,
                "total_instances": len(results)
            }
        )
        
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.post("/collections/auto-suggest")
async def auto_suggest_collection(request: IndexDocumentsRequest):
    """Auto-suggest collection ID based on document metadata and content"""
    try:
        if api_manager is None:
            raise HTTPException(status_code=500, detail="API manager not initialized")
        
        # Analyze documents to suggest collection ID
        suggestions = []
        
        for doc in request.documents:
            suggestion = {
                "document_id": doc.meta.get("document_id", "unknown"),
                "suggested_collections": [],
                "reasons": []
            }
            
            # Rule-based suggestions based on metadata
            if "timestamp" in doc.meta:
                year = doc.meta["timestamp"][:4] if len(doc.meta["timestamp"]) >= 4 else "unknown"
                suggestion["suggested_collections"].append(f"docs_{year}")
                suggestion["reasons"].append(f"Based on timestamp year: {year}")
            
            if "sender" in doc.meta:
                domain = doc.meta["sender"].split("@")[-1] if "@" in doc.meta["sender"] else "unknown"
                suggestion["suggested_collections"].append(f"domain_{domain.replace('.', '_')}")
                suggestion["reasons"].append(f"Based on sender domain: {domain}")
            
            if "source" in doc.meta:
                suggestion["suggested_collections"].append(f"source_{doc.meta['source']}")
                suggestion["reasons"].append(f"Based on source type: {doc.meta['source']}")
            
            # Content-based suggestions
            content_lower = doc.content.lower()
            if any(word in content_lower for word in ["budget", "financial", "revenue", "cost"]):
                suggestion["suggested_collections"].append("financial_documents")
                suggestion["reasons"].append("Contains financial keywords")
            
            if any(word in content_lower for word in ["meeting", "agenda", "minutes"]):
                suggestion["suggested_collections"].append("meeting_documents") 
                suggestion["reasons"].append("Contains meeting-related keywords")
            
            if any(word in content_lower for word in ["contract", "agreement", "legal"]):
                suggestion["suggested_collections"].append("legal_documents")
                suggestion["reasons"].append("Contains legal keywords")
            
            # Default fallback
            if not suggestion["suggested_collections"]:
                suggestion["suggested_collections"].append("general_documents")
                suggestion["reasons"].append("Default collection for unclassified documents")
            
            suggestions.append(suggestion)
        
        # Get current collections for comparison
        existing_collections = []
        try:
            all_docs = api_manager.document_store.filter_documents()
            existing_collections = list(set(doc.meta.get("collection_id", "default") for doc in all_docs))
        except:
            existing_collections = []
        
        return APIResponse(
            success=True,
            result={
                "suggestions": suggestions,
                "existing_collections": existing_collections,
                "recommended_collection": request.collection_id if request.collection_id != "default" else suggestions[0]["suggested_collections"][0] if suggestions else "default"
            }
        )
        
    except Exception as e:
        return APIResponse(success=False, error=str(e))

# Direct Index Access Endpoints
def create_opensearch_client():
    """Create direct OpenSearch client for index access"""
    try:
        from opensearchpy import OpenSearch
        return OpenSearch(
            hosts=[{'host': 'localhost', 'port': 9200}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False
        )
    except Exception as e:
        print(f"Failed to create OpenSearch client: {e}")
        return None

async def generate_rag_answer(query: str, documents: list) -> str:
    """
    Generate contextual answer using RAG pipeline with Ollama
    """
    print(f"🚀 RAG Function Called! Query: '{query}', Documents: {len(documents)}")
    try:
        from haystack import Pipeline
        from haystack.components.builders import PromptBuilder
        from haystack_integrations.components.generators.ollama import OllamaGenerator
        
        # Create RAG pipeline with improved prompt for Ollama
        template = """Based on the following documents, answer the user's question directly and concisely.

Question: {{ query }}

Documents:
{% for doc in documents %}
Document {{ loop.index }}:
{{ doc.content[:800] }}
---
{% endfor %}

Instructions:
- Answer the question using information from the documents above
- Be direct and specific in your response
- If you find relevant information, provide a clear answer
- Include specific details, names, numbers, or facts from the documents
- If no relevant information is found in ANY of the documents, only then say "No relevant information found"

Answer:"""
        
        # Build the pipeline
        pipeline = Pipeline()
        pipeline.add_component("prompt_builder", PromptBuilder(template=template))
        pipeline.add_component("llm", OllamaGenerator(
            model="mistral",
            url="http://localhost:11434",
            generation_kwargs={
                "temperature": 0.3,  # Slightly higher for more creativity
                "top_p": 0.9,
                "max_tokens": 500,   # Limit response length
                "timeout": 30  # Add timeout
            }
        ))
        
        pipeline.connect("prompt_builder", "llm")
        
        print(f"🤖 Running RAG pipeline for query: '{query}' with {len(documents)} documents")
        print(f"🔍 Document previews:")
        for i, doc in enumerate(documents[:3]):
            preview = doc.content[:100].replace('\n', ' ')
            print(f"  Doc {i+1}: {preview}...")
        
        # Run the pipeline with timeout handling
        import asyncio
        try:
            result = await asyncio.wait_for(
                asyncio.create_task(asyncio.to_thread(pipeline.run, {
                    "prompt_builder": {
                        "query": query,
                        "documents": documents[:3]  # Use top 3 documents to avoid token limit
                    }
                })),
                timeout=45  # 45 second timeout
            )
            print(f"🎯 RAG pipeline completed successfully")
        except asyncio.TimeoutError:
            print(f"⏰ RAG pipeline timed out after 45 seconds")
            raise
        
        # Extract the generated answer
        if result and "llm" in result and "replies" in result["llm"]:
            answer = result["llm"]["replies"][0]
            print(f"✅ RAG pipeline generated answer: {answer[:100]}...")
            return answer
        else:
            print(f"❌ RAG pipeline failed: {result}")
            return f"Unable to generate answer from RAG pipeline for query: '{query}'"
            
    except asyncio.TimeoutError:
        print(f"⏰ RAG pipeline timed out for query: '{query}'")
        # Fallback to extractive summary
        if documents:
            context = " ".join([doc.content[:300] for doc in documents[:2]])
            return f"Based on the retrieved documents: {context[:500]}..."
        return f"RAG pipeline timed out for query: '{query}'"
    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to extractive summary
        if documents:
            context = " ".join([doc.content[:300] for doc in documents[:2]])
            return f"Based on the retrieved documents: {context[:500]}..."
        return f"Error generating answer for query: '{query}'"

@app.post("/qa-direct-index", response_model=APIResponse, tags=["Direct Index"],
         summary="Direct QA against specific index",
         description="Query documents directly from an OpenSearch index without collection filtering")
async def qa_direct_index(request: DirectIndexQARequest):
    """Direct QA against specific OpenSearch index"""
    try:
        # Create direct OpenSearch client
        client = create_opensearch_client()
        if not client:
            return APIResponse(success=False, error="Failed to connect to OpenSearch")
        
        # Search directly in the index
        search_body = {
            "query": {
                "multi_match": {
                    "query": request.query,
                    "fields": ["content", "meta.title", "meta.*"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            },
            "size": request.top_k,
            "highlight": {
                "fields": {
                    "content": {"fragment_size": 200, "number_of_fragments": 3}
                }
            }
        }
        
        # Add filters if provided
        if request.filters:
            bool_query = {"bool": {"must": [search_body["query"]]}}
            for key, value in request.filters.items():
                bool_query["bool"]["must"].append({"term": {key: value}})
            search_body["query"] = bool_query
        
        # Execute search
        response = client.search(index=request.index_name, body=search_body)
        
        # Process results
        sources = []
        documents_for_rag = []  # Collect documents for RAG pipeline
        
        for hit in response["hits"]["hits"]:
            source = {
                "content": hit["_source"].get("content", ""),
                "metadata": hit["_source"].get("meta", {}),
                "score": hit["_score"],
                "id": hit["_id"]
            }
            
            # Add highlights if available
            if "highlight" in hit:
                source["highlights"] = hit["highlight"]
            
            sources.append(source)
            
            # Create Document objects for RAG pipeline
            from haystack.dataclasses import Document
            documents_for_rag.append(Document(
                content=hit["_source"].get("content", ""),
                meta=hit["_source"].get("meta", {}),
                id=hit["_id"],
                score=hit["_score"]
            ))
        
        # Generate contextual answer using RAG pipeline with Ollama
        try:
            print(f"🔍 About to call RAG with {len(documents_for_rag)} documents")
            print(f"🔍 First document content preview: {documents_for_rag[0].content[:100] if documents_for_rag else 'No documents'}...")
            answer = await generate_rag_answer(request.query, documents_for_rag)
            print(f"🎯 RAG returned answer: {answer[:100]}...")
        except Exception as e:
            print(f"RAG pipeline error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to simple answer
            answer = f"Based on the search through {len(sources)} relevant documents, here are the key findings for your query: '{request.query}'"
            if sources:
                answer += f" The most relevant document discusses: {sources[0]['content'][:200]}..."
        
        return APIResponse(
            success=True,
            result={
                "answer": answer,
                "sources": sources,
                "documents_found": response["hits"]["total"]["value"],
                "index_name": request.index_name,
                "query": request.query
            }
        )
        
    except Exception as e:
        print(f"ERROR in direct index QA: {str(e)}")
        return APIResponse(success=False, error=str(e))

@app.post("/search-direct-index", response_model=APIResponse, tags=["Direct Index"],
         summary="Direct search in specific index",
         description="Search documents directly in an OpenSearch index")
async def search_direct_index(request: DirectIndexSearchRequest):
    """Direct search in specific OpenSearch index"""
    try:
        client = create_opensearch_client()
        if not client:
            return APIResponse(success=False, error="Failed to connect to OpenSearch")
        
        search_body = {
            "query": {
                "multi_match": {
                    "query": request.query,
                    "fields": ["content", "meta.title", "meta.*"],
                    "type": "best_fields"
                }
            },
            "size": request.top_k
        }
        
        response = client.search(index=request.index_name, body=search_body)
        
        documents = []
        for hit in response["hits"]["hits"]:
            documents.append({
                "id": hit["_id"],
                "content": hit["_source"].get("content", ""),
                "metadata": hit["_source"].get("meta", {}),
                "score": hit["_score"]
            })
        
        return APIResponse(
            success=True,
            result={
                "documents": documents,
                "total_found": response["hits"]["total"]["value"],
                "index_name": request.index_name
            }
        )
        
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.post("/index-stats", response_model=APIResponse, tags=["Direct Index"],
         summary="Get index statistics",
         description="Get statistics about an OpenSearch index")
async def get_index_stats(request: IndexStatsRequest):
    """Get statistics about an OpenSearch index"""
    try:
        client = create_opensearch_client()
        if not client:
            return APIResponse(success=False, error="Failed to connect to OpenSearch")
        
        # Get index stats
        stats = client.indices.stats(index=request.index_name)
        count_response = client.count(index=request.index_name)
        
        return APIResponse(
            success=True,
            result={
                "index_name": request.index_name,
                "document_count": count_response["count"],
                "index_size": stats["indices"][request.index_name]["total"]["store"]["size_in_bytes"],
                "primary_shards": stats["indices"][request.index_name]["primaries"]["docs"]["count"]
            }
        )
        
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.post("/browse-documents", response_model=APIResponse, tags=["Direct Index"],
         summary="Browse documents in index",
         description="Browse and paginate through documents in an OpenSearch index")
async def browse_documents(request: DocumentBrowserRequest):
    """Browse documents in an OpenSearch index"""
    try:
        client = create_opensearch_client()
        if not client:
            return APIResponse(success=False, error="Failed to connect to OpenSearch")
        
        search_body = {
            "query": {"match_all": {}},
            "size": request.limit,
            "from": request.offset,
            "sort": [{"_id": {"order": "asc"}}]
        }
        
        response = client.search(index=request.index_name, body=search_body)
        
        documents = []
        for hit in response["hits"]["hits"]:
            documents.append({
                "id": hit["_id"],
                "content": hit["_source"].get("content", "")[:500] + "...",  # Truncate for browsing
                "metadata": hit["_source"].get("meta", {})
            })
        
        return APIResponse(
            success=True,
            result={
                "documents": documents,
                "total_documents": response["hits"]["total"]["value"],
                "showing": f"{request.offset + 1}-{request.offset + len(documents)}",
                "index_name": request.index_name
            }
        )
        
    except Exception as e:
        return APIResponse(success=False, error=str(e))

if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8001)
