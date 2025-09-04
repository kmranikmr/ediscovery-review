"""
Pydantic schemas for API requests and responses
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime

# Base response schema
class BaseResponse(BaseModel):
    """Base response schema"""
    success: bool
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ErrorResponse(BaseResponse):
    """Error response schema"""
    success: bool = False
    error: str
    detail: Optional[str] = None

# QA Schemas
class QARequest(BaseModel):
    """QA request schema"""
    query: str = Field(..., description="Question to ask")
    collection_id: Optional[str] = Field(None, description="Collection ID for filtering")
    index_name: Optional[str] = Field(None, description="Index name to search")
    top_k: int = Field(5, ge=1, le=20, description="Number of documents to retrieve")
    direct_access: bool = Field(False, description="Direct index access mode")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")

class QAResponse(BaseResponse):
    """QA response schema"""
    result: Dict[str, Any]
    method: str
    processing_time: float

# Summarization Schemas
class SummarizationRequest(BaseModel):
    """Summarization request schema"""
    text: str = Field(..., description="Text to summarize")
    summary_type: str = Field("regular", description="Type of summary")
    max_length: int = Field(150, ge=50, le=500, description="Maximum summary length")
    focus: str = Field("general", description="Focus area for summary")

class SummarizationResponse(BaseResponse):
    """Summarization response schema"""
    result: Dict[str, Any]
    method: str
    processing_time: float

# Classification Schemas
class ClassificationRequest(BaseModel):
    """Classification request schema"""
    text: str = Field(..., description="Text to classify")
    classification_types: List[str] = Field(
        default=["document_type", "priority", "sentiment"],
        description="Types of classification to perform"
    )

class ClassificationResponse(BaseResponse):
    """Classification response schema"""
    result: Dict[str, Any]
    method: str
    processing_time: float

# NER Schemas
class NERRequest(BaseModel):
    """NER request schema"""
    text: str = Field(..., description="Text for entity extraction")
    method: str = Field("bert", description="NER method: 'bert' or 'llm'")
    entity_types: Optional[List[str]] = Field(
        None, 
        description="Entity types to extract"
    )
    include_pii: bool = Field(True, description="Include PII detection")
    min_score: float = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence score")

class EntityInfo(BaseModel):
    """Entity information schema"""
    text: str
    confidence: float
    start: int
    end: int

class NERResponse(BaseResponse):
    """NER response schema"""
    result: Dict[str, Any]
    method: str
    processing_time: float

# Document Schemas
class DocumentMetadata(BaseModel):
    """Document metadata schema"""
    document_id: str
    document_type: Optional[str] = None
    source: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    tags: Optional[List[str]] = Field(default_factory=list)

class DocumentRequest(BaseModel):
    """Document indexing request schema"""
    content: str = Field(..., description="Document content")
    meta: DocumentMetadata

class IndexingRequest(BaseModel):
    """Indexing request schema"""
    documents: List[DocumentRequest]
    collection_id: Optional[str] = Field(None, description="Collection ID")
    index_name: Optional[str] = Field(None, description="Custom index name")

class IndexingResponse(BaseResponse):
    """Indexing response schema"""
    indexed_count: int
    failed_count: int
    index_name: str
    processing_time: float

# Health Schemas
class HealthResponse(BaseResponse):
    """Health check response schema"""
    status: str
    version: str
    services: Dict[str, Any]
    uptime: float
