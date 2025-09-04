"""
Question Answering (QA) endpoints
"""

from fastapi import APIRouter, Request, HTTPException
from app.schemas import QARequest, QAResponse, IndexingRequest, IndexingResponse
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/qa/simple", response_model=QAResponse)
async def qa_simple(request: QARequest, api_request: Request):
    """Simple QA endpoint"""
    haystack_service = api_request.app.state.haystack_service
    
    try:
        result = await haystack_service.process_qa(
            query=request.query,
            collection_id=request.collection_id,
            index_name=request.index_name,
            top_k=request.top_k,
            filters=request.filters
        )
        
        if result["success"]:
            return QAResponse(
                success=True,
                result=result["result"],
                method=result["method"],
                processing_time=result["processing_time"]
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"QA processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/qa/direct-index", response_model=QAResponse)
async def qa_direct_index(request: QARequest, api_request: Request):
    """Direct index QA endpoint"""
    haystack_service = api_request.app.state.haystack_service
    
    try:
        # Add direct access flag
        result = await haystack_service.process_qa(
            query=request.query,
            collection_id=request.collection_id,
            index_name=request.index_name,
            top_k=request.top_k,
            filters=request.filters
        )
        
        if result["success"]:
            return QAResponse(
                success=True,
                result=result["result"],
                method=result["method"],
                processing_time=result["processing_time"]
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Direct QA processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/index", response_model=IndexingResponse)
async def index_documents(request: IndexingRequest, api_request: Request):
    """Index documents endpoint"""
    haystack_service = api_request.app.state.haystack_service
    
    try:
        # Convert request format
        documents = []
        for doc_req in request.documents:
            documents.append({
                "content": doc_req.content,
                "meta": doc_req.meta.dict()
            })
        
        result = await haystack_service.index_documents(
            documents=documents,
            collection_id=request.collection_id,
            index_name=request.index_name
        )
        
        if result["success"]:
            return IndexingResponse(
                success=True,
                indexed_count=result["indexed_count"],
                failed_count=result["failed_count"],
                index_name=result["index_name"],
                processing_time=result["processing_time"]
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Document indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
