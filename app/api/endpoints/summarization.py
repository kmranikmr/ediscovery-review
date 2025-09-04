"""
Summarization endpoints
"""

from fastapi import APIRouter, Request, HTTPException
from app.schemas import SummarizationRequest, SummarizationResponse
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/summarization/regular", response_model=SummarizationResponse)
async def summarize_regular(request: SummarizationRequest, api_request: Request):
    """Regular text summarization"""
    haystack_service = api_request.app.state.haystack_service
    
    try:
        result = await haystack_service.process_summarization(
            text=request.text,
            summary_type=request.summary_type,
            max_length=request.max_length
        )
        
        if result["success"]:
            return SummarizationResponse(
                success=True,
                result=result["result"],
                method=result["method"],
                processing_time=result["processing_time"]
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/summarization/thread", response_model=SummarizationResponse)
async def summarize_thread(request: SummarizationRequest, api_request: Request):
    """Thread-aware summarization"""
    haystack_service = api_request.app.state.haystack_service
    
    try:
        # For thread summarization, we modify the summary type
        result = await haystack_service.process_summarization(
            text=request.text,
            summary_type="thread_summary",
            max_length=request.max_length
        )
        
        if result["success"]:
            return SummarizationResponse(
                success=True,
                result=result["result"],
                method=result["method"],
                processing_time=result["processing_time"]
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Thread summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
