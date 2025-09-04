"""
Classification endpoints
"""

from fastapi import APIRouter, Request, HTTPException
from app.schemas import ClassificationRequest, ClassificationResponse
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/classification/comprehensive", response_model=ClassificationResponse)
async def classify_comprehensive(request: ClassificationRequest, api_request: Request):
    """Comprehensive document classification"""
    haystack_service = api_request.app.state.haystack_service
    
    try:
        result = await haystack_service.process_classification(
            text=request.text
        )
        
        if result["success"]:
            return ClassificationResponse(
                success=True,
                result=result["result"],
                method=result["method"],
                processing_time=result["processing_time"]
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
