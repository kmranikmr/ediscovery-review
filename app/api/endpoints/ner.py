"""
Named Entity Recognition (NER) endpoints
"""

from fastapi import APIRouter, Request, HTTPException
from app.schemas import NERRequest, NERResponse
from app.services import NERService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Global NER service instance
ner_service = None

@router.on_event("startup")
async def startup_ner():
    """Initialize NER service on startup"""
    global ner_service
    ner_service = NERService()
    await ner_service.initialize()

@router.post("/ner/extract", response_model=NERResponse)
async def extract_entities(request: NERRequest, api_request: Request):
    """Extract named entities from text"""
    global ner_service
    
    if not ner_service:
        ner_service = NERService()
        await ner_service.initialize()
    
    try:
        result = await ner_service.extract_entities(
            text=request.text,
            method=request.method,
            entity_types=request.entity_types,
            include_pii=request.include_pii,
            min_score=request.min_score
        )
        
        if result["success"]:
            return NERResponse(
                success=True,
                result=result["result"],
                method=result["result"]["method"],
                processing_time=result["result"]["processing_time"]
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"NER processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
