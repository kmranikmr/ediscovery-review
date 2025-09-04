"""
Health check endpoints
"""

from fastapi import APIRouter, Request
from app.schemas import HealthResponse
import time

router = APIRouter()

# Store start time for uptime calculation
_start_time = time.time()

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Health check endpoint"""
    haystack_service = request.app.state.haystack_service
    
    # Get service health status
    health_status = await haystack_service.get_health_status()
    
    return HealthResponse(
        success=True,
        status=health_status["status"],
        version="1.0.0",
        services={
            "haystack": health_status,
            "api": {"status": "healthy"}
        },
        uptime=time.time() - _start_time
    )

@router.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "success": True,
        "message": "eDiscovery LLM Retrieval System API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/v1/health",
            "qa": "/api/v1/qa/*",
            "summarization": "/api/v1/summarization/*", 
            "classification": "/api/v1/classification/*",
            "ner": "/api/v1/ner/*",
            "docs": "/api/v1/docs"
        }
    }
