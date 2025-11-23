"""
Debug endpoints for monitoring typing state.
"""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from eve.api.typing_coordinator import get_typing_coordinator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/debug/typing", tags=["debug"])


@router.get("/status")
async def get_typing_status():
    """Get current typing system status"""
    try:
        coordinator = get_typing_coordinator()
        return JSONResponse(
            status_code=200,
            content={
                "coordinator": coordinator.get_status(),
                "message": "Typing system status retrieved",
            },
        )
    except Exception as e:
        logger.error(f"Error getting typing status: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/cleanup")
async def cleanup_stale_requests(timeout: int = 300):
    """Clean up stale typing requests"""
    try:
        coordinator = get_typing_coordinator()
        cleaned = await coordinator.cleanup_stale_requests(timeout=timeout)
        return JSONResponse(
            status_code=200,
            content={
                "cleaned": cleaned,
                "message": f"Cleaned {cleaned} stale requests",
            },
        )
    except Exception as e:
        logger.error(f"Error cleaning up typing requests: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
