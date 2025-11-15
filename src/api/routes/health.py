from datetime import datetime
from typing import Dict
from fastapi import APIRouter
from pydantic import BaseModel
from src.utils.logger import get_logger

logger = get_logger("health_api", "api")
router = APIRouter()


class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    service: str
    version: str


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check if the API service is running and healthy",
    tags=["Health"],
)
async def health_check() -> HealthCheckResponse:
    logger.info("Health check endpoint called")
    response = HealthCheckResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        service="Movie Data Analysis Platform",
        version="1.0.0",
    )
    logger.debug(f"Health check response: {response.dict()}")
    return response
