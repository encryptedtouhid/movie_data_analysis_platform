from datetime import datetime
from fastapi import APIRouter
from src.utils.logger import get_logger
from src.models import HealthCheckResponse

logger = get_logger("health_api", "api")
router = APIRouter()


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
