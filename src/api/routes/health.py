from datetime import datetime
from pathlib import Path
from fastapi import APIRouter
from src.utils.logger import get_logger
from src.models.health import HealthCheckResponse, DataFileStatus
from src.core.config import settings

logger = get_logger("health_api", "api")
router = APIRouter()

# Required data files
REQUIRED_DATA_FILES = ["movies.dat", "ratings.dat", "tags.dat", "users.dat"]


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="""
    Check if the API service is running and healthy.

    Validates:
    - Service availability
    - Required data files existence in raw folder
    - Data file sizes

    Status codes:
    - **healthy**: All systems operational, all data files present
    - **degraded**: Service running but some data files missing
    - **unhealthy**: Critical issues detected
    """,
    tags=["Health"],
)
async def health_check() -> HealthCheckResponse:
    """
    Perform comprehensive health check of the service.

    Returns:
        HealthCheckResponse with service status and data file information
    """
    logger.info("Health check endpoint called")

    # Check data files in raw folder
    raw_path = Path(settings.data_raw_path)
    data_files_status = {}
    all_files_present = True

    for filename in REQUIRED_DATA_FILES:
        file_path = raw_path / filename
        exists = file_path.exists()

        if not exists:
            all_files_present = False
            logger.warning(f"Required data file missing: {filename}")

        size_mb = None
        if exists:
            try:
                size_bytes = file_path.stat().st_size
                size_mb = round(size_bytes / (1024 * 1024), 2)  # Convert to MB
            except Exception as e:
                logger.error(f"Error getting file size for {filename}: {str(e)}")

        data_files_status[filename] = DataFileStatus(
            exists=exists,
            size_mb=size_mb,
            path=str(file_path)
        )

    # Determine overall status
    if all_files_present:
        overall_status = "healthy"
        logger.info("Health check: All systems healthy")
    else:
        overall_status = "degraded"
        missing_files = [name for name, status in data_files_status.items() if not status.exists]
        logger.warning(f"Health check: Degraded - Missing files: {missing_files}")

    response = HealthCheckResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        service="Movie Data Analysis Platform",
        version="1.0.0",
        data_files=data_files_status
    )

    logger.debug(f"Health check response: {response.dict()}")
    return response
