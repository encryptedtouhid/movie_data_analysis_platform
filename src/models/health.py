from pydantic import BaseModel, Field
from typing import Dict, Optional


class DataFileStatus(BaseModel):
    """Status of a single data file."""
    exists: bool = Field(..., description="Whether the file exists")
    size_mb: Optional[float] = Field(None, description="File size in megabytes")
    path: str = Field(..., description="File path")


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Overall health status: healthy, degraded, or unhealthy")
    timestamp: str = Field(..., description="Timestamp of health check")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    data_files: Dict[str, DataFileStatus] = Field(..., description="Status of required data files")
