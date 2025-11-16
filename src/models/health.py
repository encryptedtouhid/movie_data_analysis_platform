from pydantic import BaseModel


class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    service: str
    version: str
