from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class TopMoviesRequest(BaseModel):
    limit: int = Field(default=10, ge=1, le=100, description="Number of top movies to return")
    min_ratings: int = Field(default=50, ge=1, description="Minimum number of ratings required")


class TopMoviesResponse(BaseModel):
    status: str
    message: str
    top_movies: List[Dict[str, Any]]
    total_found: int


class GenreTrendsResponse(BaseModel):
    status: str
    message: str
    genre_analysis: Dict[str, Any]


class UserStatisticsRequest(BaseModel):
    user_id: int = Field(description="User ID to analyze")


class UserStatisticsResponse(BaseModel):
    status: str
    message: str
    user_statistics: Dict[str, Any]


class TimeSeriesResponse(BaseModel):
    status: str
    message: str
    time_series_analysis: Dict[str, Any]


class CorrelationAnalysisResponse(BaseModel):
    status: str
    message: str
    correlation_analysis: Dict[str, Any]


class VisualizationRequest(BaseModel):
    visualization_type: str = Field(description="Type of visualization: rating_distribution, genre_popularity, time_series, or dashboard")


class VisualizationResponse(BaseModel):
    status: str
    message: str
    file_path: str
    url: str
    visualization_type: str
