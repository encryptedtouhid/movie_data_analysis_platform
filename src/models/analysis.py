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


class ClusteringRequest(BaseModel):
    """Request model for user clustering analysis"""
    n_clusters: int = Field(
        default=5,
        description="Number of user segments/clusters",
        gt=2,
        le=10,
        example=5
    )


class TrendAnalysisRequest(BaseModel):
    """Request model for trend analysis"""
    period: str = Field(
        default="month",
        description="Time period for aggregation: day, week, month, or year",
        example="month"
    )


class AnomalyDetectionRequest(BaseModel):
    """Request model for anomaly detection"""
    method: str = Field(
        default="iqr",
        description="Detection method: iqr, zscore, or isolation_forest",
        example="iqr"
    )
    sensitivity: float = Field(
        default=1.5,
        description="Detection sensitivity (lower = more sensitive)",
        gt=0,
        le=5,
        example=1.5
    )


class AdvancedAnalyticsResponse(BaseModel):
    """Generic response model for advanced analytics endpoints"""
    status: str = Field(default="success")
    analysis_type: str = Field(..., description="Type of analysis performed")
    result: Dict[str, Any] = Field(..., description="Analysis results")


class RatingSentimentRequest(BaseModel):
    """Request model for rating-based sentiment analysis"""
    analysis_type: str = Field(
        default="overall",
        description="Type of analysis: overall, movie_sentiment, user_sentiment, or temporal_sentiment",
        example="overall"
    )
    movie_id: Optional[int] = Field(
        default=None,
        description="Movie ID for movie-specific sentiment analysis",
        example=1
    )
    user_id: Optional[int] = Field(
        default=None,
        description="User ID for user-specific sentiment analysis",
        example=1
    )


class RatingSentimentResponse(BaseModel):
    """Response model for rating-based sentiment analysis"""
    status: str = Field(default="success")
    message: str
    sentiment_analysis: Dict[str, Any]
