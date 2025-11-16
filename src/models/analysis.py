from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class TopMoviesRequest(BaseModel):
    limit: int = Field(default=10, ge=1, le=100, description="Number of top movies to return")
    min_ratings: int = Field(default=50, ge=1, description="Minimum number of ratings required")


class TopMoviesInsights(BaseModel):
    """Insights for top movies analysis"""
    summary: str
    key_finding: str
    methodology_note: str
    statistical_confidence: str
    recommendation: str


class TopMoviesResponse(BaseModel):
    status: str
    message: str
    top_movies: List[Dict[str, Any]]
    total_found: int
    insights: Optional[TopMoviesInsights] = None


class GenreTrendsInsights(BaseModel):
    """Insights for genre trends analysis"""
    summary: str
    most_popular_genre: str
    highest_rated_genre: str
    key_trends: List[str]
    recommendation: str


class GenreTrendsResponse(BaseModel):
    status: str
    message: str
    genre_analysis: Dict[str, Any]
    insights: Optional[GenreTrendsInsights] = None


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


class ClusteringInsights(BaseModel):
    """Insights for clustering analysis"""
    quality_assessment: str
    interpretation: str
    largest_segment: str
    most_active_segment: str
    business_value: str
    recommendation: str


class TrendAnalysisInsights(BaseModel):
    """Insights for trend analysis"""
    trend_interpretation: str
    stability_assessment: str
    key_finding: str
    implication: str
    recommendation: str


class AnomalyInsights(BaseModel):
    """Insights for anomaly detection"""
    detection_summary: str
    interpretation: str
    key_patterns: List[str]
    business_impact: str
    recommendation: str


class AdvancedAnalyticsResponse(BaseModel):
    """Generic response model for advanced analytics endpoints"""
    status: str = Field(default="success")
    analysis_type: str = Field(..., description="Type of analysis performed")
    result: Dict[str, Any] = Field(..., description="Analysis results")
    insights: Optional[Dict[str, Any]] = None


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


class SentimentInsights(BaseModel):
    """Insights for sentiment analysis"""
    dominant_sentiment: str
    interpretation: str
    health_score: str
    key_findings: List[str]
    recommendation: str


class RatingSentimentResponse(BaseModel):
    """Response model for rating-based sentiment analysis"""
    status: str = Field(default="success")
    message: str
    sentiment_analysis: Dict[str, Any]
    insights: Optional[SentimentInsights] = None
