from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class SimilarMovieRequest(BaseModel):
    """Request model for finding similar movies."""

    movie_id: int = Field(
        ...,
        description="ID of the movie to find similar movies for",
        gt=0,
        example=1
    )
    limit: int = Field(
        default=10,
        description="Maximum number of similar movies to return",
        gt=0,
        le=100,
        example=10
    )
    min_common_ratings: int = Field(
        default=50,
        description="Minimum number of ratings for a movie to be considered",
        ge=1,
        le=1000,
        example=50
    )

    @field_validator('limit')
    @classmethod
    def validate_limit(cls, v):
        """Ensure limit is reasonable."""
        if v > 100:
            raise ValueError("Limit cannot exceed 100")
        return v


class UserRecommendationRequest(BaseModel):
    """Request model for user-based recommendations."""

    user_id: int = Field(
        ...,
        description="ID of the user to generate recommendations for",
        gt=0,
        example=1
    )
    limit: int = Field(
        default=10,
        description="Maximum number of recommendations to return",
        gt=0,
        le=100,
        example=10
    )
    min_user_overlap: int = Field(
        default=50,
        description="Minimum number of common ratings to consider users similar",
        ge=1,
        le=500,
        example=50
    )

    @field_validator('limit')
    @classmethod
    def validate_limit(cls, v):
        """Ensure limit is reasonable."""
        if v > 100:
            raise ValueError("Limit cannot exceed 100")
        return v


class SimilarMovie(BaseModel):
    """Model for a similar movie result."""

    MovieID: int = Field(..., description="Movie identifier")
    Title: str = Field(..., description="Movie title")
    Genres: str = Field(..., description="Movie genres (pipe-separated)")
    Similarity: float = Field(
        ...,
        description="Similarity score (0-1, higher is more similar)",
        ge=0.0,
        le=1.0
    )
    AvgRating: float = Field(
        ...,
        description="Average rating across all users",
        ge=0.0,
        le=5.0
    )
    RatingCount: int = Field(..., description="Number of ratings", ge=0)


class RecommendedMovie(BaseModel):
    """Model for a recommended movie result."""

    MovieID: int = Field(..., description="Movie identifier")
    Title: str = Field(..., description="Movie title")
    Genres: str = Field(..., description="Movie genres (pipe-separated)")
    PredictedRating: float = Field(
        ...,
        description="Predicted rating for the user",
        ge=0.0,
        le=5.0
    )
    AvgRating: float = Field(
        ...,
        description="Average rating across all users",
        ge=0.0,
        le=5.0
    )
    RatingCount: int = Field(..., description="Number of ratings", ge=0)


class SimilarMoviesResponse(BaseModel):
    """Response model for similar movies endpoint."""

    status: str = Field(default="success", description="Response status")
    movie_id: int = Field(..., description="Input movie ID")
    similar_movies: List[SimilarMovie] = Field(
        ...,
        description="List of similar movies"
    )
    count: int = Field(..., description="Number of similar movies returned")


class UserRecommendationsResponse(BaseModel):
    """Response model for user recommendations endpoint."""

    status: str = Field(default="success", description="Response status")
    user_id: int = Field(..., description="Input user ID")
    recommendations: List[RecommendedMovie] = Field(
        ...,
        description="List of recommended movies"
    )
    count: int = Field(..., description="Number of recommendations returned")


