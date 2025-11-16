from fastapi import APIRouter, HTTPException, status
from typing import Optional
import logging

from src.models.recommendations import (
    SimilarMovieRequest,
    UserRecommendationRequest,
    SimilarMoviesResponse,
    UserRecommendationsResponse,
    SimilarMovie,
    RecommendedMovie
)
from src.services.recommender import SimpleRecommender
from src.services.data_processor import DataProcessor
from src.exceptions.data_exceptions import DataAnalysisError, DataLoadError

# Initialize logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/recommendations",
    tags=["ML Recommendations"],
    responses={404: {"description": "Not found"}},
)

# Initialize recommender (singleton pattern)
_recommender: Optional[SimpleRecommender] = None


def get_recommender() -> SimpleRecommender:
    """
    Get or create the recommender instance.
    Implements singleton pattern for efficiency.
    """
    global _recommender
    if _recommender is None:
        logger.info("Initializing SimpleRecommender instance")
        _recommender = SimpleRecommender(DataProcessor())
    return _recommender


@router.post(
    "/similar_movies",
    response_model=SimilarMoviesResponse,
    status_code=status.HTTP_200_OK,
    summary="Find Similar Movies",
    description="""
    Find movies similar to a given movie using content-based filtering.

    **Algorithm**: Uses cosine similarity on movie features including:
    - Genre vectors (multi-hot encoded)
    - Average rating (normalized)
    - Popularity (log-normalized rating count)

    **Parameters**:
    - `movie_id`: ID of the movie to find similar movies for
    - `limit`: Maximum number of results (default: 10, max: 100)
    - `min_common_ratings`: Minimum ratings required for a movie to be considered (default: 50)

    **Returns**: List of similar movies sorted by similarity score (highest first)
    """
)
async def get_similar_movies(request: SimilarMovieRequest) -> SimilarMoviesResponse:
    """
    Get similar movies based on content-based filtering.

    Args:
        request: Request containing movie_id and filter parameters

    Returns:
        SimilarMoviesResponse with list of similar movies

    Raises:
        HTTPException: If movie not found or analysis fails
    """
    try:
        logger.info(
            f"Finding similar movies for MovieID={request.movie_id}, "
            f"limit={request.limit}, min_ratings={request.min_common_ratings}"
        )

        recommender = get_recommender()
        similar_movies = recommender.get_similar_movies(
            movie_id=request.movie_id,
            limit=request.limit,
            min_common_ratings=request.min_common_ratings
        )

        # Convert to response model
        similar_movies_list = [SimilarMovie(**movie) for movie in similar_movies]

        response = SimilarMoviesResponse(
            status="success",
            movie_id=request.movie_id,
            similar_movies=similar_movies_list,
            count=len(similar_movies_list)
        )

        logger.info(f"Successfully found {len(similar_movies_list)} similar movies")
        return response

    except DataAnalysisError as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except DataLoadError as e:
        logger.error(f"Data load error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in get_similar_movies: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@router.post(
    "/user_recommendations",
    response_model=UserRecommendationsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get User Recommendations",
    description="""
    Generate personalized movie recommendations for a user using collaborative filtering.

    **Algorithm**: User-based collaborative filtering
    1. Find users with similar rating patterns (cosine similarity)
    2. Identify movies rated highly by similar users
    3. Predict ratings using weighted average
    4. Recommend top-rated unwatched movies

    **Fallback**: If no similar users found, returns popular highly-rated movies

    **Parameters**:
    - `user_id`: ID of the user to generate recommendations for
    - `limit`: Maximum number of results (default: 10, max: 100)
    - `min_user_overlap`: Minimum common ratings to consider users similar (default: 50)

    **Returns**: List of recommended movies sorted by predicted rating (highest first)
    """
)
async def get_user_recommendations(
    request: UserRecommendationRequest
) -> UserRecommendationsResponse:
    """
    Get personalized recommendations for a user using collaborative filtering.

    Args:
        request: Request containing user_id and filter parameters

    Returns:
        UserRecommendationsResponse with list of recommended movies

    Raises:
        HTTPException: If user not found or analysis fails
    """
    try:
        logger.info(
            f"Generating recommendations for UserID={request.user_id}, "
            f"limit={request.limit}, min_overlap={request.min_user_overlap}"
        )

        recommender = get_recommender()
        recommendations = recommender.get_user_recommendations(
            user_id=request.user_id,
            limit=request.limit,
            min_user_overlap=request.min_user_overlap
        )

        # Convert to response model
        recommendations_list = [RecommendedMovie(**movie) for movie in recommendations]

        response = UserRecommendationsResponse(
            status="success",
            user_id=request.user_id,
            recommendations=recommendations_list,
            count=len(recommendations_list)
        )

        logger.info(f"Successfully generated {len(recommendations_list)} recommendations")
        return response

    except DataAnalysisError as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except DataLoadError as e:
        logger.error(f"Data load error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in get_user_recommendations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
