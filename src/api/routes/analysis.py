"""
Analysis API Routes
Endpoints for movie data analysis and visualization
"""
from fastapi import APIRouter, HTTPException
from pathlib import Path
from src.services.movie_analyzer import MovieAnalyzer
from src.services.data_visualizer import DataVisualizer
from src.services.data_processor import DataProcessor
from src.exceptions import DataAnalysisError, DataValidationError
from src.utils.logger import get_logger
from src.core.config import settings
from src.models import (
    TopMoviesRequest,
    TopMoviesResponse,
    GenreTrendsResponse,
    UserStatisticsRequest,
    UserStatisticsResponse,
    TimeSeriesResponse,
    CorrelationAnalysisResponse,
    VisualizationRequest,
    VisualizationResponse,
)

logger = get_logger("analysis_api", "api")
router = APIRouter()

# Initialize services
data_processor = DataProcessor()
movie_analyzer = MovieAnalyzer(data_processor)
data_visualizer = DataVisualizer()


@router.post(
    "/top_movies",
    response_model=TopMoviesResponse,
    summary="Get Top Rated Movies",
    description="Get highest-rated movies with statistical significance using Bayesian average",
    tags=["Analysis"],
)
async def get_top_movies(request: TopMoviesRequest) -> TopMoviesResponse:
    """
    Get top-rated movies with statistical significance.

    Uses Bayesian average to account for both rating and popularity,
    ensuring movies with few ratings don't dominate the rankings.
    """
    try:
        logger.info(f"Getting top {request.limit} movies (min_ratings={request.min_ratings})")

        top_movies = movie_analyzer.get_top_movies(
            limit=request.limit,
            min_ratings=request.min_ratings
        )

        return TopMoviesResponse(
            status="success",
            message=f"Retrieved top {len(top_movies)} movies",
            top_movies=top_movies,
            total_found=len(top_movies)
        )

    except DataValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except DataAnalysisError as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get(
    "/genre_trends",
    response_model=GenreTrendsResponse,
    summary="Analyze Genre Trends",
    description="Analyze popularity and rating trends across all genres",
    tags=["Analysis"],
)
async def analyze_genre_trends() -> GenreTrendsResponse:
    """
    Analyze genre popularity and rating trends.

    Returns comprehensive statistics for each genre including:
    - Average and median ratings
    - Number of ratings and unique movies
    - Popularity score
    """
    try:
        logger.info("Analyzing genre trends")

        genre_analysis = movie_analyzer.analyze_genre_trends()

        return GenreTrendsResponse(
            status="success",
            message=f"Analyzed {genre_analysis.get('total_genres', 0)} genres",
            genre_analysis=genre_analysis
        )

    except DataValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except DataAnalysisError as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post(
    "/user_statistics",
    response_model=UserStatisticsResponse,
    summary="Get User Statistics",
    description="Generate comprehensive statistics for a specific user",
    tags=["Analysis"],
)
async def get_user_statistics(request: UserStatisticsRequest) -> UserStatisticsResponse:
    """
    Get comprehensive statistics for a specific user.

    Includes:
    - Rating behavior (average, median, distribution)
    - Genre preferences
    - Top-rated movies
    - Activity timeline
    """
    try:
        logger.info(f"Getting statistics for user {request.user_id}")

        user_stats = movie_analyzer.get_user_statistics(request.user_id)

        return UserStatisticsResponse(
            status="success",
            message=f"Generated statistics for user {request.user_id}",
            user_statistics=user_stats
        )

    except DataValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except DataAnalysisError as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get(
    "/time_series",
    response_model=TimeSeriesResponse,
    summary="Time Series Analysis",
    description="Analyze rating patterns over time (yearly, monthly, day of week)",
    tags=["Analysis"],
)
async def get_time_series_analysis() -> TimeSeriesResponse:
    """
    Analyze rating patterns over time.

    Includes:
    - Yearly trends
    - Monthly trends (last 12 months)
    - Day of week patterns
    - Peak activity periods
    """
    try:
        logger.info("Generating time-series analysis")

        time_series = movie_analyzer.generate_time_series_analysis()

        return TimeSeriesResponse(
            status="success",
            message="Time-series analysis completed",
            time_series_analysis=time_series
        )

    except DataValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except DataAnalysisError as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get(
    "/correlation_analysis",
    response_model=CorrelationAnalysisResponse,
    summary="Correlation Analysis",
    description="Analyze correlations between different movie metrics",
    tags=["Analysis"],
)
async def get_correlation_analysis() -> CorrelationAnalysisResponse:
    """
    Analyze correlations between different metrics.

    Examines relationships between:
    - Rating count and average rating
    - Number of users and average rating
    - Rating variance and average rating
    """
    try:
        logger.info("Generating correlation analysis")

        correlation_analysis = movie_analyzer.get_correlation_analysis()

        return CorrelationAnalysisResponse(
            status="success",
            message="Correlation analysis completed",
            correlation_analysis=correlation_analysis
        )

    except DataValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except DataAnalysisError as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post(
    "/visualize",
    response_model=VisualizationResponse,
    summary="Generate Visualization",
    description="Create visualizations (charts, plots, dashboard reports)",
    tags=["Visualization"],
)
async def create_visualization(request: VisualizationRequest) -> VisualizationResponse:
    """
    Generate visualizations from analysis data.

    Supported types:
    - rating_distribution: Histogram and count plot of ratings
    - genre_popularity: Genre popularity and rating charts
    - time_series: Temporal trends visualization
    - dashboard: Comprehensive HTML dashboard report
    """
    try:
        logger.info(f"Creating {request.visualization_type} visualization")

        file_path = None

        if request.visualization_type == "rating_distribution":
            # Load ratings data
            movie_analyzer.load_datasets()
            if movie_analyzer.ratings_df is None:
                raise DataValidationError("Ratings data not available")

            file_path = data_visualizer.create_rating_distribution(
                movie_analyzer.ratings_df
            )

        elif request.visualization_type == "genre_popularity":
            # Get genre analysis
            genre_data = movie_analyzer.analyze_genre_trends()
            file_path = data_visualizer.plot_genre_popularity(
                genre_data
            )

        elif request.visualization_type == "time_series":
            # Get time series data
            time_data = movie_analyzer.generate_time_series_analysis()
            file_path = data_visualizer.plot_time_series(
                time_data
            )

        elif request.visualization_type == "dashboard":
            # Generate comprehensive dashboard
            analysis_data = {
                'top_movies': movie_analyzer.get_top_movies(limit=20),
                'genre_analysis': movie_analyzer.analyze_genre_trends(),
                'time_series': movie_analyzer.generate_time_series_analysis(),
            }
            file_path = data_visualizer.generate_dashboard_report(
                analysis_data
            )

        else:
            raise DataValidationError(
                f"Invalid visualization type: {request.visualization_type}. "
                "Valid types: rating_distribution, genre_popularity, time_series, dashboard"
            )

        # Extract filename from full path and construct browsable URL
        filename = Path(file_path).name
        base_url = f"http://{settings.host}:{settings.port}"
        browsable_url = f"{base_url}/visualizations/{filename}"

        return VisualizationResponse(
            status="success",
            message=f"{request.visualization_type} visualization created successfully",
            file_path=file_path,
            url=browsable_url,
            visualization_type=request.visualization_type
        )

    except DataValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except DataAnalysisError as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
