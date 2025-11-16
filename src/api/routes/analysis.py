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
    TopMoviesInsights,
    GenreTrendsResponse,
    GenreTrendsInsights,
    UserStatisticsRequest,
    UserStatisticsResponse,
    TimeSeriesResponse,
    CorrelationAnalysisResponse,
    VisualizationRequest,
    VisualizationResponse,
    ClusteringRequest,
    TrendAnalysisRequest,
    AnomalyDetectionRequest,
    AdvancedAnalyticsResponse,
    RatingSentimentRequest,
    RatingSentimentResponse,
    SentimentInsights,
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

        # Generate insights
        insights_dict = movie_analyzer._generate_top_movies_insights(
            top_movies,
            request.min_ratings,
            len(top_movies)
        )
        insights = TopMoviesInsights(**insights_dict)

        return TopMoviesResponse(
            status="success",
            message=f"Retrieved top {len(top_movies)} movies",
            top_movies=top_movies,
            total_found=len(top_movies),
            insights=insights
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

        # Extract insights from genre_analysis
        insights_dict = genre_analysis.pop('insights', None)
        insights = GenreTrendsInsights(**insights_dict) if insights_dict else None

        return GenreTrendsResponse(
            status="success",
            message=f"Analyzed {genre_analysis.get('total_genres', 0)} genres",
            genre_analysis=genre_analysis,
            insights=insights
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


@router.post(
    "/clustering",
    response_model=AdvancedAnalyticsResponse,
    summary="User Clustering (Advanced)",
    description="""
    Perform user segmentation based on rating patterns using machine learning.
    
    Uses K-Means clustering to group users with similar rating behaviors.
    
    **Features analyzed:**
    - Average rating
    - Rating variance
    - Number of ratings
    - Movie diversity
    
    **Returns:**
    - User segments/clusters
    - Cluster characteristics
    - Quality metrics (Silhouette score, Davies-Bouldin index)
    """,
    tags=["Analysis"]
)
async def user_clustering(request: ClusteringRequest):
    """
    Perform ML-based user clustering for segmentation analysis.
    """
    try:
        logger.info(f"User clustering requested with {request.n_clusters} clusters")
        
        analyzer = MovieAnalyzer(DataProcessor())
        result = analyzer.perform_user_clustering(n_clusters=request.n_clusters)
        
        return AdvancedAnalyticsResponse(
            status="success",
            analysis_type="user_clustering",
            result=result
        )
        
    except Exception as e:
        logger.error(f"Error in user clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/trend_analysis",
    response_model=AdvancedAnalyticsResponse,
    summary="Trend Analysis (Advanced)",
    description="""
    Advanced time-series trend analysis of rating patterns over time.
    
    **Analysis includes:**
    - Rolling averages and standard deviations
    - Trend direction detection (increasing/decreasing/stable)
    - Peak and trough identification
    - Activity patterns over time
    - Rating volatility metrics
    
    **Periods available:**
    - day: Daily aggregation
    - week: Weekly aggregation
    - month: Monthly aggregation (recommended)
    - year: Yearly aggregation
    """,
    tags=["Analysis"]
)
async def trend_analysis(request: TrendAnalysisRequest):
    """
    Perform advanced trend analysis on rating data.
    """
    try:
        logger.info(f"Trend analysis requested with period: {request.period}")
        
        analyzer = MovieAnalyzer(DataProcessor())
        result = analyzer.perform_trend_analysis(period=request.period)
        
        return AdvancedAnalyticsResponse(
            status="success",
            analysis_type="trend_analysis",
            result=result
        )
        
    except Exception as e:
        logger.error(f"Error in trend analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/anomaly_detection",
    response_model=AdvancedAnalyticsResponse,
    summary="Anomaly Detection (Advanced)",
    description="""
    Identify unusual rating patterns and anomalies using statistical methods.

    **Detection methods:**
    - **iqr**: Interquartile Range (recommended for most cases)
    - **zscore**: Z-score based detection (good for normally distributed data)
    - **isolation_forest**: ML-based isolation forest (best for complex patterns)

    **Detects:**
    - Anomalous users (unusual activity levels)
    - Anomalous movies (polarizing or controversial)
    - Unusual patterns (extreme critics, generous raters, high-volume users)

    **Sensitivity:**
    - Lower values (0.5-1.0): More sensitive, detects more anomalies
    - Higher values (2.0-3.0): Less sensitive, only obvious outliers
    """,
    tags=["Analysis"]
)
async def anomaly_detection(request: AnomalyDetectionRequest):
    """
    Detect anomalies in rating patterns.
    """
    try:
        logger.info(f"Anomaly detection requested with method: {request.method}")

        analyzer = MovieAnalyzer(DataProcessor())
        result = analyzer.detect_anomalies(
            method=request.method,
            sensitivity=request.sensitivity
        )

        return AdvancedAnalyticsResponse(
            status="success",
            analysis_type="anomaly_detection",
            result=result
        )

    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/rating_sentiment",
    response_model=RatingSentimentResponse,
    summary="Rating Sentiment Analysis (Advanced)",
    description="""
    Analyze sentiment patterns derived from numerical ratings.

    **Sentiment Classification:**
    - **Positive**: Ratings 4.0-5.0 stars
    - **Neutral**: Ratings 3.0-3.5 stars
    - **Negative**: Ratings 0.5-2.5 stars

    **Analysis Types:**
    - **overall**: Platform-wide sentiment analysis (default)
      - Overall sentiment distribution
      - User behavior classification (optimistic/critical/balanced raters)
      - Movie sentiment profiles

    - **movie_sentiment**: Movie-specific analysis (requires movie_id)
      - Sentiment distribution for specific movie
      - Polarization score (controversial vs consensus)
      - Consensus score (agreement level)
      - Interpretation of audience reception

    - **user_sentiment**: User-specific analysis (requires user_id)
      - User's sentiment distribution
      - Classification (optimistic/critical/balanced rater)
      - Rating behavior patterns

    - **temporal_sentiment**: Time-based trends
      - Yearly sentiment evolution
      - Monthly sentiment patterns (last 24 months)
      - Sentiment trend direction

    **Use Cases:**
    - Identify polarizing movies (love it or hate it)
    - Find consensus favorites (everyone agrees)
    - Classify user rating behavior
    - Track sentiment changes over time
    - Detect rating patterns and trends
    """,
    tags=["Analysis"]
)
async def rating_sentiment_analysis(request: RatingSentimentRequest):
    """
    Perform sentiment analysis based on rating patterns.
    """
    try:
        logger.info(f"Rating sentiment analysis requested: {request.analysis_type}")

        analyzer = MovieAnalyzer(DataProcessor())
        result = analyzer.analyze_rating_sentiment(
            analysis_type=request.analysis_type,
            movie_id=request.movie_id,
            user_id=request.user_id
        )

        # Extract insights from result
        insights_dict = result.pop('insights', None)
        insights = SentimentInsights(**insights_dict) if insights_dict else None

        return RatingSentimentResponse(
            status="success",
            message=f"Rating sentiment analysis completed: {request.analysis_type}",
            sentiment_analysis=result,
            insights=insights
        )

    except DataValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in rating sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
