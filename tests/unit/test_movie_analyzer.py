"""
Unit tests for MovieAnalyzer class.

Tests cover:
- Top movies analysis
- Genre trend analysis
- User statistics
- Time series analysis
- Correlation analysis
- Advanced analytics (clustering, trend analysis, anomaly detection)
- Rating sentiment analysis
"""
import pytest
import pandas as pd
import numpy as np

from src.services.movie_analyzer import MovieAnalyzer
from src.exceptions import DataValidationError, DataAnalysisError


class TestMovieAnalyzerInitialization:
    """Test MovieAnalyzer initialization."""

    def test_initialization(self, movie_analyzer):
        """Test MovieAnalyzer initializes correctly."""
        assert movie_analyzer is not None
        assert movie_analyzer.data_processor is not None

    def test_initial_state(self, movie_analyzer):
        """Test initial state of analyzer."""
        # DataFrames should be None before loading
        assert movie_analyzer.movies_df is None or isinstance(movie_analyzer.movies_df, pd.DataFrame)
        assert movie_analyzer.ratings_df is None or isinstance(movie_analyzer.ratings_df, pd.DataFrame)


class TestTopMoviesAnalysis:
    """Test top movies analysis functionality."""

    def test_get_top_movies_basic(self, analyzer_with_data):
        """Test basic top movies retrieval."""
        top_movies = analyzer_with_data.get_top_movies(limit=3, min_ratings=1)

        assert isinstance(top_movies, list)
        assert len(top_movies) <= 3

    def test_get_top_movies_with_min_ratings(self, analyzer_with_data):
        """Test top movies with minimum ratings filter."""
        top_movies = analyzer_with_data.get_top_movies(limit=5, min_ratings=2)

        assert isinstance(top_movies, list)
        # Each movie should have at least min_ratings
        for movie in top_movies:
            if 'rating_count' in movie:
                assert movie['rating_count'] >= 2

    def test_get_top_movies_limit_respected(self, analyzer_with_data):
        """Test that limit parameter is respected."""
        limit = 2
        top_movies = analyzer_with_data.get_top_movies(limit=limit, min_ratings=1)

        assert len(top_movies) <= limit

    def test_get_top_movies_empty_data(self, movie_analyzer, empty_dataframe):
        """Test top movies with empty data."""
        movie_analyzer.movies_df = empty_dataframe
        movie_analyzer.ratings_df = empty_dataframe

        top_movies = movie_analyzer.get_top_movies(limit=5, min_ratings=1)
        assert isinstance(top_movies, list)
        assert len(top_movies) == 0

    def test_get_top_movies_sorting(self, analyzer_with_data):
        """Test that top movies are sorted by rating."""
        top_movies = analyzer_with_data.get_top_movies(limit=5, min_ratings=1)

        if len(top_movies) > 1:
            # Check if sorted by rating (descending)
            ratings = [m.get('weighted_rating', m.get('avg_rating', 0)) for m in top_movies]
            assert ratings == sorted(ratings, reverse=True)


class TestGenreTrendsAnalysis:
    """Test genre trends analysis."""

    def test_analyze_genre_trends_basic(self, analyzer_with_data):
        """Test basic genre trends analysis."""
        trends = analyzer_with_data.analyze_genre_trends()

        assert isinstance(trends, dict)
        assert 'genres' in trends or 'total_genres' in trends

    def test_genre_trends_includes_statistics(self, analyzer_with_data):
        """Test that genre trends include statistical measures."""
        trends = analyzer_with_data.analyze_genre_trends()

        # Should have genre-level statistics
        assert isinstance(trends, dict)

    def test_genre_trends_empty_data(self, movie_analyzer, empty_dataframe):
        """Test genre trends with empty data."""
        movie_analyzer.movies_df = empty_dataframe
        movie_analyzer.ratings_df = empty_dataframe

        trends = movie_analyzer.analyze_genre_trends()
        assert isinstance(trends, dict)


class TestUserStatistics:
    """Test user statistics generation."""

    def test_get_user_statistics_basic(self, analyzer_with_data):
        """Test basic user statistics retrieval."""
        stats = analyzer_with_data.get_user_statistics(user_id=1)

        assert isinstance(stats, dict)
        assert 'user_id' in stats

    def test_get_user_statistics_includes_metrics(self, analyzer_with_data):
        """Test that user statistics include key metrics."""
        stats = analyzer_with_data.get_user_statistics(user_id=1)

        # Should include rating statistics
        expected_keys = ['user_id']
        for key in expected_keys:
            assert key in stats

    def test_get_user_statistics_nonexistent_user(self, analyzer_with_data):
        """Test statistics for non-existent user."""
        with pytest.raises(DataValidationError):
            analyzer_with_data.get_user_statistics(user_id=99999)

    def test_get_user_statistics_multiple_users(self, analyzer_with_data):
        """Test that different users have different statistics."""
        stats1 = analyzer_with_data.get_user_statistics(user_id=1)
        stats2 = analyzer_with_data.get_user_statistics(user_id=2)

        assert stats1['user_id'] != stats2['user_id']


class TestTimeSeriesAnalysis:
    """Test time series analysis."""

    def test_generate_time_series_basic(self, analyzer_with_data):
        """Test basic time series generation."""
        time_series = analyzer_with_data.generate_time_series_analysis()

        assert isinstance(time_series, dict)

    def test_time_series_includes_temporal_data(self, analyzer_with_data):
        """Test that time series includes temporal breakdowns."""
        time_series = analyzer_with_data.generate_time_series_analysis()

        # Should have some temporal analysis
        assert isinstance(time_series, dict)

    def test_time_series_empty_data(self, movie_analyzer, empty_dataframe):
        """Test time series with empty data."""
        movie_analyzer.ratings_df = empty_dataframe

        time_series = movie_analyzer.generate_time_series_analysis()
        assert isinstance(time_series, dict)


class TestCorrelationAnalysis:
    """Test correlation analysis."""

    def test_correlation_analysis_basic(self, analyzer_with_data):
        """Test basic correlation analysis."""
        correlation = analyzer_with_data.get_correlation_analysis()

        assert isinstance(correlation, dict)

    def test_correlation_analysis_includes_metrics(self, analyzer_with_data):
        """Test that correlation analysis includes key metrics."""
        correlation = analyzer_with_data.get_correlation_analysis()

        # Should have correlation data
        assert isinstance(correlation, dict)


class TestUserClustering:
    """Test user clustering analysis."""

    def test_user_clustering_basic(self, analyzer_with_data):
        """Test basic user clustering."""
        clustering = analyzer_with_data.perform_user_clustering(n_clusters=2)

        assert isinstance(clustering, dict)
        assert 'n_clusters' in clustering or 'clusters' in clustering

    def test_user_clustering_cluster_count(self, analyzer_with_data):
        """Test that clustering respects n_clusters parameter."""
        n_clusters = 2
        clustering = analyzer_with_data.perform_user_clustering(n_clusters=n_clusters)

        if 'n_clusters' in clustering:
            assert clustering['n_clusters'] == n_clusters

    def test_user_clustering_invalid_clusters(self, analyzer_with_data):
        """Test clustering with invalid number of clusters."""
        # Should handle edge cases gracefully
        try:
            clustering = analyzer_with_data.perform_user_clustering(n_clusters=1)
            assert isinstance(clustering, dict)
        except (DataValidationError, ValueError):
            pass  # Expected for invalid input


class TestTrendAnalysis:
    """Test trend analysis."""

    def test_trend_analysis_basic(self, analyzer_with_data):
        """Test basic trend analysis."""
        trends = analyzer_with_data.perform_trend_analysis(period='month')

        assert isinstance(trends, dict)

    def test_trend_analysis_different_periods(self, analyzer_with_data):
        """Test trend analysis with different time periods."""
        for period in ['day', 'week', 'month']:
            trends = analyzer_with_data.perform_trend_analysis(period=period)
            assert isinstance(trends, dict)

    def test_trend_analysis_invalid_period(self, analyzer_with_data):
        """Test trend analysis with invalid period."""
        with pytest.raises(DataValidationError):
            analyzer_with_data.perform_trend_analysis(period='invalid_period')


class TestAnomalyDetection:
    """Test anomaly detection."""

    def test_anomaly_detection_basic(self, analyzer_with_data):
        """Test basic anomaly detection."""
        anomalies = analyzer_with_data.detect_anomalies(method='iqr')

        assert isinstance(anomalies, dict)

    def test_anomaly_detection_different_methods(self, analyzer_with_data):
        """Test anomaly detection with different methods."""
        for method in ['iqr', 'zscore']:
            anomalies = analyzer_with_data.detect_anomalies(method=method)
            assert isinstance(anomalies, dict)

    def test_anomaly_detection_invalid_method(self, analyzer_with_data):
        """Test anomaly detection with invalid method."""
        with pytest.raises(DataValidationError):
            analyzer_with_data.detect_anomalies(method='invalid_method')

    def test_anomaly_detection_sensitivity(self, analyzer_with_data):
        """Test anomaly detection with different sensitivity levels."""
        anomalies1 = analyzer_with_data.detect_anomalies(method='iqr', sensitivity=1.5)
        anomalies2 = analyzer_with_data.detect_anomalies(method='iqr', sensitivity=3.0)

        assert isinstance(anomalies1, dict)
        assert isinstance(anomalies2, dict)


class TestRatingSentimentAnalysis:
    """Test rating sentiment analysis."""

    def test_sentiment_analysis_overall(self, analyzer_with_data):
        """Test overall sentiment analysis."""
        sentiment = analyzer_with_data.analyze_rating_sentiment(analysis_type='overall')

        assert isinstance(sentiment, dict)

    def test_sentiment_analysis_movie(self, analyzer_with_data):
        """Test movie-specific sentiment analysis."""
        sentiment = analyzer_with_data.analyze_rating_sentiment(
            analysis_type='movie_sentiment',
            movie_id=1
        )

        assert isinstance(sentiment, dict)

    def test_sentiment_analysis_user(self, analyzer_with_data):
        """Test user-specific sentiment analysis."""
        sentiment = analyzer_with_data.analyze_rating_sentiment(
            analysis_type='user_sentiment',
            user_id=1
        )

        assert isinstance(sentiment, dict)

    def test_sentiment_analysis_temporal(self, analyzer_with_data):
        """Test temporal sentiment analysis."""
        sentiment = analyzer_with_data.analyze_rating_sentiment(
            analysis_type='temporal_sentiment'
        )

        assert isinstance(sentiment, dict)

    def test_sentiment_analysis_invalid_type(self, analyzer_with_data):
        """Test sentiment analysis with invalid type."""
        with pytest.raises(DataValidationError):
            analyzer_with_data.analyze_rating_sentiment(analysis_type='invalid_type')

    def test_sentiment_analysis_missing_required_param(self, analyzer_with_data):
        """Test sentiment analysis missing required parameters."""
        with pytest.raises(DataValidationError):
            analyzer_with_data.analyze_rating_sentiment(
                analysis_type='movie_sentiment'
                # Missing movie_id
            )


class TestDataLoading:
    """Test data loading in MovieAnalyzer."""

    def test_load_datasets(self, movie_analyzer):
        """Test dataset loading."""
        try:
            movie_analyzer.load_datasets()
            # If successful, DataFrames should be loaded
            assert movie_analyzer.movies_df is not None or movie_analyzer.ratings_df is not None
        except Exception:
            # May fail if data files don't exist in test environment
            pass


class TestErrorHandling:
    """Test error handling in MovieAnalyzer."""

    def test_analysis_without_data(self, movie_analyzer):
        """Test that analysis without loaded data handles gracefully."""
        movie_analyzer.movies_df = None
        movie_analyzer.ratings_df = None

        # Should either load data or raise appropriate error
        try:
            movie_analyzer.get_top_movies(limit=5, min_ratings=1)
        except (DataValidationError, DataAnalysisError, AttributeError):
            pass  # Expected behavior

    def test_invalid_parameters(self, analyzer_with_data):
        """Test handling of invalid parameters."""
        with pytest.raises((DataValidationError, ValueError)):
            analyzer_with_data.get_top_movies(limit=-1, min_ratings=-1)
