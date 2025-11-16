"""
Movie Analysis Service
Provides statistical analysis and insights from movie rating data
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
from src.services.data_processor import DataProcessor
from src.core.config import settings
from src.exceptions import DataAnalysisError, DataValidationError
from src.utils.logger import get_logger

logger = get_logger("movie_analyzer", "analysis")


class MovieAnalyzer:
    """
    Analyze movie ratings data to generate insights and statistics
    """

    def __init__(self, data_processor: DataProcessor):
        """
        Initialize MovieAnalyzer with a DataProcessor instance

        Args:
            data_processor: DataProcessor instance for loading and processing data
        """
        logger.info("Initializing MovieAnalyzer")
        self.data_processor = data_processor
        self.movies_df: Optional[pd.DataFrame] = None
        self.ratings_df: Optional[pd.DataFrame] = None
        self.combined_df: Optional[pd.DataFrame] = None

    def load_datasets(self) -> None:
        """Load required datasets for analysis"""
        try:
            logger.info("Loading datasets for analysis")

            # Load movies
            movies_path = Path(settings.data_processed_path) / "movies_cleaned.csv"
            if movies_path.exists():
                self.movies_df = self.data_processor.load_data(str(movies_path))
                logger.info(f"Loaded {len(self.movies_df)} movies")
            else:
                logger.warning(f"Movies file not found: {movies_path}")

            # Load ratings
            ratings_path = Path(settings.data_processed_path) / "ratings_cleaned.csv"
            if ratings_path.exists():
                self.ratings_df = self.data_processor.load_data(str(ratings_path))
                logger.info(f"Loaded {len(self.ratings_df)} ratings")
            else:
                logger.warning(f"Ratings file not found: {ratings_path}")

            # Create combined dataset
            if self.movies_df is not None and self.ratings_df is not None:
                self.combined_df = pd.merge(
                    self.ratings_df,
                    self.movies_df,
                    on='movieId',
                    how='left'
                )
                logger.info(f"Created combined dataset with {len(self.combined_df)} records")

        except Exception as e:
            logger.error(f"Failed to load datasets: {str(e)}")
            raise DataAnalysisError("Failed to load datasets for analysis", str(e))

    def get_top_movies(self, limit: int = 10, min_ratings: int = 50) -> List[Dict[str, Any]]:
        """
        Get highest-rated movies with statistical significance

        Args:
            limit: Number of top movies to return
            min_ratings: Minimum number of ratings required for inclusion

        Returns:
            List of top-rated movies with statistics
        """
        try:
            logger.info(f"Getting top {limit} movies (min_ratings={min_ratings})")

            if self.combined_df is None:
                self.load_datasets()

            if self.combined_df is None or self.combined_df.empty:
                raise DataValidationError("No data available for analysis")

            # Calculate movie statistics
            movie_stats = self.combined_df.groupby('movieId').agg({
                'rating': ['mean', 'count', 'std'],
                'title': 'first',
                'genres': 'first'
            }).reset_index()

            # Flatten column names
            movie_stats.columns = ['movieId', 'avg_rating', 'rating_count', 'rating_std', 'title', 'genres']

            # Filter by minimum ratings
            movie_stats = movie_stats[movie_stats['rating_count'] >= min_ratings]

            # Calculate Bayesian average (weighted rating)
            # Formula: (v/(v+m)) * R + (m/(v+m)) * C
            # v = number of votes, m = minimum votes, R = average rating, C = mean rating across all movies
            C = movie_stats['avg_rating'].mean()
            m = min_ratings

            movie_stats['weighted_rating'] = (
                (movie_stats['rating_count'] / (movie_stats['rating_count'] + m)) * movie_stats['avg_rating'] +
                (m / (movie_stats['rating_count'] + m)) * C
            )

            # Sort by weighted rating
            top_movies = movie_stats.nlargest(limit, 'weighted_rating')

            # Convert to list of dictionaries
            result = []
            for _, row in top_movies.iterrows():
                result.append({
                    'movieId': int(row['movieId']),
                    'title': row['title'],
                    'genres': row['genres'],
                    'average_rating': float(row['avg_rating']),
                    'rating_count': int(row['rating_count']),
                    'rating_std': float(row['rating_std']) if pd.notna(row['rating_std']) else 0.0,
                    'weighted_rating': float(row['weighted_rating'])
                })

            logger.info(f"Found {len(result)} top movies")
            return result

        except Exception as e:
            logger.error(f"Failed to get top movies: {str(e)}")
            raise DataAnalysisError("Failed to get top movies", str(e))

    def analyze_genre_trends(self) -> Dict[str, Any]:
        """
        Analyze popularity and rating trends by genre

        Returns:
            Dictionary with genre analysis results
        """
        try:
            logger.info("Analyzing genre trends")

            if self.combined_df is None:
                self.load_datasets()

            if self.combined_df is None or self.combined_df.empty:
                raise DataValidationError("No data available for analysis")

            # Explode genres (split pipe-separated genres into separate rows)
            df_exploded = self.combined_df.copy()
            df_exploded['genre_list'] = df_exploded['genres'].str.split('|')
            df_exploded = df_exploded.explode('genre_list')

            # Calculate genre statistics
            genre_stats = df_exploded.groupby('genre_list').agg({
                'rating': ['mean', 'count', 'std', 'median'],
                'movieId': 'nunique',
                'userId': 'nunique'
            }).reset_index()

            # Flatten column names
            genre_stats.columns = ['genre', 'avg_rating', 'rating_count', 'rating_std', 'median_rating', 'unique_movies', 'unique_users']

            # Sort by rating count
            genre_stats = genre_stats.sort_values('rating_count', ascending=False)

            # Calculate popularity score (normalized)
            max_count = genre_stats['rating_count'].max()
            genre_stats['popularity_score'] = (genre_stats['rating_count'] / max_count) * 100

            # Convert to dictionary format
            genre_trends = {
                'total_genres': int(len(genre_stats)),
                'genres': []
            }

            for _, row in genre_stats.iterrows():
                genre_trends['genres'].append({
                    'genre': row['genre'],
                    'average_rating': float(row['avg_rating']),
                    'median_rating': float(row['median_rating']),
                    'rating_std': float(row['rating_std']) if pd.notna(row['rating_std']) else 0.0,
                    'rating_count': int(row['rating_count']),
                    'unique_movies': int(row['unique_movies']),
                    'unique_users': int(row['unique_users']),
                    'popularity_score': float(row['popularity_score'])
                })

            # Add top 5 and bottom 5 by rating
            genre_trends['top_rated_genres'] = genre_trends['genres'][:5]
            genre_trends['least_rated_genres'] = genre_trends['genres'][-5:]

            logger.info(f"Analyzed {genre_trends['total_genres']} genres")
            return genre_trends

        except Exception as e:
            logger.error(f"Failed to analyze genre trends: {str(e)}")
            raise DataAnalysisError("Failed to analyze genre trends", str(e))

    def get_user_statistics(self, user_id: int) -> Dict[str, Any]:
        """
        Generate comprehensive user behavior statistics

        Args:
            user_id: User ID to analyze

        Returns:
            Dictionary with user statistics
        """
        try:
            logger.info(f"Getting statistics for user {user_id}")

            if self.combined_df is None:
                self.load_datasets()

            if self.combined_df is None or self.combined_df.empty:
                raise DataValidationError("No data available for analysis")

            # Filter user ratings
            user_ratings = self.combined_df[self.combined_df['userId'] == user_id]

            if user_ratings.empty:
                raise DataValidationError(f"No ratings found for user {user_id}")

            # Calculate user statistics
            stats = {
                'user_id': int(user_id),
                'total_ratings': int(len(user_ratings)),
                'average_rating': float(user_ratings['rating'].mean()),
                'median_rating': float(user_ratings['rating'].median()),
                'rating_std': float(user_ratings['rating'].std()) if len(user_ratings) > 1 else 0.0,
                'min_rating': float(user_ratings['rating'].min()),
                'max_rating': float(user_ratings['rating'].max()),
                'unique_movies_rated': int(user_ratings['movieId'].nunique())
            }

            # Rating distribution
            rating_dist = user_ratings['rating'].value_counts().sort_index()
            stats['rating_distribution'] = {
                str(rating): int(count) for rating, count in rating_dist.items()
            }

            # Most common rating
            stats['most_common_rating'] = float(rating_dist.idxmax())

            # Genre preferences
            user_genres = user_ratings.copy()
            user_genres['genre_list'] = user_genres['genres'].str.split('|')
            user_genres_exploded = user_genres.explode('genre_list')

            genre_prefs = user_genres_exploded.groupby('genre_list').agg({
                'rating': ['mean', 'count']
            }).reset_index()
            genre_prefs.columns = ['genre', 'avg_rating', 'count']
            genre_prefs = genre_prefs.sort_values('count', ascending=False)

            stats['favorite_genres'] = [
                {
                    'genre': row['genre'],
                    'average_rating': float(row['avg_rating']),
                    'count': int(row['count'])
                }
                for _, row in genre_prefs.head(10).iterrows()
            ]

            # Top rated movies by this user
            top_user_movies = user_ratings.nlargest(10, 'rating')[['title', 'genres', 'rating']]
            stats['top_rated_movies'] = [
                {
                    'title': row['title'],
                    'genres': row['genres'],
                    'rating': float(row['rating'])
                }
                for _, row in top_user_movies.iterrows()
            ]

            # Time-based analysis if timestamp available
            if 'timestamp' in user_ratings.columns:
                user_ratings['datetime'] = pd.to_datetime(user_ratings['timestamp'], unit='s')
                stats['first_rating_date'] = str(user_ratings['datetime'].min())
                stats['last_rating_date'] = str(user_ratings['datetime'].max())
                stats['rating_span_days'] = int((user_ratings['datetime'].max() - user_ratings['datetime'].min()).days)

            logger.info(f"Generated statistics for user {user_id}")
            return stats

        except Exception as e:
            logger.error(f"Failed to get user statistics: {str(e)}")
            raise DataAnalysisError(f"Failed to get user statistics for user {user_id}", str(e))

    def generate_time_series_analysis(self) -> Dict[str, Any]:
        """
        Analyze rating patterns over time

        Returns:
            Dictionary with time-series analysis results
        """
        try:
            logger.info("Generating time-series analysis")

            if self.ratings_df is None:
                self.load_datasets()

            if self.ratings_df is None or self.ratings_df.empty:
                raise DataValidationError("No ratings data available for analysis")

            if 'timestamp' not in self.ratings_df.columns:
                raise DataValidationError("Timestamp column not found in ratings data")

            # Convert timestamp to datetime
            df = self.ratings_df.copy()
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df['year'] = df['datetime'].dt.year
            df['month'] = df['datetime'].dt.month
            df['year_month'] = df['datetime'].dt.to_period('M')
            df['day_of_week'] = df['datetime'].dt.day_name()

            analysis = {
                'date_range': {
                    'earliest': str(df['datetime'].min()),
                    'latest': str(df['datetime'].max()),
                    'span_days': int((df['datetime'].max() - df['datetime'].min()).days)
                }
            }

            # Yearly trends
            yearly = df.groupby('year').agg({
                'rating': ['mean', 'count', 'std']
            }).reset_index()
            yearly.columns = ['year', 'avg_rating', 'rating_count', 'rating_std']

            analysis['yearly_trends'] = [
                {
                    'year': int(row['year']),
                    'average_rating': float(row['avg_rating']),
                    'rating_count': int(row['rating_count']),
                    'rating_std': float(row['rating_std']) if pd.notna(row['rating_std']) else 0.0
                }
                for _, row in yearly.iterrows()
            ]

            # Monthly trends (last 12 months)
            monthly = df.groupby('year_month').agg({
                'rating': ['mean', 'count']
            }).reset_index()
            monthly.columns = ['year_month', 'avg_rating', 'rating_count']
            monthly = monthly.tail(12)

            analysis['monthly_trends_last_12'] = [
                {
                    'year_month': str(row['year_month']),
                    'average_rating': float(row['avg_rating']),
                    'rating_count': int(row['rating_count'])
                }
                for _, row in monthly.iterrows()
            ]

            # Day of week patterns
            day_of_week = df.groupby('day_of_week').agg({
                'rating': ['mean', 'count']
            }).reset_index()
            day_of_week.columns = ['day', 'avg_rating', 'rating_count']

            # Order by day of week
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_of_week['day'] = pd.Categorical(day_of_week['day'], categories=day_order, ordered=True)
            day_of_week = day_of_week.sort_values('day')

            analysis['day_of_week_patterns'] = [
                {
                    'day': row['day'],
                    'average_rating': float(row['avg_rating']),
                    'rating_count': int(row['rating_count'])
                }
                for _, row in day_of_week.iterrows()
            ]

            # Peak activity times
            analysis['peak_activity'] = {
                'most_active_year': int(yearly.nlargest(1, 'rating_count')['year'].iloc[0]),
                'most_active_day': str(day_of_week.nlargest(1, 'rating_count')['day'].iloc[0]),
                'highest_avg_rating_year': int(yearly.nlargest(1, 'avg_rating')['year'].iloc[0])
            }

            logger.info("Time-series analysis completed")
            return analysis

        except Exception as e:
            logger.error(f"Failed to generate time-series analysis: {str(e)}")
            raise DataAnalysisError("Failed to generate time-series analysis", str(e))

    def get_correlation_analysis(self) -> Dict[str, Any]:
        """
        Analyze correlations between different metrics

        Returns:
            Dictionary with correlation analysis results
        """
        try:
            logger.info("Generating correlation analysis")

            if self.combined_df is None:
                self.load_datasets()

            if self.combined_df is None or self.combined_df.empty:
                raise DataValidationError("No data available for analysis")

            # Calculate movie-level metrics
            movie_metrics = self.combined_df.groupby('movieId').agg({
                'rating': ['mean', 'count', 'std'],
                'userId': 'nunique'
            }).reset_index()
            movie_metrics.columns = ['movieId', 'avg_rating', 'rating_count', 'rating_std', 'unique_users']

            # Calculate correlations
            correlations = {}

            # Rating count vs Average rating
            correlations['rating_count_vs_avg_rating'] = float(
                movie_metrics['rating_count'].corr(movie_metrics['avg_rating'])
            )

            # Unique users vs Average rating
            correlations['unique_users_vs_avg_rating'] = float(
                movie_metrics['unique_users'].corr(movie_metrics['avg_rating'])
            )

            # Rating std vs Average rating
            correlations['rating_std_vs_avg_rating'] = float(
                movie_metrics['rating_std'].corr(movie_metrics['avg_rating'])
            )

            analysis = {
                'correlations': correlations,
                'interpretation': {
                    'rating_count_vs_avg_rating': self._interpret_correlation(
                        correlations['rating_count_vs_avg_rating'],
                        'number of ratings',
                        'average rating'
                    ),
                    'unique_users_vs_avg_rating': self._interpret_correlation(
                        correlations['unique_users_vs_avg_rating'],
                        'number of unique users',
                        'average rating'
                    ),
                    'rating_std_vs_avg_rating': self._interpret_correlation(
                        correlations['rating_std_vs_avg_rating'],
                        'rating standard deviation',
                        'average rating'
                    )
                }
            }

            logger.info("Correlation analysis completed")
            return analysis

        except Exception as e:
            logger.error(f"Failed to generate correlation analysis: {str(e)}")
            raise DataAnalysisError("Failed to generate correlation analysis", str(e))

    def _interpret_correlation(self, corr: float, var1: str, var2: str) -> str:
        """Helper method to interpret correlation values"""
        if abs(corr) < 0.1:
            strength = "negligible"
        elif abs(corr) < 0.3:
            strength = "weak"
        elif abs(corr) < 0.5:
            strength = "moderate"
        elif abs(corr) < 0.7:
            strength = "strong"
        else:
            strength = "very strong"

        direction = "positive" if corr > 0 else "negative"

        return f"There is a {strength} {direction} correlation ({corr:.3f}) between {var1} and {var2}."
