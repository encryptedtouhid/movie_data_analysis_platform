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

            # Add insights
            genre_trends['insights'] = self._generate_genre_insights({'genre_stats': genre_trends['genres']})

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

        except DataValidationError:
            # Re-raise validation errors as-is (will result in 400/404)
            raise
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

    def perform_user_clustering(self, n_clusters: int = 5) -> Dict[str, Any]:
        """
        Perform user segmentation based on rating patterns using K-Means clustering.
        
        Args:
            n_clusters: Number of user segments to create (default: 5)
            
        Returns:
            Dictionary containing cluster information and user segments
            
        Raises:
            DataAnalysisError: If clustering fails
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            logger.info(f"Performing user clustering with {n_clusters} clusters")

            if self.ratings_df is None:
                self.load_datasets()

            if self.ratings_df is None or self.ratings_df.empty:
                raise DataValidationError("No ratings data available for clustering")

            # Create user rating profiles
            user_profiles = self.ratings_df.groupby('userId').agg({
                'rating': ['mean', 'std', 'count'],
                'movieId': 'nunique'
            }).reset_index()
            
            user_profiles.columns = ['userId', 'avg_rating', 'rating_std', 'rating_count', 'unique_movies']
            user_profiles['rating_std'] = user_profiles['rating_std'].fillna(0)
            
            # Prepare features for clustering
            features = user_profiles[['avg_rating', 'rating_std', 'rating_count', 'unique_movies']].values
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Perform K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            user_profiles['cluster'] = kmeans.fit_predict(features_scaled)
            
            # Analyze each cluster
            clusters_info = []
            for cluster_id in range(n_clusters):
                cluster_users = user_profiles[user_profiles['cluster'] == cluster_id]
                
                cluster_info = {
                    'cluster_id': int(cluster_id),
                    'user_count': int(len(cluster_users)),
                    'avg_rating_mean': float(cluster_users['avg_rating'].mean()),
                    'avg_rating_std': float(cluster_users['rating_std'].mean()),
                    'avg_movies_rated': float(cluster_users['rating_count'].mean()),
                    'avg_unique_movies': float(cluster_users['unique_movies'].mean()),
                    'characteristics': self._describe_cluster(cluster_users)
                }
                clusters_info.append(cluster_info)
            
            # Calculate clustering quality metrics
            from sklearn.metrics import silhouette_score, davies_bouldin_score
            
            silhouette = float(silhouette_score(features_scaled, kmeans.labels_))
            davies_bouldin = float(davies_bouldin_score(features_scaled, kmeans.labels_))
            
            result = {
                'n_clusters': n_clusters,
                'total_users': int(len(user_profiles)),
                'clusters': clusters_info,
                'quality_metrics': {
                    'silhouette_score': silhouette,
                    'davies_bouldin_score': davies_bouldin
                },
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'insights': self._generate_clustering_insights({
                    'n_clusters': n_clusters,
                    'total_users': int(len(user_profiles)),
                    'clusters': clusters_info,
                    'quality_metrics': {
                        'silhouette_score': silhouette,
                        'davies_bouldin_score': davies_bouldin
                    }
                })
            }

            logger.info(f"User clustering completed: {n_clusters} clusters identified")
            return result
            
        except Exception as e:
            logger.error(f"Error performing user clustering: {str(e)}")
            raise DataAnalysisError(f"Failed to perform clustering: {str(e)}")
    
    def _describe_cluster(self, cluster_data: pd.DataFrame) -> str:
        """Generate a textual description of a cluster's characteristics"""
        avg_rating = cluster_data['avg_rating'].mean()
        avg_count = cluster_data['rating_count'].mean()
        avg_std = cluster_data['rating_std'].mean()
        
        if avg_rating > 3.5:
            rating_desc = "generous raters"
        elif avg_rating < 2.5:
            rating_desc = "critical raters"
        else:
            rating_desc = "moderate raters"
        
        if avg_count > 500:
            activity_desc = "highly active"
        elif avg_count > 100:
            activity_desc = "active"
        else:
            activity_desc = "casual"
        
        if avg_std > 1.0:
            consistency_desc = "diverse opinions"
        else:
            consistency_desc = "consistent opinions"
        
        return f"{activity_desc} {rating_desc} with {consistency_desc}"

    def _generate_clustering_insights(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate business insights from clustering results"""
        clusters = result['clusters']
        quality_metrics = result['quality_metrics']
        total_users = result['total_users']

        # Find largest and most active segments
        largest_cluster = max(clusters, key=lambda x: x['user_count'])
        most_active_cluster = max(clusters, key=lambda x: x['avg_movies_rated'])

        # Assess cluster quality
        silhouette = quality_metrics['silhouette_score']
        if silhouette > 0.5:
            quality_assessment = f"Excellent cluster separation (Silhouette: {silhouette:.3f}) - users form very distinct behavioral segments"
        elif silhouette > 0.3:
            quality_assessment = f"Good cluster separation (Silhouette: {silhouette:.3f}) - clear behavioral patterns identified"
        elif silhouette > 0.2:
            quality_assessment = f"Moderate cluster separation (Silhouette: {silhouette:.3f}) - identifiable user segments with some overlap"
        else:
            quality_assessment = f"Weak cluster separation (Silhouette: {silhouette:.3f}) - segments may overlap significantly"

        largest_pct = (largest_cluster['user_count'] / total_users) * 100

        return {
            "quality_assessment": quality_assessment,
            "interpretation": f"Successfully segmented {total_users:,} users into {len(clusters)} distinct behavioral groups",
            "largest_segment": f"Cluster {largest_cluster['cluster_id']}: {largest_cluster['characteristics']} ({largest_pct:.1f}% of users)",
            "most_active_segment": f"Cluster {most_active_cluster['cluster_id']}: {most_active_cluster['avg_movies_rated']:.0f} movies per user on average",
            "business_value": "User segmentation enables targeted marketing, personalized recommendations, and segment-specific feature development",
            "recommendation": f"Focus engagement strategies on Cluster {most_active_cluster['cluster_id']} (power users) and growth initiatives on Cluster {largest_cluster['cluster_id']} (largest segment)"
        }

    def _generate_trend_insights(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from trend analysis"""
        trend = result['overall_trend']
        strength = result.get('trend_strength', 0)
        volatility = result.get('rating_volatility', 0)

        # Interpret trend
        if trend == "increasing":
            trend_interp = f"Rating quality is improving over time (trend strength: {strength:.3f})"
            implication = "Growing user satisfaction indicates positive platform trajectory"
        elif trend == "decreasing":
            trend_interp = f"Rating quality is declining over time (trend strength: {strength:.3f})"
            implication = "Declining satisfaction warrants investigation into content quality or user experience issues"
        else:
            trend_interp = f"Rating quality remains stable over time (trend strength: {strength:.3f})"
            implication = "Consistent satisfaction indicates reliable content standards and user experience"

        # Assess stability
        if volatility < 0.1:
            stability = f"Very low volatility ({volatility:.3f}) - highly predictable user satisfaction"
        elif volatility < 0.2:
            stability = f"Low volatility ({volatility:.3f}) - stable and reliable satisfaction levels"
        elif volatility < 0.3:
            stability = f"Moderate volatility ({volatility:.3f}) - some fluctuation in user satisfaction"
        else:
            stability = f"High volatility ({volatility:.3f}) - significant swings in user satisfaction"

        return {
            "trend_interpretation": trend_interp,
            "stability_assessment": stability,
            "key_finding": f"The platform shows a {trend} trend with {volatility:.1%} rating volatility",
            "implication": implication,
            "recommendation": "Continue monitoring trends monthly to detect early signals of quality changes" if trend == "stable" else "Investigate root causes and implement corrective actions"
        }

    def _generate_anomaly_insights(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from anomaly detection"""
        summary = result['summary']
        anomalous_users = result.get('anomalous_users', [])
        anomalous_movies = result.get('anomalous_movies', [])
        patterns = result.get('unusual_patterns', [])

        total_anomalies = summary.get('total_anomalous_users', 0)
        detection_rate = summary.get('anomaly_percentage', 0)

        # Identify key patterns
        pattern_types = []
        if patterns:
            for pattern in patterns[:3]:  # Top 3 patterns
                pattern_types.append(f"{pattern.get('type', 'Unknown')}: {pattern.get('count', 0)} users")

        # Business impact
        if detection_rate > 15:
            impact = "High anomaly rate suggests significant outlier behavior - may indicate bot activity or data quality issues"
        elif detection_rate > 5:
            impact = "Moderate anomaly rate - normal range for user behavior diversity"
        else:
            impact = "Low anomaly rate - most users exhibit typical behavior patterns"

        return {
            "detection_summary": f"Identified {total_anomalies} anomalous users ({detection_rate:.1f}% detection rate)",
            "interpretation": f"Anomalies represent edge cases: power users, bots, or unusual behavioral patterns",
            "key_patterns": pattern_types if pattern_types else ["High-volume users (10x normal activity)", "Extreme rating patterns (all 5s or all 1s)", "Suspicious rapid-fire rating behavior"],
            "business_impact": impact,
            "recommendation": "Engage high-volume users as potential curators; investigate suspicious patterns for bot detection"
        }

    def _generate_sentiment_insights(self, sentiment_data: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """Generate insights from sentiment analysis"""
        if analysis_type == "overall":
            overall = sentiment_data.get('overall_sentiment', {})
            positive = overall.get('positive', 0)
            negative = overall.get('negative', 0)
            neutral = overall.get('neutral', 0)

            dominant = max([('Positive', positive), ('Neutral', neutral), ('Negative', negative)], key=lambda x: x[1])

            # Calculate health score
            if positive > negative * 2:
                health = f"Excellent - Positive sentiment ({positive:.1f}%) significantly exceeds negative ({negative:.1f}%)"
            elif positive > negative:
                health = f"Good - Positive sentiment ({positive:.1f}%) exceeds negative ({negative:.1f}%)"
            elif positive < negative:
                health = f"Concerning - Negative sentiment ({negative:.1f}%) exceeds positive ({positive:.1f}%)"
            else:
                health = f"Balanced - Equal positive and negative sentiment around {positive:.1f}%"

            key_findings = [
                f"{dominant[0]} sentiment dominates at {dominant[1]:.1f}%",
                f"Positive-to-negative ratio: {positive/negative if negative > 0 else 'N/A'}:1",
                f"Neutral ratings represent {neutral:.1f}% - users withreserved opinions"
            ]

            return {
                "dominant_sentiment": f"{dominant[0]} ({dominant[1]:.1f}%)",
                "interpretation": f"Platform has generally {'satisfied' if positive > 45 else 'mixed'} user base",
                "health_score": health,
                "key_findings": key_findings,
                "recommendation": "Maintain current content quality standards" if positive > 50 else "Focus on improving user satisfaction and content curation"
            }

        elif analysis_type == "movie_sentiment":
            metrics = sentiment_data.get('metrics', {})
            polarization = metrics.get('polarization_score', 0)
            consensus = metrics.get('consensus_score', 0)

            if polarization > 30:
                polar_desc = f"Highly polarizing (score: {polarization:.1f}) - divides audience opinion"
            elif polarization > 15:
                polar_desc = f"Moderately polarizing (score: {polarization:.1f}) - mixed reactions"
            else:
                polar_desc = f"Low polarization (score: {polarization:.1f}) - general agreement"

            key_findings = [
                f"Consensus score: {consensus:.1f}% agree on sentiment",
                polar_desc,
                f"Average rating: {metrics.get('average_rating', 0):.2f}/5.0"
            ]

            return {
                "dominant_sentiment": sentiment_data.get('dominant_sentiment', 'Unknown'),
                "interpretation": sentiment_data.get('interpretation', ''),
                "health_score": f"Consensus: {consensus:.1f}%, Polarization: {polarization:.1f}%",
                "key_findings": key_findings,
                "recommendation": "Highlight this movie to target audience segments" if consensus > 60 else "Consider audience targeting for polarizing content"
            }

        else:  # user_sentiment or temporal
            return {
                "dominant_sentiment": "Analysis completed",
                "interpretation": "User-specific or temporal sentiment patterns identified",
                "health_score": "See detailed metrics for assessment",
                "key_findings": ["Detailed sentiment breakdown available in response"],
                "recommendation": "Review sentiment trends for actionable insights"
            }

    def _generate_top_movies_insights(self, top_movies: List[Dict], min_ratings: int, total_found: int) -> Dict[str, Any]:
        """Generate insights for top movies analysis"""
        if not top_movies:
            return {
                "summary": "No movies found matching criteria",
                "key_finding": f"Try lowering min_ratings threshold (current: {min_ratings})",
                "methodology_note": "Using Bayesian average to balance rating quality with volume",
                "statistical_confidence": "N/A",
                "recommendation": "Adjust filters to find qualifying movies"
            }

        top_movie = top_movies[0]
        avg_rating_top_10 = sum(m.get('bayesian_avg', m.get('avg_rating', 0)) for m in top_movies[:10]) / min(10, len(top_movies))
        avg_rating_count = sum(m.get('rating_count', 0) for m in top_movies[:10]) / min(10, len(top_movies))

        return {
            "summary": f"Found {total_found} highly-rated movies meeting minimum threshold of {min_ratings} ratings",
            "key_finding": f"'{top_movie['title']}' leads with {top_movie.get('weighted_rating', top_movie.get('average_rating', 0)):.3f}/5.0 rating from {top_movie.get('rating_count', 0):,} users",
            "methodology_note": "Using Bayesian average to prevent bias - balances high ratings with sufficient user feedback volume",
            "statistical_confidence": f"Very High - Top movies average {avg_rating_count:,.0f} ratings each, well above {min_ratings} threshold",
            "recommendation": "These movies are safe recommendations for new users - proven quality with broad appeal"
        }

    def _generate_genre_insights(self, genre_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from genre analysis"""
        genre_stats = genre_analysis.get('genre_stats', [])

        if not genre_stats:
            return {
                "summary": "No genre data available",
                "most_popular_genre": "N/A",
                "highest_rated_genre": "N/A",
                "key_trends": [],
                "recommendation": "Ensure genre data is properly loaded"
            }

        # Find most popular (by rating count) and highest rated
        most_popular = max(genre_stats, key=lambda x: x.get('rating_count', 0))
        highest_rated = max(genre_stats, key=lambda x: x.get('average_rating', 0))

        key_trends = [
            f"{most_popular['genre']} is most popular with {most_popular['rating_count']:,} ratings",
            f"{highest_rated['genre']} has highest average rating ({highest_rated['average_rating']:.2f}/5.0)",
            f"Analyzed {len(genre_stats)} distinct genres"
        ]

        return {
            "summary": f"Comprehensive analysis of {len(genre_stats)} genres across the platform",
            "most_popular_genre": f"{most_popular['genre']} ({most_popular['rating_count']:,} ratings)",
            "highest_rated_genre": f"{highest_rated['genre']} ({highest_rated['average_rating']:.2f}/5.0 avg rating)",
            "key_trends": key_trends,
            "recommendation": f"Prioritize {most_popular['genre']} for volume and {highest_rated['genre']} for quality positioning"
        }

    def perform_trend_analysis(self, period: str = 'month') -> Dict[str, Any]:
        """
        Advanced time-series trend analysis of rating patterns.
        
        Args:
            period: Aggregation period ('day', 'week', 'month', 'year')
            
        Returns:
            Dictionary containing trend analysis results
            
        Raises:
            DataAnalysisError: If trend analysis fails
        """
        try:
            logger.info(f"Performing trend analysis with period: {period}")

            if self.ratings_df is None:
                self.load_datasets()

            if self.ratings_df is None or self.ratings_df.empty:
                raise DataValidationError("No ratings data available for trend analysis")

            # Convert timestamp to datetime
            ratings_with_time = self.ratings_df.copy()
            ratings_with_time['datetime'] = pd.to_datetime(ratings_with_time['timestamp'], unit='s')
            
            # Aggregate by period
            if period == 'day':
                ratings_with_time['period'] = ratings_with_time['datetime'].dt.date
            elif period == 'week':
                ratings_with_time['period'] = ratings_with_time['datetime'].dt.to_period('W').astype(str)
            elif period == 'month':
                ratings_with_time['period'] = ratings_with_time['datetime'].dt.to_period('M').astype(str)
            elif period == 'year':
                ratings_with_time['period'] = ratings_with_time['datetime'].dt.year
            else:
                raise DataValidationError(f"Invalid period: {period}")
            
            # Calculate trend metrics
            trend_data = ratings_with_time.groupby('period').agg({
                'rating': ['mean', 'std', 'count'],
                'userId': 'nunique',
                'movieId': 'nunique'
            }).reset_index()
            
            trend_data.columns = ['period', 'avg_rating', 'rating_std', 'total_ratings', 'unique_users', 'unique_movies']
            
            # Calculate rolling statistics
            window_size = min(7, len(trend_data))
            trend_data['rolling_avg'] = trend_data['avg_rating'].rolling(window=window_size, min_periods=1).mean()
            trend_data['rolling_std'] = trend_data['avg_rating'].rolling(window=window_size, min_periods=1).std()
            
            # Detect trends
            if len(trend_data) > 1:
                first_half_avg = trend_data.iloc[:len(trend_data)//2]['avg_rating'].mean()
                second_half_avg = trend_data.iloc[len(trend_data)//2:]['avg_rating'].mean()
                
                if second_half_avg > first_half_avg + 0.1:
                    overall_trend = "increasing"
                elif second_half_avg < first_half_avg - 0.1:
                    overall_trend = "decreasing"
                else:
                    overall_trend = "stable"
            else:
                overall_trend = "insufficient_data"
            
            result = {
                'period': period,
                'overall_trend': overall_trend,
                'trend_strength': float(abs(second_half_avg - first_half_avg)) if len(trend_data) > 1 else 0.0,
                'data_points': int(len(trend_data)),
                'time_series': trend_data.fillna(0).to_dict('records'),
                'statistics': {
                    'avg_rating_overall': float(trend_data['avg_rating'].mean()),
                    'rating_volatility': float(trend_data['avg_rating'].std()),
                    'avg_daily_ratings': float(trend_data['total_ratings'].mean()),
                    'peak_period': str(trend_data.loc[trend_data['total_ratings'].idxmax(), 'period']),
                    'lowest_period': str(trend_data.loc[trend_data['total_ratings'].idxmin(), 'period'])
                }
            }

            # Add insights
            result['insights'] = self._generate_trend_insights(result)

            logger.info(f"Trend analysis completed: {overall_trend} trend detected")
            return result

        except DataValidationError:
            # Re-raise validation errors as-is (will result in 400)
            raise
        except Exception as e:
            logger.error(f"Error performing trend analysis: {str(e)}")
            raise DataAnalysisError(f"Failed to perform trend analysis: {str(e)}")
    
    def detect_anomalies(self, method: str = 'iqr', sensitivity: float = 1.5) -> Dict[str, Any]:
        """
        Identify unusual rating patterns and anomalies.
        
        Args:
            method: Detection method ('iqr', 'zscore', 'isolation_forest')
            sensitivity: Sensitivity threshold (lower = more sensitive)
            
        Returns:
            Dictionary containing detected anomalies
            
        Raises:
            DataAnalysisError: If anomaly detection fails
        """
        try:
            logger.info(f"Detecting anomalies using method: {method}")

            if self.ratings_df is None:
                self.load_datasets()

            if self.ratings_df is None or self.ratings_df.empty:
                raise DataValidationError("No ratings data available for anomaly detection")

            anomalies_results = {
                'method': method,
                'sensitivity': sensitivity,
                'anomalous_users': [],
                'anomalous_movies': [],
                'unusual_patterns': []
            }
            
            # User-level anomalies
            user_stats = self.ratings_df.groupby('userId').agg({
                'rating': ['mean', 'std', 'count']
            }).reset_index()
            user_stats.columns = ['userId', 'avg_rating', 'rating_std', 'rating_count']
            
            if method == 'iqr':
                # IQR method for outlier detection
                Q1 = user_stats['rating_count'].quantile(0.25)
                Q3 = user_stats['rating_count'].quantile(0.75)
                IQR = Q3 - Q1
                threshold_upper = Q3 + sensitivity * IQR
                threshold_lower = Q1 - sensitivity * IQR
                
                anomalous_users = user_stats[
                    (user_stats['rating_count'] > threshold_upper) |
                    (user_stats['rating_count'] < threshold_lower)
                ]
                
            elif method == 'zscore':
                # Z-score method
                mean_count = user_stats['rating_count'].mean()
                std_count = user_stats['rating_count'].std()
                user_stats['z_score'] = (user_stats['rating_count'] - mean_count) / std_count
                
                anomalous_users = user_stats[abs(user_stats['z_score']) > sensitivity]
                
            elif method == 'isolation_forest':
                # Isolation Forest for multivariate anomaly detection
                from sklearn.ensemble import IsolationForest
                
                features = user_stats[['avg_rating', 'rating_count']].fillna(0).values
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                predictions = iso_forest.fit_predict(features)
                
                user_stats['anomaly'] = predictions
                anomalous_users = user_stats[user_stats['anomaly'] == -1]
            else:
                raise DataValidationError(f"Invalid method: {method}")
            
            # Movie-level anomalies
            movie_stats = self.ratings_df.groupby('movieId').agg({
                'rating': ['mean', 'std', 'count']
            }).reset_index()
            movie_stats.columns = ['movieId', 'avg_rating', 'rating_std', 'rating_count']
            
            # Find movies with unusual rating patterns
            avg_movie_rating = movie_stats['avg_rating'].mean()
            std_movie_rating = movie_stats['avg_rating'].std()
            
            unusual_movies = movie_stats[
                (abs(movie_stats['avg_rating'] - avg_movie_rating) > 2 * std_movie_rating) &
                (movie_stats['rating_count'] > 10)
            ]
            
            # Compile results
            anomalies_results['anomalous_users'] = [
                {
                    'userId': int(row['userId']),
                    'avg_rating': float(row['avg_rating']),
                    'rating_count': int(row['rating_count']),
                    'reason': 'unusual_activity_level'
                }
                for _, row in anomalous_users.head(20).iterrows()
            ]
            
            anomalies_results['anomalous_movies'] = [
                {
                    'movieId': int(row['movieId']),
                    'avg_rating': float(row['avg_rating']),
                    'rating_count': int(row['rating_count']),
                    'deviation_from_mean': float(abs(row['avg_rating'] - avg_movie_rating)),
                    'reason': 'polarizing' if row['avg_rating'] > avg_movie_rating else 'controversial'
                }
                for _, row in unusual_movies.head(20).iterrows()
            ]
            
            # Unusual patterns
            anomalies_results['unusual_patterns'] = [
                {
                    'pattern': 'high_volume_users',
                    'count': int((user_stats['rating_count'] > user_stats['rating_count'].quantile(0.95)).sum()),
                    'threshold': float(user_stats['rating_count'].quantile(0.95))
                },
                {
                    'pattern': 'extreme_critics',
                    'count': int((user_stats['avg_rating'] < 2.0).sum()),
                    'avg_rating': float(user_stats[user_stats['avg_rating'] < 2.0]['avg_rating'].mean()) if len(user_stats[user_stats['avg_rating'] < 2.0]) > 0 else 0.0
                },
                {
                    'pattern': 'generous_raters',
                    'count': int((user_stats['avg_rating'] > 4.5).sum()),
                    'avg_rating': float(user_stats[user_stats['avg_rating'] > 4.5]['avg_rating'].mean()) if len(user_stats[user_stats['avg_rating'] > 4.5]) > 0 else 0.0
                }
            ]
            
            anomalies_results['summary'] = {
                'total_anomalous_users': len(anomalies_results['anomalous_users']),
                'total_anomalous_movies': len(anomalies_results['anomalous_movies']),
                'detection_rate': float(len(anomalous_users) / len(user_stats)) if len(user_stats) > 0 else 0.0,
                'anomaly_percentage': float((len(anomalous_users) / len(user_stats)) * 100) if len(user_stats) > 0 else 0.0
            }

            # Add insights
            anomalies_results['insights'] = self._generate_anomaly_insights(anomalies_results)

            logger.info(f"Anomaly detection completed: {len(anomalies_results['anomalous_users'])} anomalous users, {len(anomalies_results['anomalous_movies'])} anomalous movies")
            return anomalies_results

        except DataValidationError:
            # Re-raise validation errors as-is (will result in 400)
            raise
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            raise DataAnalysisError(f"Failed to detect anomalies: {str(e)}")

    def analyze_rating_sentiment(
        self,
        analysis_type: str = 'overall',
        movie_id: Optional[int] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform sentiment analysis based on rating patterns.

        Converts numerical ratings to sentiment categories and analyzes patterns.

        Args:
            analysis_type: Type of analysis ('overall', 'movie_sentiment', 'user_sentiment', 'temporal_sentiment')
            movie_id: Optional movie ID for movie-specific analysis
            user_id: Optional user ID for user-specific analysis

        Returns:
            Dictionary containing sentiment analysis results

        Raises:
            DataAnalysisError: If sentiment analysis fails
            DataValidationError: If invalid parameters provided
        """
        try:
            logger.info(f"Performing rating sentiment analysis: {analysis_type}")

            # Validate analysis_type
            valid_types = ['overall', 'movie_sentiment', 'user_sentiment', 'temporal_sentiment']
            if analysis_type not in valid_types:
                raise DataValidationError(
                    f"Invalid analysis_type: '{analysis_type}'. Must be one of: {', '.join(valid_types)}"
                )

            # Validate required parameters for specific analysis types
            if analysis_type == 'movie_sentiment' and movie_id is None:
                raise DataValidationError("movie_id is required for movie_sentiment analysis")

            if analysis_type == 'user_sentiment' and user_id is None:
                raise DataValidationError("user_id is required for user_sentiment analysis")

            if self.ratings_df is None:
                self.load_datasets()

            if self.ratings_df is None or self.ratings_df.empty:
                raise DataValidationError("No ratings data available for sentiment analysis")

            # Define sentiment classification thresholds
            def classify_sentiment(rating: float) -> str:
                if rating >= 4.0:
                    return 'positive'
                elif rating >= 3.0:
                    return 'neutral'
                else:
                    return 'negative'

            # Add sentiment column
            ratings_with_sentiment = self.ratings_df.copy()
            ratings_with_sentiment['sentiment'] = ratings_with_sentiment['rating'].apply(classify_sentiment)

            result = {
                'analysis_type': analysis_type,
                'sentiment_classification': {
                    'positive': '4.0 - 5.0 stars',
                    'neutral': '3.0 - 3.5 stars',
                    'negative': '0.5 - 2.5 stars'
                }
            }

            if analysis_type == 'movie_sentiment' and movie_id:
                # Movie-specific sentiment analysis
                movie_ratings = ratings_with_sentiment[ratings_with_sentiment['movieId'] == movie_id]

                if movie_ratings.empty:
                    raise DataValidationError(f"No ratings found for movie_id: {movie_id}")

                sentiment_dist = movie_ratings['sentiment'].value_counts(normalize=True) * 100

                # Calculate polarization score
                positive_pct = sentiment_dist.get('positive', 0)
                negative_pct = sentiment_dist.get('negative', 0)
                polarization = min(positive_pct, negative_pct) * 2  # 0-100, higher = more polarized

                # Consensus score
                max_sentiment_pct = sentiment_dist.max()
                consensus = max_sentiment_pct  # Higher = more agreement

                result['movie_id'] = movie_id
                result['sentiment_distribution'] = {
                    'positive': float(sentiment_dist.get('positive', 0)),
                    'neutral': float(sentiment_dist.get('neutral', 0)),
                    'negative': float(sentiment_dist.get('negative', 0))
                }
                result['metrics'] = {
                    'total_ratings': int(len(movie_ratings)),
                    'avg_rating': float(movie_ratings['rating'].mean()),
                    'polarization_score': float(polarization),
                    'consensus_score': float(consensus),
                    'dominant_sentiment': sentiment_dist.idxmax()
                }
                result['interpretation'] = self._interpret_movie_sentiment(polarization, consensus)

            elif analysis_type == 'user_sentiment' and user_id:
                # User-specific sentiment behavior analysis
                user_ratings = ratings_with_sentiment[ratings_with_sentiment['userId'] == user_id]

                if user_ratings.empty:
                    raise DataValidationError(f"No ratings found for user_id: {user_id}")

                sentiment_dist = user_ratings['sentiment'].value_counts(normalize=True) * 100

                # Classify user type
                positive_pct = sentiment_dist.get('positive', 0)
                negative_pct = sentiment_dist.get('negative', 0)

                if positive_pct > 60:
                    user_type = 'optimistic_rater'
                elif negative_pct > 40:
                    user_type = 'critical_rater'
                else:
                    user_type = 'balanced_rater'

                result['user_id'] = user_id
                result['sentiment_distribution'] = {
                    'positive': float(sentiment_dist.get('positive', 0)),
                    'neutral': float(sentiment_dist.get('neutral', 0)),
                    'negative': float(sentiment_dist.get('negative', 0))
                }
                result['user_profile'] = {
                    'total_ratings': int(len(user_ratings)),
                    'avg_rating': float(user_ratings['rating'].mean()),
                    'rating_std': float(user_ratings['rating'].std()),
                    'user_type': user_type,
                    'dominant_sentiment': sentiment_dist.idxmax()
                }
                result['interpretation'] = self._interpret_user_sentiment(user_type, positive_pct, negative_pct)

            elif analysis_type == 'temporal_sentiment':
                # Temporal sentiment trend analysis
                ratings_with_time = ratings_with_sentiment.copy()
                ratings_with_time['datetime'] = pd.to_datetime(ratings_with_time['timestamp'], unit='s')
                ratings_with_time['year'] = ratings_with_time['datetime'].dt.year
                ratings_with_time['month'] = ratings_with_time['datetime'].dt.to_period('M').astype(str)

                # Yearly sentiment trends
                yearly_sentiment = ratings_with_time.groupby('year')['sentiment'].value_counts(normalize=True).unstack(fill_value=0) * 100

                # Monthly sentiment trends (last 24 months)
                monthly_sentiment = ratings_with_time.groupby('month')['sentiment'].value_counts(normalize=True).unstack(fill_value=0) * 100
                monthly_sentiment = monthly_sentiment.tail(24)

                result['yearly_trends'] = yearly_sentiment.to_dict('index')
                result['monthly_trends'] = monthly_sentiment.to_dict('index')
                result['overall_trend'] = {
                    'positive_change': float(yearly_sentiment['positive'].iloc[-1] - yearly_sentiment['positive'].iloc[0]) if len(yearly_sentiment) > 1 else 0.0,
                    'negative_change': float(yearly_sentiment['negative'].iloc[-1] - yearly_sentiment['negative'].iloc[0]) if len(yearly_sentiment) > 1 else 0.0
                }

            else:  # overall analysis
                # Overall platform sentiment analysis
                overall_sentiment_dist = ratings_with_sentiment['sentiment'].value_counts(normalize=True) * 100

                # User behavior classification
                user_sentiment = ratings_with_sentiment.groupby('userId').apply(
                    lambda x: (x['sentiment'].value_counts(normalize=True) * 100).to_dict()
                )

                user_types = {
                    'optimistic_raters': 0,
                    'critical_raters': 0,
                    'balanced_raters': 0
                }

                for sentiment_dist in user_sentiment:
                    positive_pct = sentiment_dist.get('positive', 0)
                    negative_pct = sentiment_dist.get('negative', 0)

                    if positive_pct > 60:
                        user_types['optimistic_raters'] += 1
                    elif negative_pct > 40:
                        user_types['critical_raters'] += 1
                    else:
                        user_types['balanced_raters'] += 1

                # Movie sentiment profiles
                movie_sentiment = self.ratings_df.groupby('movieId').agg({
                    'rating': ['mean', 'std', 'count']
                }).reset_index()
                movie_sentiment.columns = ['movieId', 'avg_rating', 'rating_std', 'rating_count']
                movie_sentiment['sentiment'] = movie_sentiment['avg_rating'].apply(classify_sentiment)

                movie_sentiment_dist = movie_sentiment['sentiment'].value_counts(normalize=True) * 100

                result['overall_sentiment'] = {
                    'positive': float(overall_sentiment_dist.get('positive', 0)),
                    'neutral': float(overall_sentiment_dist.get('neutral', 0)),
                    'negative': float(overall_sentiment_dist.get('negative', 0))
                }
                result['user_behavior'] = {
                    'optimistic_raters': int(user_types['optimistic_raters']),
                    'critical_raters': int(user_types['critical_raters']),
                    'balanced_raters': int(user_types['balanced_raters']),
                    'total_users': int(sum(user_types.values()))
                }
                result['movie_sentiment'] = {
                    'positive_movies': float(movie_sentiment_dist.get('positive', 0)),
                    'neutral_movies': float(movie_sentiment_dist.get('neutral', 0)),
                    'negative_movies': float(movie_sentiment_dist.get('negative', 0))
                }
                result['statistics'] = {
                    'total_ratings': int(len(ratings_with_sentiment)),
                    'avg_rating_platform': float(ratings_with_sentiment['rating'].mean()),
                    'sentiment_variance': float(ratings_with_sentiment['rating'].std())
                }

            # Add insights
            result['insights'] = self._generate_sentiment_insights(result, analysis_type)

            logger.info(f"Rating sentiment analysis completed: {analysis_type}")
            return result

        except DataValidationError:
            raise
        except Exception as e:
            logger.error(f"Error analyzing rating sentiment: {str(e)}")
            raise DataAnalysisError(f"Failed to analyze rating sentiment: {str(e)}")

    def _interpret_movie_sentiment(self, polarization: float, consensus: float) -> str:
        """Interpret movie sentiment metrics"""
        if consensus > 80:
            return "Strong consensus - Most viewers agree on this movie"
        elif polarization > 40:
            return "Highly polarizing - Divides audiences between love and hate"
        elif polarization > 20:
            return "Moderately divisive - Mixed but leaning toward one sentiment"
        else:
            return "General agreement with some variation in opinions"

    def _interpret_user_sentiment(self, user_type: str, positive_pct: float, negative_pct: float) -> str:
        """Interpret user sentiment behavior"""
        if user_type == 'optimistic_rater':
            return f"Optimistic rater - Tends to rate movies positively ({positive_pct:.1f}% positive ratings)"
        elif user_type == 'critical_rater':
            return f"Critical rater - More selective with high ratings ({negative_pct:.1f}% negative ratings)"
        else:
            return "Balanced rater - Provides varied ratings across the spectrum"
