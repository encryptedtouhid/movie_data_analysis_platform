"""
Machine Learning Recommendation Service

This module provides movie recommendations using:
1. Content-based filtering (similar movies based on genre and ratings)
2. Collaborative filtering (user-based recommendations)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import logging

from src.services.data_processor import DataProcessor
from src.exceptions.data_exceptions import DataAnalysisError, DataLoadError

logger = logging.getLogger(__name__)


class SimpleRecommender:
    """
    Simple movie recommender system using content-based and collaborative filtering.

    Attributes:
        data_processor: DataProcessor instance for loading data
        movies_df: DataFrame containing movie information
        ratings_df: DataFrame containing user ratings
        user_item_matrix: Sparse matrix of user-item ratings for collaborative filtering
        movie_features: Feature matrix for content-based filtering
    """

    def __init__(self, data_processor: Optional[DataProcessor] = None):
        """
        Initialize the recommender system.

        Args:
            data_processor: Optional DataProcessor instance. If None, creates a new one.
        """
        self.data_processor = data_processor or DataProcessor()
        self.movies_df: Optional[pd.DataFrame] = None
        self.ratings_df: Optional[pd.DataFrame] = None
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.movie_features: Optional[pd.DataFrame] = None
        self._is_initialized = False

        logger.info("SimpleRecommender initialized")

    def initialize(self) -> None:
        """
        Load and prepare data for recommendations.
        Lazy initialization to avoid loading data on instantiation.

        Raises:
            DataLoadError: If data files cannot be loaded
        """
        try:
            logger.info("Initializing recommender system...")

            # Construct file paths - try cleaned files first, then regular files
            processed_path = Path(self.data_processor.data_processed_path)

            # Try to load movies
            movies_file = processed_path / "movies_cleaned.csv"
            if not movies_file.exists():
                movies_file = processed_path / "movies.csv"
            if not movies_file.exists():
                raise DataLoadError("Movies data file not found in processed folder")

            # Try to load ratings
            ratings_file = processed_path / "ratings_cleaned.csv"
            if not ratings_file.exists():
                ratings_file = processed_path / "ratings.csv"
            if not ratings_file.exists():
                raise DataLoadError("Ratings data file not found in processed folder")

            # Load the data
            self.movies_df = self.data_processor.load_data(str(movies_file))
            self.ratings_df = self.data_processor.load_data(str(ratings_file))

            if self.movies_df is None or self.ratings_df is None:
                raise DataLoadError("Failed to load movies or ratings data")

            # Prepare user-item matrix for collaborative filtering
            self._prepare_user_item_matrix()

            # Prepare movie features for content-based filtering
            self._prepare_movie_features()

            self._is_initialized = True
            logger.info(
                f"Recommender initialized with {len(self.movies_df)} movies "
                f"and {len(self.ratings_df)} ratings"
            )

        except Exception as e:
            logger.error(f"Failed to initialize recommender: {str(e)}")
            raise DataLoadError(f"Recommender initialization failed: {str(e)}")

    def _ensure_initialized(self) -> None:
        """Ensure the recommender is initialized before use."""
        if not self._is_initialized:
            self.initialize()

    def _prepare_user_item_matrix(self) -> None:
        """
        Create a user-item rating matrix for collaborative filtering.
        Uses sparse representation for memory efficiency.
        """
        logger.info("Preparing user-item matrix...")

        # Create pivot table: rows=users, columns=movies, values=ratings
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )

        logger.info(
            f"User-item matrix created: {self.user_item_matrix.shape[0]} users × "
            f"{self.user_item_matrix.shape[1]} movies"
        )

    def _prepare_movie_features(self) -> None:
        """
        Prepare movie features for content-based filtering.
        Creates a feature matrix based on:
        1. Genre vectors (multi-hot encoding)
        2. Average rating
        3. Rating count (popularity)
        """
        logger.info("Preparing movie features...")

        # Calculate movie statistics
        movie_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        movie_stats.columns = ['movieId', 'AvgRating', 'RatingCount']

        # Merge with movie data
        movies_with_stats = self.movies_df.merge(movie_stats, on='movieId', how='left')
        movies_with_stats['AvgRating'] = movies_with_stats['AvgRating'].fillna(0)
        movies_with_stats['RatingCount'] = movies_with_stats['RatingCount'].fillna(0)

        # Process genres - split and create multi-hot encoding
        movies_with_stats['GenreList'] = movies_with_stats['genres'].str.split('|')

        # Create genre matrix using MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(movies_with_stats['GenreList'])
        genre_df = pd.DataFrame(
            genre_matrix,
            columns=[f'genre_{g}' for g in mlb.classes_],
            index=movies_with_stats['movieId']
        )

        # Normalize rating and count for feature scaling
        avg_rating_norm = movies_with_stats['AvgRating'] / 5.0  # Scale to 0-1
        rating_count_log = np.log1p(movies_with_stats['RatingCount'])  # Log transform
        rating_count_norm = (
            (rating_count_log - rating_count_log.min()) /
            (rating_count_log.max() - rating_count_log.min() + 1e-10)
        )

        # Combine all features
        self.movie_features = genre_df.copy()
        self.movie_features['avg_rating'] = avg_rating_norm.values
        self.movie_features['popularity'] = rating_count_norm.values

        logger.info(f"Movie features prepared: {self.movie_features.shape[1]} features per movie")

    def get_similar_movies(
        self,
        movie_id: int,
        limit: int = 10,
        min_common_ratings: int = 50
    ) -> List[Dict]:
        """
        Find similar movies based on genre and ratings using content-based filtering.

        Uses cosine similarity on movie features including:
        - Genre vectors (multi-hot encoded)
        - Average rating (normalized)
        - Popularity (log-normalized rating count)

        Args:
            movie_id: ID of the movie to find similar movies for
            limit: Maximum number of similar movies to return
            min_common_ratings: Minimum number of ratings for a movie to be considered

        Returns:
            List of dictionaries containing similar movie information:
            - MovieID: Movie identifier
            - Title: Movie title
            - Genres: Movie genres
            - Similarity: Similarity score (0-1)
            - AvgRating: Average rating
            - RatingCount: Number of ratings

        Raises:
            DataAnalysisError: If movie_id not found or analysis fails
        """
        self._ensure_initialized()

        try:
            logger.info(f"Finding similar movies for MovieID={movie_id}")

            # Check if movie exists
            if movie_id not in self.movie_features.index:
                raise DataAnalysisError(f"Movie with ID {movie_id} not found")

            # Get the movie's feature vector
            target_features = self.movie_features.loc[[movie_id]]

            # Filter movies by minimum rating count
            movie_stats = self.ratings_df.groupby('movieId')['rating'].agg(['mean', 'count'])
            movie_stats.columns = ['AvgRating', 'RatingCount']
            valid_movies = movie_stats[movie_stats['RatingCount'] >= min_common_ratings].index

            # Filter feature matrix to only include valid movies
            valid_features = self.movie_features.loc[
                self.movie_features.index.isin(valid_movies)
            ]

            # Exclude the target movie itself
            valid_features = valid_features.drop(movie_id, errors='ignore')

            if len(valid_features) == 0:
                logger.warning("No valid movies found for similarity comparison")
                return []

            # Calculate cosine similarity
            similarities = cosine_similarity(target_features, valid_features)[0]

            # Get top similar movies
            similar_indices = np.argsort(similarities)[::-1][:limit]
            similar_movie_ids = valid_features.index[similar_indices]
            similarity_scores = similarities[similar_indices]

            # Prepare results with movie information
            results = []
            for movie_id_sim, score in zip(similar_movie_ids, similarity_scores):
                movie_info = self.movies_df[self.movies_df['movieId'] == movie_id_sim].iloc[0]
                movie_stat = movie_stats.loc[movie_id_sim]

                results.append({
                    'MovieID': int(movie_id_sim),
                    'Title': movie_info['title'],
                    'Genres': movie_info['genres'],
                    'Similarity': float(score),
                    'AvgRating': float(movie_stat['AvgRating']),
                    'RatingCount': int(movie_stat['RatingCount'])
                })

            logger.info(f"Found {len(results)} similar movies")
            return results

        except DataAnalysisError:
            raise
        except Exception as e:
            logger.error(f"Error finding similar movies: {str(e)}")
            raise DataAnalysisError(f"Failed to find similar movies: {str(e)}")

    def get_user_recommendations(
        self,
        user_id: int,
        limit: int = 10,
        min_user_overlap: int = 50
    ) -> List[Dict]:
        """
        Generate movie recommendations using collaborative filtering.

        Uses user-based collaborative filtering:
        1. Find similar users based on rating patterns (cosine similarity)
        2. Recommend movies that similar users liked but target user hasn't seen
        3. Weight recommendations by user similarity and movie ratings

        Args:
            user_id: ID of the user to generate recommendations for
            limit: Maximum number of recommendations to return
            min_user_overlap: Minimum number of common ratings to consider users similar

        Returns:
            List of dictionaries containing recommended movies:
            - MovieID: Movie identifier
            - Title: Movie title
            - Genres: Movie genres
            - PredictedRating: Predicted rating for the user
            - AvgRating: Average rating across all users
            - RatingCount: Number of ratings

        Raises:
            DataAnalysisError: If user_id not found or analysis fails
        """
        self._ensure_initialized()

        try:
            logger.info(f"Generating recommendations for UserID={user_id}")

            # Check if user exists
            if user_id not in self.user_item_matrix.index:
                raise DataAnalysisError(f"User with ID {user_id} not found")

            # Get user's rating vector
            user_ratings = self.user_item_matrix.loc[user_id]

            # Find movies the user hasn't rated
            unrated_movies = user_ratings[user_ratings == 0].index.tolist()

            if len(unrated_movies) == 0:
                logger.warning("User has rated all movies")
                return []

            # Find similar users
            similar_users = self._find_similar_users(
                user_id,
                min_overlap=min_user_overlap,
                top_k=50  # Use top 50 similar users
            )

            if len(similar_users) == 0:
                logger.warning("No similar users found")
                return self._get_popular_recommendations(limit, exclude_user=user_id)

            # Generate predictions for unrated movies
            predictions = []
            for movie_id in unrated_movies:
                predicted_rating = self._predict_rating(
                    user_id,
                    movie_id,
                    similar_users
                )

                if predicted_rating > 0:
                    predictions.append((movie_id, predicted_rating))

            # Sort by predicted rating and get top N
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_predictions = predictions[:limit]

            # Prepare results with movie information
            results = []
            for movie_id, pred_rating in top_predictions:
                movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
                movie_ratings = self.ratings_df[self.ratings_df['movieId'] == movie_id]['rating']

                results.append({
                    'MovieID': int(movie_id),
                    'Title': movie_info['title'],
                    'Genres': movie_info['genres'],
                    'PredictedRating': float(pred_rating),
                    'AvgRating': float(movie_ratings.mean()),
                    'RatingCount': int(len(movie_ratings))
                })

            logger.info(f"Generated {len(results)} recommendations")
            return results

        except DataAnalysisError:
            raise
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise DataAnalysisError(f"Failed to generate recommendations: {str(e)}")

    def _find_similar_users(
        self,
        user_id: int,
        min_overlap: int = 50,
        top_k: int = 50
    ) -> List[Tuple[int, float]]:
        """
        Find users similar to the target user based on rating patterns.

        Args:
            user_id: Target user ID
            min_overlap: Minimum number of commonly rated movies
            top_k: Number of similar users to return

        Returns:
            List of tuples (user_id, similarity_score)
        """
        user_ratings = self.user_item_matrix.loc[user_id].values.reshape(1, -1)

        # Calculate cosine similarity with all users
        similarities = cosine_similarity(user_ratings, self.user_item_matrix.values)[0]

        # Check overlap (number of commonly rated movies)
        user_rated_mask = self.user_item_matrix.loc[user_id] > 0
        other_rated_mask = self.user_item_matrix > 0
        overlap_counts = (other_rated_mask & user_rated_mask).sum(axis=1)

        # Filter by minimum overlap and exclude self
        valid_users_mask = (overlap_counts >= min_overlap) & (self.user_item_matrix.index != user_id)
        valid_user_indices = np.where(valid_users_mask)[0]

        if len(valid_user_indices) == 0:
            return []

        # Get similarities for valid users
        valid_similarities = similarities[valid_user_indices]
        valid_user_ids = self.user_item_matrix.index[valid_user_indices]

        # Sort by similarity and get top K
        top_indices = np.argsort(valid_similarities)[::-1][:top_k]

        similar_users = [
            (int(valid_user_ids[i]), float(valid_similarities[i]))
            for i in top_indices
            if valid_similarities[i] > 0  # Only positive similarities
        ]

        return similar_users

    def _predict_rating(
        self,
        user_id: int,
        movie_id: int,
        similar_users: List[Tuple[int, float]]
    ) -> float:
        """
        Predict a user's rating for a movie based on similar users' ratings.

        Uses weighted average of similar users' ratings, where weights are
        the similarity scores.

        Args:
            user_id: Target user ID
            movie_id: Movie to predict rating for
            similar_users: List of (user_id, similarity) tuples

        Returns:
            Predicted rating (0 if cannot predict)
        """
        weighted_sum = 0.0
        similarity_sum = 0.0

        for similar_user_id, similarity in similar_users:
            rating = self.user_item_matrix.loc[similar_user_id, movie_id]

            if rating > 0:  # User has rated this movie
                weighted_sum += similarity * rating
                similarity_sum += similarity

        if similarity_sum == 0:
            return 0.0

        return weighted_sum / similarity_sum

    def _get_popular_recommendations(
        self,
        limit: int,
        exclude_user: Optional[int] = None
    ) -> List[Dict]:
        """
        Fallback: Get popular movies when collaborative filtering fails.

        Returns highly-rated popular movies that the user hasn't seen.

        Args:
            limit: Number of recommendations
            exclude_user: User ID to exclude their rated movies

        Returns:
            List of popular movie recommendations
        """
        logger.info("Using popular recommendations as fallback")

        # Calculate movie statistics
        movie_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        })
        movie_stats.columns = ['AvgRating', 'RatingCount']

        # Filter movies with enough ratings (at least 100)
        popular_movies = movie_stats[movie_stats['RatingCount'] >= 100].copy()

        # Calculate weighted rating (Bayesian average)
        C = movie_stats['AvgRating'].mean()  # Mean rating across all movies
        m = 100  # Minimum votes required
        popular_movies['WeightedRating'] = (
            (popular_movies['RatingCount'] / (popular_movies['RatingCount'] + m)) * popular_movies['AvgRating'] +
            (m / (popular_movies['RatingCount'] + m)) * C
        )

        # Exclude movies rated by the user
        if exclude_user is not None and exclude_user in self.user_item_matrix.index:
            user_rated = self.user_item_matrix.loc[exclude_user]
            user_rated_movies = user_rated[user_rated > 0].index
            popular_movies = popular_movies.drop(user_rated_movies, errors='ignore')

        # Sort by weighted rating
        popular_movies = popular_movies.sort_values('WeightedRating', ascending=False)

        # Get top N
        top_movie_ids = popular_movies.head(limit).index

        # Prepare results
        results = []
        for movie_id in top_movie_ids:
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
            stats = popular_movies.loc[movie_id]

            results.append({
                'MovieID': int(movie_id),
                'Title': movie_info['title'],
                'Genres': movie_info['genres'],
                'PredictedRating': float(stats['WeightedRating']),
                'AvgRating': float(stats['AvgRating']),
                'RatingCount': int(stats['RatingCount'])
            })

        return results

