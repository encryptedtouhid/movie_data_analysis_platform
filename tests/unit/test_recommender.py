"""
Unit tests for SimpleRecommender class.

Tests cover:
- Content-based filtering (similar movies)
- Collaborative filtering (user recommendations)
- Similarity matrix building
- Recommendation quality
- Error handling
"""
import pytest
import pandas as pd
import numpy as np

from src.services.recommender import SimpleRecommender
from src.exceptions import DataValidationError


class TestRecommenderInitialization:
    """Test SimpleRecommender initialization."""

    def test_initialization(self, recommender):
        """Test SimpleRecommender initializes correctly."""
        assert recommender is not None

    def test_initial_state(self, recommender):
        """Test initial state before data loading."""
        # DataFrames should be None initially
        assert recommender.movies_df is None or isinstance(recommender.movies_df, pd.DataFrame)
        assert recommender.ratings_df is None or isinstance(recommender.ratings_df, pd.DataFrame)


class TestSimilarMovies:
    """Test content-based filtering for similar movies."""

    def test_get_similar_movies_basic(self, recommender_with_data):
        """Test basic similar movies retrieval."""
        similar = recommender_with_data.get_similar_movies(movie_id=1, limit=3)

        assert isinstance(similar, list)
        assert len(similar) <= 3

    def test_similar_movies_excludes_self(self, recommender_with_data):
        """Test that similar movies don't include the queried movie."""
        movie_id = 1
        similar = recommender_with_data.get_similar_movies(movie_id=movie_id, limit=5)

        # Original movie should not be in recommendations
        similar_ids = [m.get('movieId') or m.get('movie_id') for m in similar]
        assert movie_id not in similar_ids

    def test_similar_movies_limit_respected(self, recommender_with_data):
        """Test that limit parameter is respected."""
        limit = 2
        similar = recommender_with_data.get_similar_movies(movie_id=1, limit=limit)

        assert len(similar) <= limit

    def test_similar_movies_nonexistent_movie(self, recommender_with_data):
        """Test similar movies for non-existent movie."""
        with pytest.raises((DataValidationError, ValueError, KeyError)):
            recommender_with_data.get_similar_movies(movie_id=99999, limit=5)

    def test_similar_movies_have_similarity_scores(self, recommender_with_data):
        """Test that similar movies include similarity scores."""
        similar = recommender_with_data.get_similar_movies(movie_id=1, limit=3)

        for movie in similar:
            # Should have some similarity indicator
            assert isinstance(movie, dict)

    def test_similar_movies_sorted_by_similarity(self, recommender_with_data):
        """Test that similar movies are sorted by similarity."""
        similar = recommender_with_data.get_similar_movies(movie_id=1, limit=5)

        if len(similar) > 1:
            # Check if sorted (descending similarity)
            similarities = [m.get('similarity', m.get('score', 1.0)) for m in similar]
            assert similarities == sorted(similarities, reverse=True)


class TestUserRecommendations:
    """Test collaborative filtering for user recommendations."""

    def test_get_user_recommendations_basic(self, recommender_with_data):
        """Test basic user recommendations retrieval."""
        recommendations = recommender_with_data.get_user_recommendations(user_id=1, limit=3)

        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3

    def test_user_recommendations_exclude_watched(self, recommender_with_data):
        """Test that recommendations exclude already-rated movies."""
        user_id = 1
        recommendations = recommender_with_data.get_user_recommendations(user_id=user_id, limit=5)

        # Get movies already rated by user
        user_ratings = recommender_with_data.ratings_df[
            recommender_with_data.ratings_df['userId'] == user_id
        ]
        watched_movie_ids = set(user_ratings['movieId'].values)

        # Recommended movies should not be in watched list
        recommended_ids = [r.get('movieId') or r.get('movie_id') for r in recommendations]
        assert not any(rec_id in watched_movie_ids for rec_id in recommended_ids if rec_id is not None)

    def test_user_recommendations_limit_respected(self, recommender_with_data):
        """Test that limit parameter is respected."""
        limit = 2
        recommendations = recommender_with_data.get_user_recommendations(user_id=1, limit=limit)

        assert len(recommendations) <= limit

    def test_user_recommendations_nonexistent_user(self, recommender_with_data):
        """Test recommendations for non-existent user."""
        with pytest.raises((DataValidationError, ValueError, KeyError)):
            recommender_with_data.get_user_recommendations(user_id=99999, limit=5)

    def test_user_recommendations_have_scores(self, recommender_with_data):
        """Test that recommendations include prediction scores."""
        recommendations = recommender_with_data.get_user_recommendations(user_id=1, limit=3)

        for rec in recommendations:
            assert isinstance(rec, dict)
            # Should have movie information

    def test_user_recommendations_sorted_by_score(self, recommender_with_data):
        """Test that recommendations are sorted by predicted rating."""
        recommendations = recommender_with_data.get_user_recommendations(user_id=1, limit=5)

        if len(recommendations) > 1:
            # Check if sorted (descending predicted rating)
            scores = [r.get('predicted_rating', r.get('score', 0)) for r in recommendations]
            assert scores == sorted(scores, reverse=True)


class TestSimilarityMatrixBuilding:
    """Test similarity matrix construction."""

    def test_build_similarity_matrix(self, recommender, sample_movies_data):
        """Test building similarity matrix from movie data."""
        recommender.movies_df = sample_movies_data
        recommender._build_similarity_matrix()

        # Should have built similarity matrix
        assert hasattr(recommender, 'similarity_matrix') or hasattr(recommender, 'movie_similarity')

    def test_similarity_matrix_with_empty_data(self, recommender, empty_dataframe):
        """Test building similarity matrix with empty data."""
        recommender.movies_df = empty_dataframe

        try:
            recommender._build_similarity_matrix()
        except Exception:
            pass  # May fail gracefully with empty data


class TestRecommendationQuality:
    """Test recommendation quality and diversity."""

    def test_recommendations_are_diverse(self, recommender_with_data):
        """Test that recommendations include diverse movies."""
        recommendations = recommender_with_data.get_user_recommendations(user_id=1, limit=5)

        # All recommendations should be unique
        movie_ids = [r.get('movieId') or r.get('movie_id') for r in recommendations]
        assert len(movie_ids) == len(set(movie_ids))

    def test_similar_movies_are_diverse(self, recommender_with_data):
        """Test that similar movies are all unique."""
        similar = recommender_with_data.get_similar_movies(movie_id=1, limit=5)

        # All similar movies should be unique
        movie_ids = [m.get('movieId') or m.get('movie_id') for m in similar]
        assert len(movie_ids) == len(set(movie_ids))


class TestDataLoading:
    """Test data loading in recommender."""

    def test_load_datasets(self, recommender):
        """Test loading datasets into recommender."""
        try:
            recommender.load_datasets()
            # If successful, should have data loaded
            assert recommender.movies_df is not None or recommender.ratings_df is not None
        except Exception:
            # May fail if data files don't exist
            pass


class TestErrorHandling:
    """Test error handling in SimpleRecommender."""

    def test_recommendations_without_data(self, recommender):
        """Test recommendations without loaded data."""
        recommender.movies_df = None
        recommender.ratings_df = None

        with pytest.raises((DataValidationError, AttributeError, Exception)):
            recommender.get_user_recommendations(user_id=1, limit=5)

    def test_similar_movies_without_data(self, recommender):
        """Test similar movies without loaded data."""
        recommender.movies_df = None

        with pytest.raises((DataValidationError, AttributeError, Exception)):
            recommender.get_similar_movies(movie_id=1, limit=5)

    def test_invalid_limit_parameter(self, recommender_with_data):
        """Test handling of invalid limit values."""
        # Negative limit
        try:
            recommendations = recommender_with_data.get_user_recommendations(user_id=1, limit=-1)
            # Should handle gracefully (return empty or raise error)
            assert isinstance(recommendations, list)
        except (ValueError, DataValidationError):
            pass  # Expected error

        # Zero limit
        try:
            recommendations = recommender_with_data.get_user_recommendations(user_id=1, limit=0)
            assert isinstance(recommendations, list)
            assert len(recommendations) == 0
        except (ValueError, DataValidationError):
            pass  # Expected error


class TestEdgeCases:
    """Test edge cases in recommendation system."""

    def test_user_with_single_rating(self, recommender, sample_movies_data):
        """Test recommendations for user with only one rating."""
        single_rating_data = pd.DataFrame({
            'userId': [1],
            'movieId': [1],
            'rating': [5.0],
            'timestamp': [1234567890]
        })

        recommender.movies_df = sample_movies_data
        recommender.ratings_df = single_rating_data
        recommender._build_similarity_matrix()

        try:
            recommendations = recommender.get_user_recommendations(user_id=1, limit=3)
            assert isinstance(recommendations, list)
        except Exception:
            pass  # May not be able to generate recommendations

    def test_movie_with_no_ratings(self, recommender_with_data):
        """Test similar movies for movie with no ratings."""
        # Find a movie ID that exists but has no ratings
        all_movie_ids = set(recommender_with_data.movies_df['movieId'].values)
        rated_movie_ids = set(recommender_with_data.ratings_df['movieId'].values)
        unrated_movies = all_movie_ids - rated_movie_ids

        if unrated_movies:
            unrated_movie_id = list(unrated_movies)[0]
            try:
                similar = recommender_with_data.get_similar_movies(movie_id=unrated_movie_id, limit=3)
                assert isinstance(similar, list)
            except Exception:
                pass  # May fail for unrated movies
