"""
Pytest configuration and fixtures for unit and integration tests.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Generator, Dict, Any

# Add src to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.data_processor import DataProcessor
from src.services.movie_analyzer import MovieAnalyzer
from src.services.data_visualizer import DataVisualizer
from src.services.recommender import SimpleRecommender


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_movies_data() -> pd.DataFrame:
    """Create sample movies DataFrame for testing."""
    return pd.DataFrame({
        'movieId': [1, 2, 3, 4, 5],
        'title': [
            'Toy Story (1995)',
            'Jumanji (1995)',
            'Grumpier Old Men (1995)',
            'Waiting to Exhale (1995)',
            'Father of the Bride Part II (1995)'
        ],
        'genres': [
            'Animation|Children|Comedy',
            'Adventure|Children|Fantasy',
            'Comedy|Romance',
            'Comedy|Drama|Romance',
            'Comedy'
        ]
    })


@pytest.fixture
def sample_ratings_data() -> pd.DataFrame:
    """Create sample ratings DataFrame for testing."""
    np.random.seed(42)
    data = {
        'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        'movieId': [1, 2, 3, 1, 2, 4, 1, 3, 5, 2, 4, 5, 1, 3, 4],
        'rating': [5.0, 4.5, 4.0, 4.0, 3.5, 5.0, 5.0, 4.5, 3.0, 4.0, 4.5, 4.0, 3.5, 4.0, 5.0],
        'timestamp': pd.date_range('2020-01-01', periods=15, freq='D').astype('int64') // 10**9
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_tags_data() -> pd.DataFrame:
    """Create sample tags DataFrame for testing."""
    return pd.DataFrame({
        'userId': [1, 1, 2, 3, 4],
        'movieId': [1, 2, 1, 3, 4],
        'tag': ['fun', 'adventure', 'funny', 'heartwarming', 'family'],
        'timestamp': pd.date_range('2020-01-01', periods=5, freq='D').astype('int64') // 10**9
    })


@pytest.fixture
def sample_users_data() -> pd.DataFrame:
    """Create sample users DataFrame for testing."""
    return pd.DataFrame({
        'userId': [1, 2, 3, 4, 5],
        'gender': ['M', 'F', 'M', 'F', 'M'],
        'age': [25, 30, 22, 35, 28],
        'occupation': [1, 2, 3, 4, 5],
        'zipCode': ['10001', '10002', '10003', '10004', '10005']
    })


@pytest.fixture
def sample_dirty_data() -> pd.DataFrame:
    """Create sample DataFrame with data quality issues for testing cleaning."""
    return pd.DataFrame({
        'movieId': [1, 2, 2, 3, None, 4],  # Duplicate and missing value
        'title': ['Movie 1', 'Movie 2', 'Movie 2', 'Movie 3', 'Movie 4', 'Movie 5'],
        'rating': [5.0, 4.5, 4.5, None, 3.0, 4.0],  # Missing value
        'genre': ['Action', 'Comedy', 'Comedy', 'Drama', 'Horror', '']  # Empty string
    })


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data files."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_csv_file(temp_data_dir: Path, sample_movies_data: pd.DataFrame) -> Path:
    """Create a temporary CSV file with sample data."""
    csv_path = temp_data_dir / "test_movies.csv"
    sample_movies_data.to_csv(csv_path, index=False)
    return csv_path


# ============================================================================
# Service Instance Fixtures
# ============================================================================

@pytest.fixture
def data_processor() -> DataProcessor:
    """Create a DataProcessor instance for testing."""
    return DataProcessor()


@pytest.fixture
def movie_analyzer(data_processor: DataProcessor) -> MovieAnalyzer:
    """Create a MovieAnalyzer instance for testing."""
    return MovieAnalyzer(data_processor)


@pytest.fixture
def data_visualizer() -> DataVisualizer:
    """Create a DataVisualizer instance for testing."""
    return DataVisualizer()


@pytest.fixture
def recommender() -> SimpleRecommender:
    """Create a SimpleRecommender instance for testing."""
    return SimpleRecommender()


# ============================================================================
# Pre-loaded Data Fixtures
# ============================================================================

@pytest.fixture
def analyzer_with_data(
    movie_analyzer: MovieAnalyzer,
    sample_movies_data: pd.DataFrame,
    sample_ratings_data: pd.DataFrame
) -> MovieAnalyzer:
    """Create a MovieAnalyzer with preloaded sample data."""
    movie_analyzer.movies_df = sample_movies_data
    movie_analyzer.ratings_df = sample_ratings_data
    return movie_analyzer


@pytest.fixture
def recommender_with_data(
    recommender: SimpleRecommender,
    sample_movies_data: pd.DataFrame,
    sample_ratings_data: pd.DataFrame
) -> SimpleRecommender:
    """Create a SimpleRecommender with preloaded sample data."""
    recommender.movies_df = sample_movies_data
    recommender.ratings_df = sample_ratings_data
    recommender._build_similarity_matrix()
    return recommender


# ============================================================================
# Mock Data Fixtures for Edge Cases
# ============================================================================

@pytest.fixture
def empty_dataframe() -> pd.DataFrame:
    """Create an empty DataFrame for testing edge cases."""
    return pd.DataFrame()


@pytest.fixture
def single_row_dataframe() -> pd.DataFrame:
    """Create a DataFrame with a single row for testing edge cases."""
    return pd.DataFrame({
        'movieId': [1],
        'title': ['Test Movie'],
        'genres': ['Action']
    })


@pytest.fixture
def large_ratings_data() -> pd.DataFrame:
    """Create a larger dataset for performance testing."""
    np.random.seed(42)
    n_users = 100
    n_movies = 50
    n_ratings = 1000

    return pd.DataFrame({
        'userId': np.random.randint(1, n_users + 1, n_ratings),
        'movieId': np.random.randint(1, n_movies + 1, n_ratings),
        'rating': np.random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], n_ratings),
        'timestamp': np.random.randint(946684800, 1609459200, n_ratings)  # 2000-2020
    })


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_loggers():
    """Reset logger state between tests."""
    import logging
    # Clear all handlers from all loggers
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)
    yield


# ============================================================================
# Parametrize Fixtures
# ============================================================================

@pytest.fixture(params=[
    {'min_ratings': 1, 'limit': 5},
    {'min_ratings': 2, 'limit': 10},
    {'min_ratings': 3, 'limit': 3}
])
def top_movies_params(request) -> Dict[str, int]:
    """Parametrized fixture for testing various top movies parameters."""
    return request.param


@pytest.fixture(params=['day', 'week', 'month', 'year'])
def time_period(request) -> str:
    """Parametrized fixture for testing different time periods."""
    return request.param
