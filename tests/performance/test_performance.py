"""
Performance tests for Movie Data Analysis Platform.

Tests cover:
- Execution time benchmarks
- Memory usage profiling
- Data processing efficiency
- API response times
- Large dataset handling
"""
import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
from pathlib import Path
from typing import Dict, Any

from src.services.data_processor import DataProcessor
from src.services.movie_analyzer import MovieAnalyzer
from src.services.recommender import SimpleRecommender


# Mark all tests in this module as performance tests
pytestmark = [pytest.mark.performance, pytest.mark.slow]


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def measure_execution_time(func, *args, **kwargs) -> tuple:
    """
    Measure execution time of a function.

    Returns:
        tuple: (result, execution_time_seconds)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time


def measure_memory(func, *args, **kwargs) -> tuple:
    """
    Measure memory usage of a function.

    Returns:
        tuple: (result, memory_used_mb)
    """
    initial_memory = get_memory_usage()
    result = func(*args, **kwargs)
    final_memory = get_memory_usage()
    memory_used = final_memory - initial_memory
    return result, memory_used


class TestDataProcessingPerformance:
    """Performance tests for data processing operations."""

    @pytest.fixture
    def large_dataset(self) -> pd.DataFrame:
        """Create large dataset for performance testing."""
        n_rows = 100000
        return pd.DataFrame({
            'movieId': np.random.randint(1, 10000, n_rows),
            'userId': np.random.randint(1, 50000, n_rows),
            'rating': np.random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], n_rows),
            'timestamp': np.random.randint(946684800, 1609459200, n_rows)
        })

    def test_load_data_performance(self, data_processor, temp_data_dir, large_dataset):
        """Test data loading performance with large files."""
        # Create large CSV file
        csv_file = temp_data_dir / "large_ratings.csv"
        large_dataset.to_csv(csv_file, index=False)

        # Measure loading time
        _, execution_time = measure_execution_time(
            data_processor.load_data,
            str(csv_file)
        )

        print(f"\nLoad Data Performance:")
        print(f"  Rows: {len(large_dataset):,}")
        print(f"  Time: {execution_time:.3f}s")
        print(f"  Throughput: {len(large_dataset)/execution_time:,.0f} rows/sec")

        # Performance assertions
        assert execution_time < 5.0, f"Loading took {execution_time:.2f}s, expected <5s"

    def test_clean_data_performance(self, data_processor, large_dataset):
        """Test data cleaning performance."""
        # Add some dirty data
        dirty_df = large_dataset.copy()
        dirty_df.loc[::10, 'rating'] = None  # 10% missing values
        dirty_df = pd.concat([dirty_df, dirty_df.head(1000)])  # Add duplicates

        _, execution_time = measure_execution_time(
            data_processor.clean_data,
            dirty_df
        )

        print(f"\nClean Data Performance:")
        print(f"  Rows: {len(dirty_df):,}")
        print(f"  Time: {execution_time:.3f}s")

        assert execution_time < 3.0, f"Cleaning took {execution_time:.2f}s, expected <3s"

    def test_aggregate_statistics_performance(self, data_processor, large_dataset):
        """Test statistical aggregation performance."""
        _, execution_time = measure_execution_time(
            data_processor.aggregate_statistics,
            large_dataset
        )

        print(f"\nAggregate Statistics Performance:")
        print(f"  Rows: {len(large_dataset):,}")
        print(f"  Time: {execution_time:.3f}s")

        assert execution_time < 2.0, f"Aggregation took {execution_time:.2f}s, expected <2s"

    def test_filter_data_performance(self, data_processor, large_dataset):
        """Test data filtering performance."""
        _, execution_time = measure_execution_time(
            data_processor.filter_data,
            large_dataset,
            min_rating=4.0
        )

        print(f"\nFilter Data Performance:")
        print(f"  Rows: {len(large_dataset):,}")
        print(f"  Time: {execution_time:.3f}s")

        assert execution_time < 1.0, f"Filtering took {execution_time:.2f}s, expected <1s"


class TestAnalysisPerformance:
    """Performance tests for analysis operations."""

    @pytest.fixture
    def analyzer_with_large_data(self, movie_analyzer, large_ratings_data):
        """MovieAnalyzer with large dataset."""
        movie_analyzer.ratings_df = large_ratings_data
        movie_analyzer.movies_df = pd.DataFrame({
            'movieId': range(1, 10001),
            'title': [f'Movie {i}' for i in range(1, 10001)],
            'genres': ['Action|Drama'] * 10000
        })
        return movie_analyzer

    def test_top_movies_performance(self, analyzer_with_large_data):
        """Test top movies calculation performance."""
        _, execution_time = measure_execution_time(
            analyzer_with_large_data.get_top_movies,
            limit=100,
            min_ratings=10
        )

        print(f"\nTop Movies Performance:")
        print(f"  Time: {execution_time:.3f}s")

        assert execution_time < 5.0, f"Top movies took {execution_time:.2f}s, expected <5s"

    def test_genre_trends_performance(self, analyzer_with_large_data):
        """Test genre trends analysis performance."""
        _, execution_time = measure_execution_time(
            analyzer_with_large_data.analyze_genre_trends
        )

        print(f"\nGenre Trends Performance:")
        print(f"  Time: {execution_time:.3f}s")

        assert execution_time < 10.0, f"Genre trends took {execution_time:.2f}s, expected <10s"

    def test_user_statistics_performance(self, analyzer_with_large_data):
        """Test user statistics generation performance."""
        _, execution_time = measure_execution_time(
            analyzer_with_large_data.get_user_statistics,
            user_id=1
        )

        print(f"\nUser Statistics Performance:")
        print(f"  Time: {execution_time:.3f}s")

        assert execution_time < 2.0, f"User stats took {execution_time:.2f}s, expected <2s"


class TestMemoryUsage:
    """Memory usage tests for data operations."""

    def test_large_dataframe_memory(self, data_processor):
        """Test memory usage when loading large datasets."""
        initial_memory = get_memory_usage()

        # Create large dataset
        large_df = pd.DataFrame({
            'col1': np.random.random(1000000),
            'col2': np.random.randint(0, 100, 1000000),
            'col3': ['test' * 10] * 1000000
        })

        final_memory = get_memory_usage()
        memory_used = final_memory - initial_memory

        print(f"\nMemory Usage Test:")
        print(f"  Rows: {len(large_df):,}")
        print(f"  Memory used: {memory_used:.2f} MB")
        print(f"  Memory per row: {memory_used * 1024 / len(large_df):.2f} KB")

        # Memory should be reasonable (less than 500MB for 1M rows)
        assert memory_used < 500, f"Memory usage {memory_used:.2f}MB exceeds 500MB"

    def test_chunked_processing_memory(self, data_processor):
        """Test memory efficiency of chunked processing."""
        initial_memory = get_memory_usage()

        # Simulate chunked processing
        total_processed = 0
        chunk_size = data_processor.chunk_size

        for i in range(5):  # Process 5 chunks
            chunk = pd.DataFrame({
                'data': np.random.random(chunk_size)
            })
            total_processed += len(chunk)

        final_memory = get_memory_usage()
        memory_used = final_memory - initial_memory

        print(f"\nChunked Processing Memory:")
        print(f"  Total processed: {total_processed:,} rows")
        print(f"  Memory used: {memory_used:.2f} MB")

        # Should use minimal additional memory due to chunking
        assert memory_used < 100, f"Chunked processing used {memory_used:.2f}MB, expected <100MB"


class TestRecommenderPerformance:
    """Performance tests for recommendation system."""

    @pytest.fixture
    def recommender_with_large_data(self, recommender):
        """Recommender with large dataset."""
        # Create large synthetic dataset
        n_users = 10000
        n_movies = 5000
        n_ratings = 100000

        recommender.movies_df = pd.DataFrame({
            'movieId': range(1, n_movies + 1),
            'title': [f'Movie {i}' for i in range(1, n_movies + 1)],
            'genres': ['Action|Drama'] * n_movies
        })

        recommender.ratings_df = pd.DataFrame({
            'userId': np.random.randint(1, n_users + 1, n_ratings),
            'movieId': np.random.randint(1, n_movies + 1, n_ratings),
            'rating': np.random.choice([3.0, 3.5, 4.0, 4.5, 5.0], n_ratings),
            'timestamp': np.random.randint(946684800, 1609459200, n_ratings)
        })

        return recommender

    def test_similarity_matrix_build_performance(self, recommender_with_large_data):
        """Test similarity matrix building performance."""
        _, execution_time = measure_execution_time(
            recommender_with_large_data._build_similarity_matrix
        )

        print(f"\nSimilarity Matrix Build Performance:")
        print(f"  Movies: {len(recommender_with_large_data.movies_df):,}")
        print(f"  Time: {execution_time:.3f}s")

        assert execution_time < 30.0, f"Matrix build took {execution_time:.2f}s, expected <30s"

    def test_similar_movies_performance(self, recommender_with_large_data):
        """Test similar movies recommendation performance."""
        recommender_with_large_data._build_similarity_matrix()

        _, execution_time = measure_execution_time(
            recommender_with_large_data.get_similar_movies,
            movie_id=1,
            limit=10
        )

        print(f"\nSimilar Movies Performance:")
        print(f"  Time: {execution_time:.3f}s")

        assert execution_time < 1.0, f"Similar movies took {execution_time:.2f}s, expected <1s"

    def test_user_recommendations_performance(self, recommender_with_large_data):
        """Test user recommendations performance."""
        recommender_with_large_data._build_similarity_matrix()

        _, execution_time = measure_execution_time(
            recommender_with_large_data.get_user_recommendations,
            user_id=1,
            limit=10
        )

        print(f"\nUser Recommendations Performance:")
        print(f"  Time: {execution_time:.3f}s")

        assert execution_time < 2.0, f"Recommendations took {execution_time:.2f}s, expected <2s"


class TestPerformanceRegression:
    """Tests to detect performance regression."""

    # Baseline performance metrics (update these as you optimize)
    BASELINES = {
        'load_10k_rows': 0.5,  # seconds
        'clean_10k_rows': 0.3,  # seconds
        'aggregate_10k_rows': 0.2,  # seconds
        'top_movies_calc': 1.0,  # seconds
    }

    def test_no_performance_regression(self, data_processor):
        """Ensure performance doesn't degrade over time."""
        # Create test dataset
        test_df = pd.DataFrame({
            'movieId': np.random.randint(1, 100, 10000),
            'rating': np.random.choice([3.0, 4.0, 5.0], 10000)
        })

        # Test cleaning performance
        _, clean_time = measure_execution_time(
            data_processor.clean_data,
            test_df
        )

        # Test aggregation performance
        _, agg_time = measure_execution_time(
            data_processor.aggregate_statistics,
            test_df
        )

        print(f"\nRegression Test Results:")
        print(f"  Clean time: {clean_time:.3f}s (baseline: {self.BASELINES['clean_10k_rows']:.3f}s)")
        print(f"  Aggregate time: {agg_time:.3f}s (baseline: {self.BASELINES['aggregate_10k_rows']:.3f}s)")

        # Assert no significant regression (allow 50% overhead)
        assert clean_time < self.BASELINES['clean_10k_rows'] * 1.5, \
            f"Performance regression in clean_data: {clean_time:.3f}s vs baseline {self.BASELINES['clean_10k_rows']:.3f}s"

        assert agg_time < self.BASELINES['aggregate_10k_rows'] * 1.5, \
            f"Performance regression in aggregate_statistics: {agg_time:.3f}s vs baseline {self.BASELINES['aggregate_10k_rows']:.3f}s"


@pytest.mark.benchmark
class TestBenchmarks:
    """Benchmark tests for comparison."""

    def test_pandas_vs_numpy_performance(self):
        """Compare pandas vs numpy operations."""
        size = 1000000

        # Pandas operation
        df = pd.DataFrame({'col': np.random.random(size)})
        start = time.time()
        result_pandas = df['col'].mean()
        pandas_time = time.time() - start

        # Numpy operation
        arr = np.random.random(size)
        start = time.time()
        result_numpy = np.mean(arr)
        numpy_time = time.time() - start

        print(f"\nPandas vs Numpy Benchmark:")
        print(f"  Pandas mean: {pandas_time:.6f}s")
        print(f"  Numpy mean: {numpy_time:.6f}s")
        print(f"  Speedup: {pandas_time/numpy_time:.2f}x")

        # Results should be approximately equal
        assert abs(result_pandas - result_numpy) < 0.0001
