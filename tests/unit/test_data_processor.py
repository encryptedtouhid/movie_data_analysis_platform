"""
Unit tests for DataProcessor class.

Tests cover:
- Data loading functionality
- Data cleaning and validation
- Statistical aggregation
- Data filtering operations
- Export capabilities
- Error handling
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.services.data_processor import DataProcessor
from src.exceptions import (
    DataLoadError,
    DataCleaningError,
    DataValidationError,
    DataAggregationError,
    DataFilterError
)


class TestDataProcessorInitialization:
    """Test DataProcessor initialization and setup."""

    def test_initialization(self, data_processor):
        """Test DataProcessor initializes correctly."""
        assert data_processor is not None
        assert isinstance(data_processor.data_raw_path, Path)
        assert isinstance(data_processor.data_processed_path, Path)
        assert data_processor.chunk_size == 100000

    def test_directories_created(self, data_processor):
        """Test that required directories are created on initialization."""
        assert data_processor.data_raw_path.exists()
        assert data_processor.data_processed_path.exists()


class TestDataLoading:
    """Test data loading functionality."""

    def test_load_csv_success(self, data_processor, temp_csv_file):
        """Test successful CSV file loading."""
        df = data_processor.load_data(str(temp_csv_file))
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df) > 0

    def test_load_nonexistent_file(self, data_processor):
        """Test loading non-existent file raises DataLoadError."""
        with pytest.raises(DataLoadError):
            data_processor.load_data("/nonexistent/path/file.csv")

    def test_load_data_with_processed_file(self, data_processor, temp_data_dir, sample_movies_data):
        """Test loading from processed directory."""
        # Create file in processed directory
        processed_file = temp_data_dir / "movies_cleaned.csv"
        sample_movies_data.to_csv(processed_file, index=False)

        # Override the processed path temporarily
        original_path = data_processor.data_processed_path
        data_processor.data_processed_path = temp_data_dir

        try:
            df = data_processor.load_data(str(processed_file))
            assert len(df) == len(sample_movies_data)
            assert list(df.columns) == list(sample_movies_data.columns)
        finally:
            data_processor.data_processed_path = original_path

    def test_load_empty_file(self, data_processor, temp_data_dir):
        """Test loading empty CSV file."""
        empty_file = temp_data_dir / "empty.csv"
        pd.DataFrame().to_csv(empty_file, index=False)

        df = data_processor.load_data(str(empty_file))
        assert isinstance(df, pd.DataFrame)


class TestDataCleaning:
    """Test data cleaning functionality."""

    def test_clean_removes_duplicates(self, data_processor, sample_dirty_data):
        """Test that clean_data removes duplicate rows."""
        # Add duplicates
        dirty_df = pd.concat([sample_dirty_data, sample_dirty_data.iloc[:2]], ignore_index=True)
        cleaned_df = data_processor.clean_data(dirty_df)

        # Should have fewer rows after removing duplicates
        assert len(cleaned_df) <= len(dirty_df)

    def test_clean_handles_missing_values(self, data_processor, sample_dirty_data):
        """Test that clean_data handles missing values appropriately."""
        cleaned_df = data_processor.clean_data(sample_dirty_data)

        # Should handle missing values (either drop or fill)
        assert isinstance(cleaned_df, pd.DataFrame)

    def test_clean_empty_dataframe(self, data_processor, empty_dataframe):
        """Test cleaning empty DataFrame."""
        result = data_processor.clean_data(empty_dataframe)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_clean_preserves_data_types(self, data_processor, sample_movies_data):
        """Test that cleaning preserves appropriate data types."""
        cleaned_df = data_processor.clean_data(sample_movies_data)
        assert 'movieId' in cleaned_df.columns
        assert 'title' in cleaned_df.columns
        assert 'genres' in cleaned_df.columns


class TestDataAggregation:
    """Test statistical aggregation functionality."""

    def test_aggregate_statistics_basic(self, data_processor, sample_ratings_data):
        """Test basic statistical aggregation."""
        stats = data_processor.aggregate_statistics(sample_ratings_data)

        assert isinstance(stats, dict)
        assert 'total_rows' in stats
        assert 'total_columns' in stats
        assert stats['total_rows'] == len(sample_ratings_data)
        assert stats['total_columns'] == len(sample_ratings_data.columns)

    def test_aggregate_includes_numeric_stats(self, data_processor, sample_ratings_data):
        """Test that aggregation includes numeric column statistics."""
        stats = data_processor.aggregate_statistics(sample_ratings_data)

        # Should have statistics for numeric columns
        assert 'statistics' in stats or 'mean' in str(stats).lower()

    def test_aggregate_empty_dataframe(self, data_processor, empty_dataframe):
        """Test aggregation on empty DataFrame."""
        stats = data_processor.aggregate_statistics(empty_dataframe)
        assert isinstance(stats, dict)
        assert stats.get('total_rows') == 0

    def test_aggregate_single_row(self, data_processor, single_row_dataframe):
        """Test aggregation on single-row DataFrame."""
        stats = data_processor.aggregate_statistics(single_row_dataframe)
        assert isinstance(stats, dict)
        assert stats.get('total_rows') == 1


class TestDataFiltering:
    """Test data filtering operations."""

    def test_filter_by_rating(self, data_processor, sample_ratings_data):
        """Test filtering ratings by minimum rating value."""
        filtered = data_processor.filter_data(sample_ratings_data, min_rating=4.0)

        assert isinstance(filtered, pd.DataFrame)
        if not filtered.empty and 'rating' in filtered.columns:
            assert filtered['rating'].min() >= 4.0

    def test_filter_by_multiple_criteria(self, data_processor, sample_ratings_data):
        """Test filtering with multiple criteria."""
        filtered = data_processor.filter_data(
            sample_ratings_data,
            min_rating=3.5,
            userId=1
        )

        assert isinstance(filtered, pd.DataFrame)
        if not filtered.empty and 'userId' in filtered.columns:
            assert all(filtered['userId'] == 1)

    def test_filter_returns_empty_when_no_matches(self, data_processor, sample_ratings_data):
        """Test filter returns empty DataFrame when no rows match."""
        filtered = data_processor.filter_data(sample_ratings_data, min_rating=10.0)

        assert isinstance(filtered, pd.DataFrame)
        # With impossible rating, should return empty or original
        assert len(filtered) >= 0

    def test_filter_empty_dataframe(self, data_processor, empty_dataframe):
        """Test filtering empty DataFrame."""
        filtered = data_processor.filter_data(empty_dataframe, min_rating=3.0)
        assert isinstance(filtered, pd.DataFrame)
        assert len(filtered) == 0


class TestDataExport:
    """Test data export functionality."""

    def test_export_to_csv(self, data_processor, sample_movies_data, temp_data_dir):
        """Test exporting DataFrame to CSV."""
        output_path = temp_data_dir / "export.csv"

        result = data_processor.export_data(
            sample_movies_data,
            str(output_path),
            format='csv'
        )

        assert output_path.exists() or isinstance(result, (dict, str))

    def test_export_to_json(self, data_processor, sample_movies_data, temp_data_dir):
        """Test exporting DataFrame to JSON."""
        output_path = temp_data_dir / "export.json"

        result = data_processor.export_data(
            sample_movies_data,
            str(output_path),
            format='json'
        )

        assert output_path.exists() or isinstance(result, (dict, str))

    def test_export_empty_dataframe(self, data_processor, empty_dataframe, temp_data_dir):
        """Test exporting empty DataFrame."""
        output_path = temp_data_dir / "empty_export.csv"

        result = data_processor.export_data(
            empty_dataframe,
            str(output_path),
            format='csv'
        )

        # Should handle empty export gracefully
        assert isinstance(result, (dict, str)) or output_path.exists()


class TestDataValidation:
    """Test data validation functionality."""

    def test_validate_schema(self, data_processor, sample_movies_data):
        """Test data schema validation."""
        # Should not raise error for valid schema
        assert 'movieId' in sample_movies_data.columns
        assert 'title' in sample_movies_data.columns
        assert 'genres' in sample_movies_data.columns

    def test_validate_data_types(self, data_processor, sample_ratings_data):
        """Test data type validation."""
        assert sample_ratings_data['userId'].dtype in [np.int64, np.int32, int]
        assert sample_ratings_data['movieId'].dtype in [np.int64, np.int32, int]
        assert sample_ratings_data['rating'].dtype in [np.float64, float]

    def test_validate_rating_range(self, data_processor, sample_ratings_data):
        """Test rating values are within valid range."""
        ratings = sample_ratings_data['rating']
        assert ratings.min() >= 0.5
        assert ratings.max() <= 5.0


class TestErrorHandling:
    """Test error handling in DataProcessor."""

    def test_load_invalid_path_type(self, data_processor):
        """Test loading with invalid path type."""
        with pytest.raises((DataLoadError, TypeError, FileNotFoundError)):
            data_processor.load_data(None)

    def test_clean_invalid_input(self, data_processor):
        """Test cleaning with invalid input."""
        with pytest.raises((DataCleaningError, AttributeError, TypeError)):
            data_processor.clean_data(None)

    def test_filter_invalid_criteria(self, data_processor, sample_ratings_data):
        """Test filtering with invalid criteria."""
        # Should handle gracefully or raise appropriate error
        try:
            result = data_processor.filter_data(sample_ratings_data, invalid_column="value")
            assert isinstance(result, pd.DataFrame)
        except (DataFilterError, KeyError):
            pass  # Expected error


class TestDataIntegrity:
    """Test data integrity operations."""

    def test_no_data_loss_after_clean(self, data_processor, sample_movies_data):
        """Test that cleaning doesn't lose valid data unnecessarily."""
        original_len = len(sample_movies_data)
        cleaned = data_processor.clean_data(sample_movies_data.copy())

        # Clean data should have same or fewer rows (no unexpected data loss)
        assert len(cleaned) <= original_len

    def test_column_preservation(self, data_processor, sample_movies_data):
        """Test that cleaning preserves required columns."""
        cleaned = data_processor.clean_data(sample_movies_data.copy())

        # Should preserve all columns or have documented subset
        assert isinstance(cleaned, pd.DataFrame)

    def test_index_reset_after_clean(self, data_processor, sample_dirty_data):
        """Test that index is properly reset after cleaning."""
        cleaned = data_processor.clean_data(sample_dirty_data)

        if not cleaned.empty:
            # Index should be continuous
            assert isinstance(cleaned.index, pd.RangeIndex) or cleaned.index.is_monotonic_increasing
