import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import zipfile
import urllib.request
import shutil
from abc import ABC, abstractmethod
from src.core.config import settings
from src.exceptions import (
    DataDownloadError,
    DataLoadError,
    DataCleaningError,
    DataValidationError,
    DataAggregationError,
    DataFilterError,
)
from src.utils.logger import get_logger

logger = get_logger("data_processor", "data_processing")


class BaseProcessor(ABC):
    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class DataProcessor(BaseProcessor):
    def __init__(self) -> None:
        logger.info("Initializing DataProcessor")
        self.data_raw_path: Path = Path(settings.data_raw_path)
        self.data_processed_path: Path = Path(settings.data_processed_path)
        self.data_temp_path: Path = Path(settings.data_temp_path)
        self.data_delimiter: str = settings.data_delimiter
        self.chunk_size: int = 100000

        self.data_raw_path.mkdir(parents=True, exist_ok=True)
        self.data_processed_path.mkdir(parents=True, exist_ok=True)
        self.data_temp_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"DataProcessor initialized with paths - raw: {self.data_raw_path}, processed: {self.data_processed_path}")

    def convert_raw_to_csv(self) -> Dict[str, str]:
        try:
            logger.info("Starting raw data conversion to CSV")
            converted_files: Dict[str, str] = {}

            logger.info("Converting movies data")
            self._convert_file(
                self.data_raw_path,
                ["movies.dat", "movies.csv"],
                "movies.csv",
                ['movieId', 'title', 'genres'],
                converted_files,
                'movies'
            )

            logger.info("Converting ratings data")
            self._convert_file(
                self.data_raw_path,
                ["ratings.dat", "ratings.csv"],
                "ratings.csv",
                ['userId', 'movieId', 'rating', 'timestamp'],
                converted_files,
                'ratings'
            )

            logger.info("Converting tags data")
            self._convert_file(
                self.data_raw_path,
                ["tags.dat", "tags.csv"],
                "tags.csv",
                ['userId', 'movieId', 'tag', 'timestamp'],
                converted_files,
                'tags'
            )

            logger.info(f"Raw data conversion completed successfully. Converted files: {converted_files}")
            return converted_files

        except Exception as e:
            logger.error(f"Unexpected error during conversion: {str(e)}")
            raise DataProcessingError("Unexpected error during conversion", str(e))

    def _find_extracted_folder(self) -> Optional[Path]:
        for item in self.data_temp_path.iterdir():
            if item.is_dir() and item.name != '.gitkeep':
                return item
        return None

    def _convert_file(
        self,
        directory: Path,
        filenames: List[str],
        output_name: str,
        columns: List[str],
        result_dict: Dict[str, str],
        result_key: str
    ) -> None:
        file_path: Optional[Path] = self._find_file(directory, filenames)
        if file_path:
            df: pd.DataFrame = self._read_file(file_path)
            if not all(col in df.columns for col in ['movieId', 'userId', 'title', 'rating', 'tag', 'genres', 'timestamp']):
                df.columns = columns
            output_path: Path = self.data_processed_path / output_name

            if len(df) > self.chunk_size:
                df.to_csv(output_path, index=False, chunksize=self.chunk_size)
            else:
                df.to_csv(output_path, index=False)

            result_dict[result_key] = str(output_path)

    def _find_file(self, directory: Path, filenames: List[str]) -> Optional[Path]:
        for filename in filenames:
            file_path: Path = directory / filename
            if file_path.exists():
                return file_path
        return None

    def _read_file(self, file_path: Path) -> pd.DataFrame:
        try:
            if file_path.suffix == '.dat':
                return pd.read_csv(
                    file_path,
                    sep=self.data_delimiter,
                    encoding='latin-1',
                    engine='python'
                )
            else:
                return pd.read_csv(file_path, encoding='latin-1', low_memory=False)
        except Exception as e:
            raise DataLoadError(f"Failed to read file: {file_path.name}", str(e))

    def load_data(self, file_path: str) -> pd.DataFrame:
        try:
            logger.info(f"Loading data from {file_path}")
            path: Path = Path(file_path)

            if not path.exists():
                logger.error(f"File not found: {file_path}")
                raise DataLoadError(f"File not found: {file_path}")

            if path.suffix == '.dat':
                logger.debug(f"Reading .dat file with delimiter '{self.data_delimiter}'")
                df: pd.DataFrame = pd.read_csv(
                    path,
                    sep=self.data_delimiter,
                    encoding='latin-1',
                    engine='python'
                )
            elif path.suffix == '.csv':
                logger.debug("Reading .csv file")
                df = pd.read_csv(path, encoding='latin-1', low_memory=False)
            else:
                logger.error(f"Unsupported file format: {path.suffix}")
                raise DataLoadError(f"Unsupported file format: {path.suffix}")

            if df.empty:
                logger.error(f"File is empty: {file_path}")
                raise DataValidationError(f"File is empty: {file_path}")

            logger.info(f"Data loaded successfully. Rows: {len(df)}, Columns: {list(df.columns)}")
            return df

        except pd.errors.EmptyDataError as e:
            logger.error(f"File contains no data: {str(e)}")
            raise DataLoadError("File contains no data", str(e))
        except pd.errors.ParserError as e:
            logger.error(f"Failed to parse file: {str(e)}")
            raise DataLoadError("Failed to parse file", str(e))
        except Exception as e:
            logger.error(f"Unexpected error while loading data: {str(e)}")
            raise DataLoadError("Unexpected error while loading data", str(e))

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info(f"Starting data cleaning. Initial rows: {len(df)}")
            if df.empty:
                logger.error("Cannot clean empty DataFrame")
                raise DataValidationError("Cannot clean empty DataFrame")

            cleaned_df: pd.DataFrame = df.copy()
            initial_rows = len(cleaned_df)

            logger.debug("Removing duplicates")
            cleaned_df = cleaned_df.drop_duplicates()
            duplicates_removed = initial_rows - len(cleaned_df)
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate rows")

            logger.debug("Handling missing values")
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].fillna('')
                else:
                    if col in ['rating', 'movieId', 'userId']:
                        before_drop = len(cleaned_df)
                        cleaned_df = cleaned_df.dropna(subset=[col])
                        dropped = before_drop - len(cleaned_df)
                        if dropped > 0:
                            logger.info(f"Dropped {dropped} rows with missing {col}")
                    else:
                        median_value: float = cleaned_df[col].median()
                        cleaned_df[col] = cleaned_df[col].fillna(median_value)

            if 'rating' in cleaned_df.columns:
                logger.debug("Validating rating values")
                before_filter = len(cleaned_df)
                cleaned_df = cleaned_df[
                    (cleaned_df['rating'] >= 0.5) & (cleaned_df['rating'] <= 5.0)
                ]
                invalid_ratings = before_filter - len(cleaned_df)
                if invalid_ratings > 0:
                    logger.info(f"Removed {invalid_ratings} rows with invalid ratings")

            if 'title' in cleaned_df.columns:
                logger.debug("Removing empty titles")
                before_filter = len(cleaned_df)
                cleaned_df = cleaned_df[cleaned_df['title'].str.strip() != '']
                empty_titles = before_filter - len(cleaned_df)
                if empty_titles > 0:
                    logger.info(f"Removed {empty_titles} rows with empty titles")

            cleaned_df = cleaned_df.reset_index(drop=True)

            logger.info(f"Data cleaning completed. Final rows: {len(cleaned_df)}, Rows removed: {initial_rows - len(cleaned_df)}")
            return cleaned_df

        except KeyError as e:
            logger.error(f"Missing expected column: {str(e)}")
            raise DataCleaningError("Missing expected column", str(e))
        except Exception as e:
            logger.error(f"Unexpected error during cleaning: {str(e)}")
            raise DataCleaningError("Unexpected error during cleaning", str(e))

    def aggregate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            logger.info("Starting statistics aggregation")
            if df.empty:
                logger.error("Cannot aggregate statistics on empty DataFrame")
                raise DataValidationError("Cannot aggregate statistics on empty DataFrame")

            stats: Dict[str, Any] = {
                'total_records': int(len(df)),
                'columns': list(df.columns),
                'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'missing_values': {col: int(count) for col, count in df.isnull().sum().items()},
                'memory_usage_mb': float(df.memory_usage(deep=True).sum() / (1024 * 1024))
            }
            logger.debug(f"Basic stats calculated. Total records: {stats['total_records']}, Memory: {stats['memory_usage_mb']:.2f}MB")

            numeric_cols: pd.Index = df.select_dtypes(include=[np.number]).columns
            logger.debug(f"Calculating statistics for {len(numeric_cols)} numeric columns")
            for col in numeric_cols:
                stats[f'{col}_statistics'] = {
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'count': int(df[col].count())
                }

            if 'rating' in df.columns:
                logger.debug("Calculating rating distribution")
                rating_dist: pd.Series = df['rating'].value_counts().sort_index()
                stats['rating_distribution'] = {
                    str(k): int(v) for k, v in rating_dist.items()
                }

            if 'genres' in df.columns:
                logger.debug("Analyzing genres")
                all_genres: List[str] = []
                for genres in df['genres'].dropna():
                    if isinstance(genres, str):
                        all_genres.extend(genres.split('|'))
                genre_counts: pd.Series = pd.Series(all_genres).value_counts().head(10)
                stats['top_genres'] = {k: int(v) for k, v in genre_counts.items()}
                logger.debug(f"Found {len(all_genres)} total genre entries")

            if 'userId' in df.columns:
                unique_users = int(df['userId'].nunique())
                stats['unique_users'] = unique_users
                logger.debug(f"Unique users: {unique_users}")

            if 'movieId' in df.columns:
                unique_movies = int(df['movieId'].nunique())
                stats['unique_movies'] = unique_movies
                logger.debug(f"Unique movies: {unique_movies}")

            logger.info("Statistics aggregation completed successfully")
            return stats

        except Exception as e:
            logger.error(f"Failed to aggregate statistics: {str(e)}")
            raise DataAggregationError("Failed to aggregate statistics", str(e))

    def filter_data(self, df: pd.DataFrame, **filters: Any) -> pd.DataFrame:
        try:
            logger.info(f"Starting data filtering with filters: {filters}")
            if df.empty:
                logger.error("Cannot filter empty DataFrame")
                raise DataValidationError("Cannot filter empty DataFrame")

            filtered_df: pd.DataFrame = df.copy()
            initial_rows = len(filtered_df)

            if 'min_rating' in filters and 'rating' in filtered_df.columns:
                min_rating: float = float(filters['min_rating'])
                filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
                logger.debug(f"Applied min_rating filter: {min_rating}, Rows: {len(filtered_df)}")

            if 'max_rating' in filters and 'rating' in filtered_df.columns:
                max_rating: float = float(filters['max_rating'])
                filtered_df = filtered_df[filtered_df['rating'] <= max_rating]
                logger.debug(f"Applied max_rating filter: {max_rating}, Rows: {len(filtered_df)}")

            if 'user_id' in filters and 'userId' in filtered_df.columns:
                user_id: int = int(filters['user_id'])
                filtered_df = filtered_df[filtered_df['userId'] == user_id]
                logger.debug(f"Applied user_id filter: {user_id}, Rows: {len(filtered_df)}")

            if 'movie_id' in filters and 'movieId' in filtered_df.columns:
                movie_id: int = int(filters['movie_id'])
                filtered_df = filtered_df[filtered_df['movieId'] == movie_id]
                logger.debug(f"Applied movie_id filter: {movie_id}, Rows: {len(filtered_df)}")

            if 'genres' in filters and 'genres' in filtered_df.columns:
                genre_filter: Union[str, List[str]] = filters['genres']
                if isinstance(genre_filter, str):
                    filtered_df = filtered_df[
                        filtered_df['genres'].str.contains(genre_filter, case=False, na=False)
                    ]
                    logger.debug(f"Applied genre filter: {genre_filter}, Rows: {len(filtered_df)}")
                elif isinstance(genre_filter, list):
                    pattern: str = '|'.join(genre_filter)
                    filtered_df = filtered_df[
                        filtered_df['genres'].str.contains(pattern, case=False, na=False)
                    ]
                    logger.debug(f"Applied genres filter: {genre_filter}, Rows: {len(filtered_df)}")

            if 'min_year' in filters and 'title' in filtered_df.columns:
                min_year: int = int(filters['min_year'])
                filtered_df = self._filter_by_year(filtered_df, min_year=min_year)
                logger.debug(f"Applied min_year filter: {min_year}, Rows: {len(filtered_df)}")

            if 'max_year' in filters and 'title' in filtered_df.columns:
                max_year: int = int(filters['max_year'])
                filtered_df = self._filter_by_year(filtered_df, max_year=max_year)
                logger.debug(f"Applied max_year filter: {max_year}, Rows: {len(filtered_df)}")

            if 'limit' in filters:
                limit: int = int(filters['limit'])
                filtered_df = filtered_df.head(limit)
                logger.debug(f"Applied limit: {limit}")

            logger.info(f"Filtering completed. Initial rows: {initial_rows}, Final rows: {len(filtered_df)}, Filtered out: {initial_rows - len(filtered_df)}")
            return filtered_df

        except (ValueError, TypeError) as e:
            logger.error(f"Invalid filter parameter: {str(e)}")
            raise DataFilterError("Invalid filter parameter", str(e))
        except Exception as e:
            logger.error(f"Unexpected error during filtering: {str(e)}")
            raise DataFilterError("Unexpected error during filtering", str(e))

    def _filter_by_year(
        self,
        df: pd.DataFrame,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None
    ) -> pd.DataFrame:
        df_copy: pd.DataFrame = df.copy()

        df_copy['year'] = df_copy['title'].str.extract(r'\((\d{4})\)$')
        df_copy['year'] = pd.to_numeric(df_copy['year'], errors='coerce')

        if min_year is not None:
            df_copy = df_copy[df_copy['year'] >= min_year]

        if max_year is not None:
            df_copy = df_copy[df_copy['year'] <= max_year]

        df_copy = df_copy.drop(columns=['year'])

        return df_copy
