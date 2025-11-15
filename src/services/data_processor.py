import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from abc import ABC, abstractmethod
from src.core.config import settings
from src.exceptions import (
    DataProcessingError,
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
        self.data_delimiter: str = settings.data_delimiter
        self.chunk_size: int = 100000

        self.data_raw_path.mkdir(parents=True, exist_ok=True)
        self.data_processed_path.mkdir(parents=True, exist_ok=True)
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

            logger.info("Converting users data")
            self._convert_file(
                self.data_raw_path,
                ["users.dat", "users.csv"],
                "users.csv",
                ['userId', 'gender', 'age', 'occupation', 'zipCode'],
                converted_files,
                'users'
            )

            logger.info(f"Raw data conversion completed successfully. Converted files: {converted_files}")
            return converted_files

        except Exception as e:
            logger.error(f"Unexpected error during conversion: {str(e)}")
            raise DataProcessingError("Unexpected error during conversion", str(e))

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
            df: pd.DataFrame = self._read_file(file_path, columns)
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

    def _read_file(self, file_path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
        try:
            if file_path.suffix == '.dat':
                if columns:
                    return pd.read_csv(
                        file_path,
                        sep=self.data_delimiter,
                        encoding='latin-1',
                        engine='python',
                        header=None,
                        names=columns
                    )
                else:
                    return pd.read_csv(
                        file_path,
                        sep=self.data_delimiter,
                        encoding='latin-1',
                        engine='python',
                        header=None
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
                columns: Optional[List[str]] = None
                if 'movies' in path.name.lower():
                    columns = ['movieId', 'title', 'genres']
                elif 'ratings' in path.name.lower():
                    columns = ['userId', 'movieId', 'rating', 'timestamp']
                elif 'tags' in path.name.lower():
                    columns = ['userId', 'movieId', 'tag', 'timestamp']
                elif 'users' in path.name.lower():
                    columns = ['userId', 'gender', 'age', 'occupation', 'zipCode']

                if columns:
                    df: pd.DataFrame = pd.read_csv(
                        path,
                        sep=self.data_delimiter,
                        encoding='latin-1',
                        engine='python',
                        header=None,
                        names=columns
                    )
                else:
                    df = pd.read_csv(
                        path,
                        sep=self.data_delimiter,
                        encoding='latin-1',
                        engine='python',
                        header=None
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
            logger.info("Starting comprehensive statistics aggregation")
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

            logger.debug("Calculating Total Counts")
            total_counts: Dict[str, int] = {}
            if 'userId' in df.columns:
                total_counts['total_users'] = int(df['userId'].nunique())
            if 'movieId' in df.columns:
                total_counts['total_movies'] = int(df['movieId'].nunique())
            if 'rating' in df.columns:
                total_counts['total_ratings'] = int(len(df[df['rating'].notna()]))
            if total_counts:
                stats['total_counts'] = total_counts
                logger.debug(f"Total counts: {total_counts}")

            if 'userId' in df.columns and 'movieId' in df.columns and 'rating' in df.columns:
                logger.debug("Calculating Sparsity")
                num_users = df['userId'].nunique()
                num_movies = df['movieId'].nunique()
                num_ratings = len(df)
                possible_ratings = num_users * num_movies
                sparsity = 1 - (num_ratings / possible_ratings) if possible_ratings > 0 else 0
                stats['sparsity'] = {
                    'value': float(sparsity),
                    'percentage': float(sparsity * 100),
                    'description': f"{sparsity * 100:.2f}% of possible ratings are missing",
                    'total_possible_ratings': int(possible_ratings),
                    'actual_ratings': int(num_ratings),
                    'missing_ratings': int(possible_ratings - num_ratings)
                }
                logger.debug(f"Sparsity: {sparsity * 100:.2f}%")

            if 'rating' in df.columns:
                logger.debug("Calculating Average Ratings")
                average_ratings: Dict[str, Any] = {
                    'overall_average_rating': float(df['rating'].mean()),
                    'overall_median_rating': float(df['rating'].median())
                }

                if 'movieId' in df.columns:
                    movie_avg = df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
                    movie_avg.columns = ['movieId', 'average_rating', 'rating_count']
                    top_rated = movie_avg.nlargest(10, 'average_rating')
                    most_rated = movie_avg.nlargest(10, 'rating_count')

                    average_ratings['average_rating_per_movie'] = {
                        'mean': float(movie_avg['average_rating'].mean()),
                        'median': float(movie_avg['average_rating'].median()),
                        'std': float(movie_avg['average_rating'].std()),
                        'min': float(movie_avg['average_rating'].min()),
                        'max': float(movie_avg['average_rating'].max())
                    }
                    average_ratings['top_10_highest_rated_movies'] = top_rated.to_dict(orient='records')
                    average_ratings['top_10_most_rated_movies'] = most_rated.to_dict(orient='records')

                if 'userId' in df.columns:
                    user_avg = df.groupby('userId')['rating'].agg(['mean', 'count']).reset_index()
                    user_avg.columns = ['userId', 'average_rating', 'rating_count']

                    average_ratings['average_rating_per_user'] = {
                        'mean': float(user_avg['average_rating'].mean()),
                        'median': float(user_avg['average_rating'].median()),
                        'std': float(user_avg['average_rating'].std()),
                        'min': float(user_avg['average_rating'].min()),
                        'max': float(user_avg['average_rating'].max())
                    }
                    average_ratings['top_10_most_active_users'] = user_avg.nlargest(10, 'rating_count').to_dict(orient='records')

                stats['average_ratings'] = average_ratings
                logger.debug(f"Average ratings calculated. Overall: {average_ratings['overall_average_rating']:.2f}")

            if 'rating' in df.columns:
                logger.debug("Calculating Rating Distribution")
                rating_dist: pd.Series = df['rating'].value_counts().sort_index()
                total_ratings_count = len(df['rating'].dropna())

                rating_distribution: Dict[str, Any] = {
                    'distribution': {str(k): int(v) for k, v in rating_dist.items()},
                    'distribution_percentage': {
                        str(k): float((v / total_ratings_count) * 100)
                        for k, v in rating_dist.items()
                    },
                    'most_common_rating': float(rating_dist.idxmax()),
                    'least_common_rating': float(rating_dist.idxmin())
                }
                stats['rating_distribution'] = rating_distribution
                logger.debug(f"Rating distribution: Most common rating = {rating_distribution['most_common_rating']}")

            if 'genres' in df.columns:
                logger.debug("Performing Genre Analysis")
                all_genres: List[str] = []
                genre_movie_map: Dict[str, List[int]] = {}

                for idx, row in df.iterrows():
                    genres = row.get('genres')
                    if pd.notna(genres) and isinstance(genres, str):
                        genre_list = genres.split('|')
                        all_genres.extend(genre_list)

                        if 'movieId' in df.columns:
                            movie_id = row.get('movieId')
                            for genre in genre_list:
                                if genre not in genre_movie_map:
                                    genre_movie_map[genre] = []
                                genre_movie_map[genre].append(movie_id)

                genre_counts: pd.Series = pd.Series(all_genres).value_counts()

                genre_analysis: Dict[str, Any] = {
                    'total_genres': len(genre_counts),
                    'total_genre_tags': len(all_genres),
                    'movies_per_genre': {k: int(v) for k, v in genre_counts.items()},
                    'top_10_genres': {k: int(v) for k, v in genre_counts.head(10).items()}
                }
                stats['genre_analysis'] = genre_analysis
                logger.debug(f"Genre analysis: {len(genre_counts)} unique genres found")

            if 'timestamp' in df.columns:
                logger.debug("Analyzing Temporal Trends")
                df_temp = df.copy()
                df_temp['datetime'] = pd.to_datetime(df_temp['timestamp'], unit='s', errors='coerce')
                df_temp['year'] = df_temp['datetime'].dt.year
                df_temp['month'] = df_temp['datetime'].dt.month
                df_temp['year_month'] = df_temp['datetime'].dt.to_period('M').astype(str)

                temporal_trends: Dict[str, Any] = {
                    'date_range': {
                        'earliest': str(df_temp['datetime'].min()),
                        'latest': str(df_temp['datetime'].max()),
                        'span_days': int((df_temp['datetime'].max() - df_temp['datetime'].min()).days)
                    }
                }

                if 'rating' in df.columns:
                    ratings_by_year = df_temp.groupby('year')['rating'].agg(['mean', 'count']).reset_index()
                    ratings_by_year.columns = ['year', 'average_rating', 'rating_count']
                    ratings_by_year = ratings_by_year.dropna()

                    temporal_trends['ratings_by_year'] = ratings_by_year.to_dict(orient='records')

                    ratings_by_month = df_temp.groupby('year_month')['rating'].agg(['mean', 'count']).reset_index()
                    ratings_by_month.columns = ['year_month', 'average_rating', 'rating_count']
                    ratings_by_month = ratings_by_month.dropna().tail(12)

                    temporal_trends['ratings_by_month_last_12'] = ratings_by_month.to_dict(orient='records')

                activity_by_year = df_temp.groupby('year').size().reset_index(name='activity_count')
                activity_by_year = activity_by_year.dropna()
                temporal_trends['activity_by_year'] = activity_by_year.to_dict(orient='records')

                stats['temporal_trends'] = temporal_trends
                logger.debug(f"Temporal trends: Data spans {temporal_trends['date_range']['span_days']} days")

            numeric_cols: pd.Index = df.select_dtypes(include=[np.number]).columns
            logger.debug(f"Calculating statistics for {len(numeric_cols)} numeric columns")
            for col in numeric_cols:
                if col not in ['timestamp']:
                    stats[f'{col}_statistics'] = {
                        'mean': float(df[col].mean()),
                        'median': float(df[col].median()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'count': int(df[col].count())
                    }

            logger.info("Comprehensive statistics aggregation completed successfully")
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
