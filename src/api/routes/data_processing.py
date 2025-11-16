from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from src.services.data_processor import DataProcessor
from pathlib import Path
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
from src.models import (
    DataProcessResponse,
    LoadDataRequest,
    LoadDataResponse,
    CleanDataRequest,
    CleanDataResponse,
    AggregateStatsRequest,
    AggregateStatsResponse,
    FilterDataRequest,
    FilterDataResponse,
    ExportDataRequest,
    ExportDataResponse,
)

logger = get_logger("data_processing_api", "api")
router = APIRouter()
data_processor = DataProcessor()


@router.post(
    "/process_all_data",
    response_model=DataProcessResponse,
    summary="Process All Data",
    description="Convert raw data to CSV, load, clean, and analyze all datasets sequentially",
    tags=["Data Processing"],
)
async def process_all_data() -> DataProcessResponse:
    try:
        logger.info("=== Data Processing Pipeline Initiated ===")

        logger.info("Step 1: Converting raw data to CSV")
        converted_files: Dict[str, str] = data_processor.convert_raw_to_csv()
        logger.info(f"Dataset conversion completed: {list(converted_files.keys())}")

        movies_path: str = converted_files.get('movies')
        ratings_path: str = converted_files.get('ratings')
        tags_path: str = converted_files.get('tags')
        users_path: str = converted_files.get('users')

        movies_result: Dict[str, Any] = {}
        if movies_path:
            logger.info("Step 2: Processing movies data")
            movies_df = data_processor.load_data(movies_path)
            initial_movies_rows: int = len(movies_df)
            logger.info(f"Loaded {initial_movies_rows} movies")

            cleaned_movies_df = data_processor.clean_data(movies_df)
            final_movies_rows: int = len(cleaned_movies_df)
            logger.info(f"Cleaned movies data: {final_movies_rows} rows remaining")

            movies_stats: Dict[str, Any] = data_processor.aggregate_statistics(cleaned_movies_df)

            cleaned_movies_path: Path = Path(settings.data_processed_path) / "movies_cleaned.csv"
            cleaned_movies_df.to_csv(cleaned_movies_path, index=False)
            logger.info(f"Saved cleaned movies to {cleaned_movies_path}")

            movies_result = {
                "file_path": str(cleaned_movies_path),
                "initial_rows": initial_movies_rows,
                "final_rows": final_movies_rows,
                "rows_removed": initial_movies_rows - final_movies_rows,
                "statistics": movies_stats
            }

        ratings_result: Dict[str, Any] = {}
        if ratings_path:
            logger.info("Step 3: Processing ratings data")
            ratings_df = data_processor.load_data(ratings_path)
            initial_ratings_rows: int = len(ratings_df)
            logger.info(f"Loaded {initial_ratings_rows} ratings")

            cleaned_ratings_df = data_processor.clean_data(ratings_df)
            final_ratings_rows: int = len(cleaned_ratings_df)
            logger.info(f"Cleaned ratings data: {final_ratings_rows} rows remaining")

            ratings_stats: Dict[str, Any] = data_processor.aggregate_statistics(cleaned_ratings_df)

            cleaned_ratings_path: Path = Path(settings.data_processed_path) / "ratings_cleaned.csv"
            cleaned_ratings_df.to_csv(cleaned_ratings_path, index=False)
            logger.info(f"Saved cleaned ratings to {cleaned_ratings_path}")

            ratings_result = {
                "file_path": str(cleaned_ratings_path),
                "initial_rows": initial_ratings_rows,
                "final_rows": final_ratings_rows,
                "rows_removed": initial_ratings_rows - final_ratings_rows,
                "statistics": ratings_stats
            }

        tags_result: Dict[str, Any] = {}
        if tags_path:
            logger.info("Step 4: Processing tags data")
            tags_df = data_processor.load_data(tags_path)
            initial_tags_rows: int = len(tags_df)
            logger.info(f"Loaded {initial_tags_rows} tags")

            cleaned_tags_df = data_processor.clean_data(tags_df)
            final_tags_rows: int = len(cleaned_tags_df)
            logger.info(f"Cleaned tags data: {final_tags_rows} rows remaining")

            tags_stats: Dict[str, Any] = data_processor.aggregate_statistics(cleaned_tags_df)

            cleaned_tags_path: Path = Path(settings.data_processed_path) / "tags_cleaned.csv"
            cleaned_tags_df.to_csv(cleaned_tags_path, index=False)
            logger.info(f"Saved cleaned tags to {cleaned_tags_path}")

            tags_result = {
                "file_path": str(cleaned_tags_path),
                "initial_rows": initial_tags_rows,
                "final_rows": final_tags_rows,
                "rows_removed": initial_tags_rows - final_tags_rows,
                "statistics": tags_stats
            }

        users_result: Dict[str, Any] = {}
        if users_path:
            logger.info("Step 5: Processing users data")
            users_df = data_processor.load_data(users_path)
            initial_users_rows: int = len(users_df)
            logger.info(f"Loaded {initial_users_rows} users")

            cleaned_users_df = data_processor.clean_data(users_df)
            final_users_rows: int = len(cleaned_users_df)
            logger.info(f"Cleaned users data: {final_users_rows} rows remaining")

            users_stats: Dict[str, Any] = data_processor.aggregate_statistics(cleaned_users_df)

            cleaned_users_path: Path = Path(settings.data_processed_path) / "users_cleaned.csv"
            cleaned_users_df.to_csv(cleaned_users_path, index=False)
            logger.info(f"Saved cleaned users to {cleaned_users_path}")

            users_result = {
                "file_path": str(cleaned_users_path),
                "initial_rows": initial_users_rows,
                "final_rows": final_users_rows,
                "rows_removed": initial_users_rows - final_users_rows,
                "statistics": users_stats
            }

        logger.info("=== Data Processing Pipeline Completed Successfully ===")
        return DataProcessResponse(
            status="success",
            message="Data processing completed successfully",
            converted_files=converted_files,
            movies_result=movies_result,
            ratings_result=ratings_result,
            tags_result=tags_result,
            users_result=users_result
        )
    except DataLoadError as e:
        logger.error(f"Load error: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Load error: {str(e)}")
    except DataCleaningError as e:
        logger.error(f"Cleaning error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cleaning error: {str(e)}")
    except DataValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except DataAggregationError as e:
        logger.error(f"Aggregation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Aggregation error: {str(e)}")
    except DataProcessingError as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post(
    "/load_data",
    response_model=LoadDataResponse,
    summary="Load Data",
    description="Load data from raw folder, convert to CSV, and save to processed folder",
    tags=["Data Processing"],
)
async def load_data(request: LoadDataRequest) -> LoadDataResponse:
    try:
        logger.info(f"Load data endpoint called for dataset: {request.dataset}")

        # Handle "all" datasets
        if request.dataset.lower() == "all":
            logger.info("Loading all datasets")
            datasets = ["movies", "ratings", "tags", "users"]
            datasets_loaded = {}
            total_rows = 0

            for dataset_name in datasets:
                source_file: Path = Path(settings.data_raw_path) / f"{dataset_name}.dat"

                if source_file.exists():
                    logger.info(f"Loading {dataset_name} dataset")
                    df = data_processor.load_data(str(source_file))

                    output_file: Path = Path(settings.data_processed_path) / f"{dataset_name}.csv"
                    df.to_csv(output_file, index=False)

                    datasets_loaded[dataset_name] = {
                        "source_file": str(source_file),
                        "output_file": str(output_file),
                        "rows": len(df),
                        "columns": list(df.columns),
                        "status": "success"
                    }
                    total_rows += len(df)
                    logger.info(f"Loaded {dataset_name}: {len(df)} rows")
                else:
                    logger.warning(f"Dataset file not found: {source_file}")
                    datasets_loaded[dataset_name] = {
                        "status": "not_found",
                        "message": f"File {source_file} not found"
                    }

            successful_datasets = [k for k, v in datasets_loaded.items() if v.get("status") == "success"]

            return LoadDataResponse(
                status="success",
                message=f"Loaded {len(successful_datasets)} datasets: {', '.join(successful_datasets)}. Total rows: {total_rows}",
                datasets_loaded=datasets_loaded
            )

        # Handle single dataset
        source_file: Path = Path(settings.data_raw_path) / f"{request.dataset}.dat"
        if not source_file.exists():
            logger.error(f"Source file not found: {source_file}")
            raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset}' not found in raw folder")

        df = data_processor.load_data(str(source_file))
        logger.info(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")

        output_file: Path = Path(settings.data_processed_path) / f"{request.dataset}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Data saved to: {output_file}")

        # Limit data preview to first 100 rows to avoid timeout with large datasets
        preview_limit = 100
        preview_data = df.head(preview_limit).to_dict(orient='records')
        logger.info(f"Returning preview of first {preview_limit} rows out of {len(df)} total rows")

        return LoadDataResponse(
            status="success",
            message=f"Data loaded from raw folder and saved to processed folder (showing first {preview_limit} rows)",
            source_file=str(source_file),
            output_file=str(output_file),
            rows=len(df),
            columns=list(df.columns),
            data=preview_data
        )
    except DataLoadError as e:
        logger.error(f"Load error: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Load error: {str(e)}")
    except DataValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post(
    "/clean_data",
    response_model=CleanDataResponse,
    summary="Clean Data",
    description="Clean CSV data from processed folder and save cleaned version",
    tags=["Data Processing"],
)
async def clean_data(request: CleanDataRequest) -> CleanDataResponse:
    try:
        logger.info(f"Clean data endpoint called for dataset: {request.dataset}")

        # Handle "all" datasets
        if request.dataset.lower() == "all":
            logger.info("Cleaning all datasets")
            datasets = ["movies", "ratings", "tags", "users"]
            datasets_cleaned = {}
            total_initial_rows = 0
            total_final_rows = 0

            for dataset_name in datasets:
                source_file: Path = Path(settings.data_processed_path) / f"{dataset_name}.csv"

                if source_file.exists():
                    logger.info(f"Cleaning {dataset_name} dataset")
                    df = data_processor.load_data(str(source_file))
                    initial_rows = len(df)

                    cleaned_df = data_processor.clean_data(df)
                    final_rows = len(cleaned_df)

                    output_file: Path = Path(settings.data_processed_path) / f"{dataset_name}_cleaned.csv"
                    cleaned_df.to_csv(output_file, index=False)

                    datasets_cleaned[dataset_name] = {
                        "source_file": str(source_file),
                        "output_file": str(output_file),
                        "initial_rows": initial_rows,
                        "final_rows": final_rows,
                        "rows_removed": initial_rows - final_rows,
                        "columns": list(cleaned_df.columns),
                        "status": "success"
                    }
                    total_initial_rows += initial_rows
                    total_final_rows += final_rows
                    logger.info(f"Cleaned {dataset_name}: {initial_rows} -> {final_rows} rows")
                else:
                    logger.warning(f"Dataset file not found: {source_file}")
                    datasets_cleaned[dataset_name] = {
                        "status": "not_found",
                        "message": f"File {source_file} not found. Please run load_data first."
                    }

            successful_datasets = [k for k, v in datasets_cleaned.items() if v.get("status") == "success"]
            total_removed = total_initial_rows - total_final_rows

            return CleanDataResponse(
                status="success",
                message=f"Cleaned {len(successful_datasets)} datasets: {', '.join(successful_datasets)}. Removed {total_removed} rows total",
                datasets_cleaned=datasets_cleaned
            )

        # Handle single dataset
        source_file: Path = Path(settings.data_processed_path) / f"{request.dataset}.csv"
        if not source_file.exists():
            logger.error(f"Source file not found: {source_file}")
            raise HTTPException(status_code=404, detail=f"CSV file for dataset '{request.dataset}' not found. Please run load_data first.")

        df = data_processor.load_data(str(source_file))
        initial_rows = len(df)

        cleaned_df = data_processor.clean_data(df)
        final_rows = len(cleaned_df)

        output_file: Path = Path(settings.data_processed_path) / f"{request.dataset}_cleaned.csv"
        cleaned_df.to_csv(output_file, index=False)
        logger.info(f"Cleaned data saved to: {output_file}")

        # Limit data preview to first 100 rows to avoid timeout with large datasets
        preview_limit = 100
        preview_data = cleaned_df.head(preview_limit).to_dict(orient='records')
        logger.info(f"Data cleaned: {initial_rows} -> {final_rows} rows (returning preview of first {preview_limit} rows)")

        return CleanDataResponse(
            status="success",
            message=f"Data cleaned and saved successfully (showing first {preview_limit} rows)",
            source_file=str(source_file),
            output_file=str(output_file),
            initial_rows=initial_rows,
            final_rows=final_rows,
            rows_removed=initial_rows - final_rows,
            columns=list(cleaned_df.columns),
            data=preview_data
        )
    except DataLoadError as e:
        logger.error(f"Load error: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Load error: {str(e)}")
    except DataCleaningError as e:
        logger.error(f"Cleaning error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cleaning error: {str(e)}")
    except DataValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post(
    "/aggregate_statistics",
    response_model=AggregateStatsResponse,
    summary="Aggregate Statistics",
    description="Calculate comprehensive statistics from cleaned dataset. Use 'all' to get combined statistics for all datasets.",
    tags=["Data Processing"],
)
async def aggregate_statistics(request: AggregateStatsRequest) -> AggregateStatsResponse:
    try:
        logger.info(f"Aggregate statistics endpoint called for dataset: {request.dataset}")

        if request.dataset.lower() == "all":
            logger.info("Processing combined statistics for all datasets")

            datasets = ["movies", "ratings", "tags", "users"]
            combined_stats: Dict[str, Any] = {
                "dataset_type": "combined",
                "datasets_included": [],
                "total_datasets": 0,
                "total_records_across_all_datasets": 0,
                "individual_dataset_statistics": {}
            }

            for dataset_name in datasets:
                cleaned_file: Path = Path(settings.data_processed_path) / f"{dataset_name}_cleaned.csv"
                if cleaned_file.exists():
                    logger.info(f"Loading {dataset_name} for combined statistics")
                    df = data_processor.load_data(str(cleaned_file))
                    dataset_stats = data_processor.aggregate_statistics(df)

                    combined_stats["datasets_included"].append(dataset_name)
                    combined_stats["total_datasets"] += 1
                    combined_stats["total_records_across_all_datasets"] += len(df)
                    combined_stats["individual_dataset_statistics"][dataset_name] = dataset_stats
                    logger.info(f"Added {dataset_name} statistics: {len(df)} rows")
                else:
                    logger.warning(f"Cleaned file not found for {dataset_name}, skipping")

            if combined_stats["total_datasets"] == 0:
                raise HTTPException(status_code=404, detail="No cleaned datasets found. Please run clean_data for at least one dataset first.")

            logger.info(f"Combined statistics completed for {combined_stats['total_datasets']} datasets with {combined_stats['total_records_across_all_datasets']} total records")
            return AggregateStatsResponse(
                status="success",
                message=f"Combined statistics aggregated successfully for {combined_stats['total_datasets']} datasets",
                statistics=combined_stats
            )
        else:
            cleaned_file: Path = Path(settings.data_processed_path) / f"{request.dataset}_cleaned.csv"
            if not cleaned_file.exists():
                logger.error(f"Cleaned file not found: {cleaned_file}")
                raise HTTPException(status_code=404, detail=f"Cleaned file for dataset '{request.dataset}' not found. Please run clean_data first.")

            df = data_processor.load_data(str(cleaned_file))
            stats = data_processor.aggregate_statistics(df)
            logger.info(f"Statistics aggregated for {len(df)} rows")
            return AggregateStatsResponse(
                status="success",
                message="Statistics aggregated successfully",
                statistics=stats
            )
    except DataLoadError as e:
        logger.error(f"Load error: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Load error: {str(e)}")
    except DataAggregationError as e:
        logger.error(f"Aggregation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Aggregation error: {str(e)}")
    except DataValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post(
    "/filter_data",
    response_model=FilterDataResponse,
    summary="Filter Data",
    description="Apply various filters to the cleaned dataset. Use 'all' to filter across all datasets.",
    tags=["Data Processing"],
)
async def filter_data(request: FilterDataRequest) -> FilterDataResponse:
    try:
        logger.info(f"Filter data endpoint called for dataset: {request.dataset}")

        filters = {}
        if request.min_rating is not None and request.min_rating > 0:
            filters['min_rating'] = request.min_rating
        if request.max_rating is not None and request.max_rating > 0:
            filters['max_rating'] = request.max_rating
        if request.user_id is not None and request.user_id > 0:
            filters['user_id'] = request.user_id
        if request.movie_id is not None and request.movie_id > 0:
            filters['movie_id'] = request.movie_id
        if request.genres is not None and request.genres.strip():
            filters['genres'] = request.genres
        if request.min_year is not None and request.min_year > 0:
            filters['min_year'] = request.min_year
        if request.max_year is not None and request.max_year > 0:
            filters['max_year'] = request.max_year
        if request.limit is not None and request.limit > 0:
            filters['limit'] = request.limit

        if request.dataset.lower() == "all":
            logger.info("Processing combined filtering for all datasets")

            datasets = ["movies", "ratings", "tags", "users"]
            combined_results: Dict[str, Any] = {
                "dataset_type": "combined",
                "datasets_included": [],
                "total_datasets": 0,
                "total_original_rows": 0,
                "total_filtered_rows": 0,
                "individual_dataset_results": {}
            }

            for dataset_name in datasets:
                cleaned_file: Path = Path(settings.data_processed_path) / f"{dataset_name}_cleaned.csv"
                if cleaned_file.exists():
                    logger.info(f"Filtering {dataset_name} dataset")
                    df = data_processor.load_data(str(cleaned_file))
                    original_rows = len(df)

                    filtered_df = data_processor.filter_data(df, **filters)
                    filtered_rows = len(filtered_df)

                    # Limit preview for each dataset
                    preview_limit = 100
                    preview_data = filtered_df.head(preview_limit).to_dict(orient='records')

                    combined_results["datasets_included"].append(dataset_name)
                    combined_results["total_datasets"] += 1
                    combined_results["total_original_rows"] += original_rows
                    combined_results["total_filtered_rows"] += filtered_rows

                    combined_results["individual_dataset_results"][dataset_name] = {
                        "original_rows": original_rows,
                        "filtered_rows": filtered_rows,
                        "data": preview_data,
                        "preview_note": f"Showing first {min(preview_limit, filtered_rows)} of {filtered_rows} rows"
                    }
                    logger.info(f"Filtered {dataset_name}: {original_rows} -> {filtered_rows} rows")
                else:
                    logger.warning(f"Cleaned file not found for {dataset_name}, skipping")

            if combined_results["total_datasets"] == 0:
                raise HTTPException(status_code=404, detail="No cleaned datasets found. Please run clean_data for at least one dataset first.")

            logger.info(f"Combined filtering completed for {combined_results['total_datasets']} datasets")
            return FilterDataResponse(
                status="success",
                message=f"Data filtered successfully across {combined_results['total_datasets']} datasets",
                original_rows=combined_results["total_original_rows"],
                filtered_rows=combined_results["total_filtered_rows"],
                filters_applied=filters,
                data=combined_results
            )
        else:
            cleaned_file: Path = Path(settings.data_processed_path) / f"{request.dataset}_cleaned.csv"
            if not cleaned_file.exists():
                logger.error(f"Cleaned file not found: {cleaned_file}")
                raise HTTPException(status_code=404, detail=f"Cleaned file for dataset '{request.dataset}' not found. Please run clean_data first.")

            df = data_processor.load_data(str(cleaned_file))
            filtered_df = data_processor.filter_data(df, **filters)

            # Limit data preview to first 1000 rows to avoid timeout with large result sets
            preview_limit = 1000
            preview_data = filtered_df.head(preview_limit).to_dict(orient='records')
            logger.info(f"Data filtered: {len(df)} -> {len(filtered_df)} rows (returning preview of first {min(preview_limit, len(filtered_df))} rows)")

            return FilterDataResponse(
                status="success",
                message=f"Data filtered successfully (showing first {min(preview_limit, len(filtered_df))} of {len(filtered_df)} rows)",
                original_rows=len(df),
                filtered_rows=len(filtered_df),
                filters_applied=filters,
                data=preview_data
            )
    except DataLoadError as e:
        logger.error(f"Load error: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Load error: {str(e)}")
    except DataFilterError as e:
        logger.error(f"Filter error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Filter error: {str(e)}")
    except DataValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post(
    "/export_data",
    response_model=ExportDataResponse,
    summary="Export Data",
    description="Export cleaned or filtered data to CSV or JSON format",
    tags=["Data Processing"],
)
async def export_data(request: ExportDataRequest) -> ExportDataResponse:
    """
    Export dataset to CSV or JSON format.

    Supports:
    - CSV export with optional index inclusion
    - JSON export with multiple orientations (records, index, columns, values, split, table)
    - Automatic file naming with timestamps
    - Export from cleaned datasets
    """
    try:
        logger.info(f"Export request received for dataset '{request.dataset}' in '{request.format}' format")

        # Validate format
        if request.format.lower() not in ['csv', 'json']:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}. Use 'csv' or 'json'")

        # Load the cleaned dataset
        cleaned_file: Path = Path(settings.data_processed_path) / f"{request.dataset}_cleaned.csv"
        if not cleaned_file.exists():
            logger.error(f"Cleaned file not found: {cleaned_file}")
            raise HTTPException(
                status_code=404,
                detail=f"Cleaned file for dataset '{request.dataset}' not found. Please run clean_data first."
            )

        df = data_processor.load_data(str(cleaned_file))
        rows_count = len(df)

        # Generate output file name if not provided
        if request.file_name:
            output_file_name = request.file_name
            if not output_file_name.endswith(f'.{request.format}'):
                output_file_name += f'.{request.format}'
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file_name = f"{request.dataset}_export_{timestamp}.{request.format}"

        output_path = Path(settings.data_processed_path) / output_file_name

        # Export based on format
        if request.format.lower() == 'csv':
            logger.info(f"Exporting to CSV with index={request.include_index}")
            file_path = data_processor.export_to_csv(
                df,
                str(output_path),
                index=request.include_index
            )
        else:  # json
            logger.info(f"Exporting to JSON with orient={request.orient}")
            file_path = data_processor.export_to_json(
                df,
                str(output_path),
                orient=request.orient,
                indent=2
            )

        # Get file size
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)

        logger.info(f"Export completed successfully: {file_path} ({file_size_mb:.2f} MB)")
        return ExportDataResponse(
            status="success",
            message=f"Data exported successfully to {request.format.upper()} format",
            file_path=file_path,
            file_size_mb=file_size_mb,
            rows_exported=rows_count,
            format=request.format.lower()
        )

    except HTTPException:
        # Re-raise HTTPException as-is (already has correct status code)
        raise
    except DataLoadError as e:
        logger.error(f"Load error: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Load error: {str(e)}")
    except DataProcessingError as e:
        logger.error(f"Export error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")
    except DataValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
