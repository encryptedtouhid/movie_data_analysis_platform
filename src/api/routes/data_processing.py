from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from pydantic import BaseModel
from src.services.data_processor import DataProcessor
from pathlib import Path
from src.core.config import settings
from src.exceptions import (
    DataProcessingError,
    DataDownloadError,
    DataLoadError,
    DataCleaningError,
    DataValidationError,
    DataAggregationError,
)
from src.utils.logger import get_logger

logger = get_logger("data_processing_api", "api")
router = APIRouter()
data_processor = DataProcessor()


class DataProcessResponse(BaseModel):
    status: str
    message: str
    download_result: Dict[str, Any]
    movies_result: Dict[str, Any]
    ratings_result: Dict[str, Any]
    tags_result: Dict[str, Any]


@router.post(
    "/process_all_data",
    response_model=DataProcessResponse,
    summary="Process All Data",
    description="Download, load, clean, and analyze all datasets sequentially",
    tags=["Data Processing"],
)
async def process_all_data() -> DataProcessResponse:
    try:
        logger.info("=== Data Processing Pipeline Initiated ===")

        logger.info("Step 1: Downloading and converting dataset")
        converted_files: Dict[str, str] = data_processor.download_and_convert_dataset()
        logger.info(f"Dataset conversion completed: {list(converted_files.keys())}")

        movies_path: str = converted_files.get('movies')
        ratings_path: str = converted_files.get('ratings')
        tags_path: str = converted_files.get('tags')

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

        logger.info("=== Data Processing Pipeline Completed Successfully ===")
        return DataProcessResponse(
            status="success",
            message="Data processing completed successfully",
            download_result=converted_files,
            movies_result=movies_result,
            ratings_result=ratings_result,
            tags_result=tags_result
        )
    except DataDownloadError as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download error: {str(e)}")
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
