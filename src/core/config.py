import os
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseModel):
    app_name: str = os.getenv("APP_NAME", "Movie Data Analysis Platform")
    app_version: str = os.getenv("APP_VERSION", "1.0.0")
    api_v1_prefix: str = os.getenv("API_V1_PREFIX", "/api/v1")

    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

    data_raw_path: str = os.getenv("DATA_RAW_PATH", "data/raw")
    data_processed_path: str = os.getenv("DATA_PROCESSED_PATH", "data/processed")
    data_temp_path: str = os.getenv("DATA_TEMP_PATH", "data/temp")
    movie_lens_dataset_url: str = os.getenv("MOVIE_LENS_DATASET_URL", "")
    movies_file: str = os.getenv("MOVIES_FILE", "movies.dat")
    ratings_file: str = os.getenv("RATINGS_FILE", "ratings.dat")
    tags_file: str = os.getenv("TAGS_FILE", "tags.dat")

    data_delimiter: str = os.getenv("DATA_DELIMITER", "::")

    class Config:
        case_sensitive = False


settings = Settings()
