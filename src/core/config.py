from typing import Optional
from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "Movie Data Analysis Platform"
    app_version: str = "1.0.0"
    api_v1_prefix: str = "/api/v1"

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    data_raw_path: str = "data/raw"
    data_processed_path: str = "data/processed"
    movies_file: str = "movies.dat"
    ratings_file: str = "ratings.dat"
    tags_file: str = "tags.dat"

    data_delimiter: str = "::"

    class Config:
        case_sensitive = False


settings = Settings()
