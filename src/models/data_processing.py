from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class DataProcessResponse(BaseModel):
    status: str
    message: str
    converted_files: Dict[str, str]
    movies_result: Dict[str, Any]
    ratings_result: Dict[str, Any]
    tags_result: Dict[str, Any]
    users_result: Dict[str, Any]


class LoadDataRequest(BaseModel):
    dataset: str


class LoadDataResponse(BaseModel):
    status: str
    message: str
    source_file: str
    output_file: str
    rows: int
    columns: List[str]
    sample: List[Dict[str, Any]]


class CleanDataRequest(BaseModel):
    dataset: str


class CleanDataResponse(BaseModel):
    status: str
    message: str
    source_file: str
    output_file: str
    initial_rows: int
    final_rows: int
    rows_removed: int
    columns: List[str]
    sample: List[Dict[str, Any]]


class AggregateStatsRequest(BaseModel):
    dataset: str


class AggregateStatsResponse(BaseModel):
    status: str
    message: str
    statistics: Dict[str, Any]


class FilterDataRequest(BaseModel):
    dataset: str
    min_rating: Optional[float] = None
    max_rating: Optional[float] = None
    user_id: Optional[int] = None
    movie_id: Optional[int] = None
    genres: Optional[str] = None
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    limit: Optional[int] = None


class FilterDataResponse(BaseModel):
    status: str
    message: str
    original_rows: int
    filtered_rows: int
    filters_applied: Dict[str, Any]
    data: Any
    sample: Any


class ExportDataRequest(BaseModel):
    dataset: str
    format: str  # 'csv' or 'json'
    file_name: Optional[str] = None
    orient: Optional[str] = 'records'  # For JSON: records, index, columns, values, split, table
    include_index: Optional[bool] = False  # For CSV


class ExportDataResponse(BaseModel):
    status: str
    message: str
    file_path: str
    file_size_mb: float
    rows_exported: int
    format: str
