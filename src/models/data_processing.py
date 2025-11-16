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
    source_file: Optional[str] = None
    output_file: Optional[str] = None
    rows: Optional[int] = None
    columns: Optional[List[str]] = None
    data: Optional[List[Dict[str, Any]]] = None
    datasets_loaded: Optional[Dict[str, Any]] = None  # For "all" datasets


class CleanDataRequest(BaseModel):
    dataset: str


class CleanDataResponse(BaseModel):
    status: str
    message: str
    source_file: Optional[str] = None
    output_file: Optional[str] = None
    initial_rows: Optional[int] = None
    final_rows: Optional[int] = None
    rows_removed: Optional[int] = None
    columns: Optional[List[str]] = None
    data: Optional[List[Dict[str, Any]]] = None
    datasets_cleaned: Optional[Dict[str, Any]] = None  # For "all" datasets


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
