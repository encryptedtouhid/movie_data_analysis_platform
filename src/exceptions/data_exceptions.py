from typing import Optional


class DataProcessingError(Exception):
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class DataDownloadError(DataProcessingError):
    pass


class DataLoadError(DataProcessingError):
    pass


class DataCleaningError(DataProcessingError):
    pass


class DataValidationError(DataProcessingError):
    pass


class DataAggregationError(DataProcessingError):
    pass


class DataFilterError(DataProcessingError):
    pass


class DataAnalysisError(DataProcessingError):
    pass
