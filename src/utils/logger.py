import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class LoggerSetup:
    _loggers = {}

    @staticmethod
    def get_logger(
        name: str,
        module_folder: str,
        level: int = logging.INFO
    ) -> logging.Logger:
        if name in LoggerSetup._loggers:
            return LoggerSetup._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False

        if logger.handlers:
            return logger

        logs_base_path = Path("logs")
        module_log_path = logs_base_path / module_folder
        module_log_path.mkdir(parents=True, exist_ok=True)

        log_filename = f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        log_file_path = module_log_path / log_filename

        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(level)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        LoggerSetup._loggers[name] = logger

        return logger


def get_logger(name: str, module_folder: str, level: int = logging.INFO) -> logging.Logger:
    return LoggerSetup.get_logger(name, module_folder, level)
