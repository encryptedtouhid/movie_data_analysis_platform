#!/usr/bin/env python
"""
Command-line interface for Movie Data Analysis Platform.

This module provides CLI entry points for running the application
and performing data analysis tasks from the command line.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List
import uvicorn

from src.core.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__, "cli")


def main(args: Optional[List[str]] = None) -> int:
    """
    Main CLI entry point for movie data analysis platform.

    Args:
        args: Command line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        description="Movie Data Analysis Platform - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --version                     Show version information
  %(prog)s server                        Start the API server
  %(prog)s analyze --top-movies 10       Get top 10 movies
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {settings.app_version}",
        help="Show version information",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Server command
    server_parser = subparsers.add_parser("server", help="Start the API server")
    server_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    server_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    server_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    server_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (production mode)",
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run data analysis")
    analyze_parser.add_argument(
        "--top-movies",
        type=int,
        metavar="N",
        help="Get top N movies by rating",
    )
    analyze_parser.add_argument(
        "--genre-trends",
        action="store_true",
        help="Analyze genre trends over time",
    )
    analyze_parser.add_argument(
        "--user-stats",
        type=int,
        metavar="USER_ID",
        help="Get statistics for a specific user",
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests only",
    )
    test_parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests only",
    )
    test_parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report",
    )

    # Parse arguments
    parsed_args = parser.parse_args(args)

    # Handle commands
    if parsed_args.command == "server":
        return run_server(
            host=parsed_args.host,
            port=parsed_args.port,
            reload=parsed_args.reload,
            workers=parsed_args.workers,
        )
    elif parsed_args.command == "analyze":
        return run_analysis(parsed_args)
    elif parsed_args.command == "test":
        return run_tests(parsed_args)
    else:
        parser.print_help()
        return 0


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1,
) -> int:
    """
    Start the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
        workers: Number of worker processes

    Returns:
        Exit code
    """
    logger.info(f"Starting Movie Data Analysis Platform API server on {host}:{port}")

    try:
        uvicorn.run(
            "src.main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
            log_level="info",
        )
        return 0
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return 1


def run_analysis(args: argparse.Namespace) -> int:
    """
    Run data analysis tasks.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code
    """
    try:
        from src.services.data_processor import DataProcessor
        from src.services.movie_analyzer import MovieAnalyzer

        # Initialize services
        processor = DataProcessor()
        analyzer = MovieAnalyzer(processor)

        logger.info("Loading datasets...")
        analyzer.load_datasets()

        # Execute requested analysis
        if args.top_movies:
            logger.info(f"Getting top {args.top_movies} movies...")
            results = analyzer.get_top_movies(limit=args.top_movies)
            print("\n=== Top Movies ===")
            print(results.to_string(index=False))

        if args.genre_trends:
            logger.info("Analyzing genre trends...")
            results = analyzer.analyze_genre_trends()
            print("\n=== Genre Trends ===")
            print(results)

        if args.user_stats:
            logger.info(f"Getting statistics for user {args.user_stats}...")
            results = analyzer.get_user_statistics(args.user_stats)
            print("\n=== User Statistics ===")
            for key, value in results.items():
                print(f"{key}: {value}")

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_tests(args: argparse.Namespace) -> int:
    """
    Run test suite.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code
    """
    import subprocess

    cmd = ["python", "-m", "pytest"]

    if args.unit:
        cmd.append("tests/unit")
    elif args.integration:
        cmd.append("tests/integration")
    else:
        cmd.append("tests")

    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term-missing"])

    cmd.extend(["-v", "--tb=short"])

    logger.info(f"Running tests: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except Exception as e:
        logger.error(f"Failed to run tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
