.PHONY: help install test test-unit test-integration test-performance lint format type-check quality clean run docker-build docker-up docker-down

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy
UVICORN := uvicorn

# Source directories
SRC_DIR := src
TEST_DIR := tests
MAIN_APP := src.main:app

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

##@ Help

help: ## Display this help message
	@echo "$(BLUE)Movie Data Analysis Platform - Makefile Commands$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make $(YELLOW)<target>$(NC)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(GREEN)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Installation

install: ## Install all dependencies
	@echo "$(GREEN)Installing dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)Dependencies installed successfully!$(NC)"

install-dev: install ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	pip install -e ".[dev]"
	@echo "$(GREEN)Development environment ready!$(NC)"

##@ Testing

test: ## Run all tests
	@echo "$(GREEN)Running all tests...$(NC)"
	$(PYTEST) $(TEST_DIR) -v --tb=short

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/unit -v

test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/integration -v

test-performance: ## Run performance tests
	@echo "$(YELLOW)Running performance tests (may take a while)...$(NC)"
	$(PYTEST) $(TEST_DIR)/performance -v -m performance

test-cov: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Coverage report generated in htmlcov/index.html$(NC)"

test-watch: ## Run tests in watch mode
	$(PYTEST) $(TEST_DIR) -f

##@ Code Quality

lint: ## Run flake8 linter
	@echo "$(GREEN)Running flake8 linter...$(NC)"
	$(FLAKE8) $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)Linting complete!$(NC)"

format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code with black...$(NC)"
	$(BLACK) $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)Sorting imports with isort...$(NC)"
	$(ISORT) $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)Code formatted successfully!$(NC)"

format-check: ## Check if code is formatted correctly
	@echo "$(GREEN)Checking code formatting...$(NC)"
	$(BLACK) --check $(SRC_DIR) $(TEST_DIR)
	$(ISORT) --check-only $(SRC_DIR) $(TEST_DIR)

type-check: ## Run mypy type checker
	@echo "$(GREEN)Running mypy type checker...$(NC)"
	$(MYPY) $(SRC_DIR)
	@echo "$(GREEN)Type checking complete!$(NC)"

quality: lint type-check ## Run all code quality checks
	@echo "$(GREEN)All quality checks passed!$(NC)"

##@ Application

run: ## Run the application
	@echo "$(GREEN)Starting Movie Data Analysis Platform...$(NC)"
	$(UVICORN) $(MAIN_APP) --reload --host 0.0.0.0 --port 8000

run-prod: ## Run the application in production mode
	@echo "$(GREEN)Starting application in production mode...$(NC)"
	$(UVICORN) $(MAIN_APP) --host 0.0.0.0 --port 8000 --workers 4

##@ Docker

docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker-compose build
	@echo "$(GREEN)Docker image built successfully!$(NC)"

docker-up: ## Start Docker containers
	@echo "$(GREEN)Starting Docker containers...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)Containers started! API available at http://localhost:8000$(NC)"

docker-down: ## Stop Docker containers
	@echo "$(YELLOW)Stopping Docker containers...$(NC)"
	docker-compose down
	@echo "$(GREEN)Containers stopped.$(NC)"

docker-logs: ## View Docker container logs
	docker-compose logs -f

##@ Cleanup

clean: ## Clean up generated files
	@echo "$(YELLOW)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete!$(NC)"

clean-data: ## Clean generated data files (be careful!)
	@echo "$(RED)WARNING: This will delete generated visualizations and logs!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/visualizations/*; \
		rm -rf logs/*; \
		echo "$(GREEN)Data cleaned.$(NC)"; \
	else \
		echo "$(YELLOW)Cancelled.$(NC)"; \
	fi

##@ CI/CD

ci: format-check lint type-check test ## Run all CI checks
	@echo "$(GREEN)All CI checks passed!$(NC)"

pre-commit: format lint ## Run pre-commit checks
	@echo "$(GREEN)Pre-commit checks complete!$(NC)"

##@ Database

db-migrate: ## Run database migrations (if applicable)
	@echo "$(YELLOW)No database migrations configured yet$(NC)"

db-seed: ## Seed database with sample data
	@echo "$(YELLOW)No database seeding configured yet$(NC)"

##@ Analysis

analyze: ## Run quick data analysis
	@echo "$(GREEN)Running data analysis...$(NC)"
	$(PYTHON) -c "from src.services.movie_analyzer import MovieAnalyzer; from src.services.data_processor import DataProcessor; analyzer = MovieAnalyzer(DataProcessor()); analyzer.load_datasets(); print(analyzer.get_top_movies(limit=10))"

jupyter: ## Start Jupyter notebook
	@echo "$(GREEN)Starting Jupyter notebook...$(NC)"
	jupyter notebook

##@ Documentation

docs: ## Generate documentation
	@echo "$(YELLOW)Documentation generation not configured yet$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(YELLOW)Documentation server not configured yet$(NC)"

##@ Package

build: clean ## Build distribution packages
	@echo "$(GREEN)Building distribution packages...$(NC)"
	python -m pip install --upgrade build
	python -m build
	@echo "$(GREEN)Build complete! Check dist/ directory$(NC)"

build-check: ## Check built packages
	@echo "$(GREEN)Checking built packages...$(NC)"
	python -m pip install --upgrade twine
	python -m twine check dist/*
	@echo "$(GREEN)Package check complete!$(NC)"

install-local: ## Install package locally in editable mode
	@echo "$(GREEN)Installing package in editable mode...$(NC)"
	pip install -e .
	@echo "$(GREEN)Package installed! Try: movie-analysis --help$(NC)"

install-local-dev: ## Install package with dev dependencies
	@echo "$(GREEN)Installing package with dev dependencies...$(NC)"
	pip install -e ".[dev]"
	@echo "$(GREEN)Dev environment ready!$(NC)"

uninstall: ## Uninstall the package
	@echo "$(YELLOW)Uninstalling package...$(NC)"
	pip uninstall -y movie-data-analysis-platform
	@echo "$(GREEN)Package uninstalled.$(NC)"

publish-test: build build-check ## Publish to TestPyPI
	@echo "$(YELLOW)Publishing to TestPyPI...$(NC)"
	python -m twine upload --repository testpypi dist/*
	@echo "$(GREEN)Published to TestPyPI!$(NC)"

publish: build build-check ## Publish to PyPI
	@echo "$(RED)WARNING: This will publish to PyPI!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		python -m twine upload dist/*; \
		echo "$(GREEN)Published to PyPI!$(NC)"; \
	else \
		echo "$(YELLOW)Cancelled.$(NC)"; \
	fi
