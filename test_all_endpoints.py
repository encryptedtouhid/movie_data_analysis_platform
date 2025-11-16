#!/usr/bin/env python3
"""
COMPREHENSIVE TEST SUITE FOR ENTIRE APPLICATION
Tests all 20 API endpoints across all modules
"""

import sys
import requests
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"
API_V1 = f"{BASE_URL}/api/v1"


def print_section(title: str):
    """Print section header"""
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)


def print_success(message: str):
    """Print success message"""
    print(f"✅ {message}")


def print_error(message: str):
    """Print error message"""
    print(f"❌ {message}")


def print_info(message: str):
    """Print info message"""
    print(f"ℹ️  {message}")


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = {}

    def add_result(self, test_name: str, passed: bool, message: str = ""):
        self.tests[test_name] = {"passed": passed, "message": message}
        if passed:
            self.passed += 1
            print_success(f"{test_name} - {message if message else 'PASSED'}")
        else:
            self.failed += 1
            print_error(f"{test_name} - {message if message else 'FAILED'}")

    def summary(self):
        print_section("TEST SUMMARY")
        print(f"\n{'Test Name':^60} {'Result':^15} {'Details':^25}")
        print("-" * 100)

        for test_name, result in self.tests.items():
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            message = result["message"][:25] if result["message"] else ""
            print(f"{test_name:^60} {status:^15} {message:^25}")

        print("-" * 100)
        total = self.passed + self.failed
        print(f"\nTotal Tests: {total}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"Success Rate: {(self.passed/total)*100:.1f}%")

        return self.failed == 0


results = TestResults()


# ================================================================================
# 1. HEALTH ENDPOINT TESTS
# ================================================================================

def test_health_endpoint():
    """Test GET /api/v1/health"""
    print_section("TEST MODULE 1: Health Endpoint")

    try:
        print_info("Testing health check endpoint...")
        response = requests.get(f"{API_V1}/health", timeout=10)

        if response.status_code == 200:
            data = response.json()

            # Verify structure
            assert "status" in data, "Missing 'status' field"
            assert "data_files" in data, "Missing 'data_files' field"
            assert "recommender" in data, "Missing 'recommender' field"

            # Verify data files
            required_files = ["movies.dat", "ratings.dat", "tags.dat", "users.dat"]
            for file in required_files:
                assert file in data["data_files"], f"Missing {file} in data_files"
                assert data["data_files"][file]["exists"], f"{file} does not exist"

            results.add_result("Health Check", True, f"Status: {data['status']}")
        else:
            results.add_result("Health Check", False, f"HTTP {response.status_code}")

    except Exception as e:
        results.add_result("Health Check", False, str(e))


# ================================================================================
# 2. DATA PROCESSING ENDPOINT TESTS
# ================================================================================

def test_data_processing_endpoints():
    """Test all 6 data processing endpoints"""
    print_section("TEST MODULE 2: Data Processing Endpoints")

    # Test 1: Load Data
    try:
        print_info("Testing load_data endpoint...")
        response = requests.post(
            f"{API_V1}/dataprocess/load_data",
            json={"dataset": "movies"},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            results.add_result("Data Processing - Load Data", True, f"Loaded {data.get('rows_loaded', 0)} rows")
        else:
            results.add_result("Data Processing - Load Data", False, f"HTTP {response.status_code}")
    except Exception as e:
        results.add_result("Data Processing - Load Data", False, str(e))

    # Test 2: Clean Data
    try:
        print_info("Testing clean_data endpoint...")
        response = requests.post(
            f"{API_V1}/dataprocess/clean_data",
            json={"dataset": "movies"},
            timeout=30
        )
        if response.status_code == 200:
            results.add_result("Data Processing - Clean Data", True)
        else:
            results.add_result("Data Processing - Clean Data", False, f"HTTP {response.status_code}")
    except Exception as e:
        results.add_result("Data Processing - Clean Data", False, str(e))

    # Test 3: Aggregate Statistics
    try:
        print_info("Testing aggregate_statistics endpoint...")
        response = requests.post(
            f"{API_V1}/dataprocess/aggregate_statistics",
            json={"dataset": "movies"},
            timeout=30
        )
        if response.status_code == 200:
            results.add_result("Data Processing - Aggregate Stats", True)
        else:
            results.add_result("Data Processing - Aggregate Stats", False, f"HTTP {response.status_code}")
    except Exception as e:
        results.add_result("Data Processing - Aggregate Stats", False, str(e))

    # Test 4: Filter Data
    try:
        print_info("Testing filter_data endpoint...")
        response = requests.post(
            f"{API_V1}/dataprocess/filter_data",
            json={"dataset": "ratings", "min_rating": 4.0, "limit": 100},
            timeout=30
        )
        if response.status_code == 200:
            results.add_result("Data Processing - Filter Data", True)
        else:
            results.add_result("Data Processing - Filter Data", False, f"HTTP {response.status_code}")
    except Exception as e:
        results.add_result("Data Processing - Filter Data", False, str(e))

    # Test 5: Export Data
    try:
        print_info("Testing export_data endpoint...")
        response = requests.post(
            f"{API_V1}/dataprocess/export_data",
            json={"dataset": "movies", "format": "json", "limit": 10},
            timeout=30
        )
        if response.status_code == 200:
            results.add_result("Data Processing - Export Data", True)
        else:
            results.add_result("Data Processing - Export Data", False, f"HTTP {response.status_code}")
    except Exception as e:
        results.add_result("Data Processing - Export Data", False, str(e))

    # Test 6: Process All Data
    try:
        print_info("Testing process_all_data endpoint (may take longer)...")
        response = requests.post(
            f"{API_V1}/dataprocess/process_all_data",
            timeout=120
        )
        if response.status_code == 200:
            results.add_result("Data Processing - Process All", True)
        else:
            results.add_result("Data Processing - Process All", False, f"HTTP {response.status_code}")
    except Exception as e:
        results.add_result("Data Processing - Process All", False, str(e))


# ================================================================================
# 3. BASIC ANALYSIS ENDPOINT TESTS
# ================================================================================

def test_basic_analysis_endpoints():
    """Test basic analysis endpoints (non-advanced)"""
    print_section("TEST MODULE 3: Basic Analysis Endpoints")

    # Test 1: Top Movies
    try:
        print_info("Testing top_movies endpoint...")
        response = requests.post(
            f"{API_V1}/analysis/top_movies",
            json={"limit": 10, "min_ratings": 50},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            results.add_result("Analysis - Top Movies", True, f"Found {data.get('total_found', 0)} movies")
        else:
            results.add_result("Analysis - Top Movies", False, f"HTTP {response.status_code}")
    except Exception as e:
        results.add_result("Analysis - Top Movies", False, str(e))

    # Test 2: Genre Trends
    try:
        print_info("Testing genre_trends endpoint...")
        response = requests.get(f"{API_V1}/analysis/genre_trends", timeout=60)
        if response.status_code == 200:
            results.add_result("Analysis - Genre Trends", True)
        else:
            results.add_result("Analysis - Genre Trends", False, f"HTTP {response.status_code}")
    except Exception as e:
        results.add_result("Analysis - Genre Trends", False, str(e))

    # Test 3: User Statistics
    try:
        print_info("Testing user_statistics endpoint...")
        response = requests.post(
            f"{API_V1}/analysis/user_statistics",
            json={"user_id": 1},
            timeout=30
        )
        if response.status_code == 200:
            results.add_result("Analysis - User Statistics", True)
        else:
            results.add_result("Analysis - User Statistics", False, f"HTTP {response.status_code}")
    except Exception as e:
        results.add_result("Analysis - User Statistics", False, str(e))

    # Test 4: Time Series
    try:
        print_info("Testing time_series endpoint...")
        response = requests.get(f"{API_V1}/analysis/time_series", timeout=30)
        if response.status_code == 200:
            results.add_result("Analysis - Time Series", True)
        else:
            results.add_result("Analysis - Time Series", False, f"HTTP {response.status_code}")
    except Exception as e:
        results.add_result("Analysis - Time Series", False, str(e))

    # Test 5: Correlation Analysis
    try:
        print_info("Testing correlation_analysis endpoint...")
        response = requests.get(f"{API_V1}/analysis/correlation_analysis", timeout=30)
        if response.status_code == 200:
            results.add_result("Analysis - Correlation", True)
        else:
            results.add_result("Analysis - Correlation", False, f"HTTP {response.status_code}")
    except Exception as e:
        results.add_result("Analysis - Correlation", False, str(e))

    # Test 6: Visualization
    try:
        print_info("Testing visualize endpoint...")
        response = requests.post(
            f"{API_V1}/analysis/visualize",
            json={"visualization_type": "rating_distribution"},
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            results.add_result("Analysis - Visualization", True, f"Created {data.get('visualization_type', '')}")
        else:
            results.add_result("Analysis - Visualization", False, f"HTTP {response.status_code}")
    except Exception as e:
        results.add_result("Analysis - Visualization", False, str(e))


# ================================================================================
# 4. ADVANCED ANALYSIS ENDPOINT TESTS
# ================================================================================

def test_advanced_analysis_endpoints():
    """Test advanced analytics endpoints"""
    print_section("TEST MODULE 4: Advanced Analytics Endpoints")

    # Test 1: User Clustering
    try:
        print_info("Testing clustering endpoint (may take 30-60s)...")
        response = requests.post(
            f"{API_V1}/analysis/clustering",
            json={"n_clusters": 5},
            timeout=120
        )
        if response.status_code == 200:
            data = response.json()
            results.add_result("Advanced - User Clustering", True, f"Created {data['result'].get('n_clusters', 0)} clusters")
        else:
            results.add_result("Advanced - User Clustering", False, f"HTTP {response.status_code}")
    except Exception as e:
        results.add_result("Advanced - User Clustering", False, str(e))

    # Test 2: Trend Analysis
    try:
        print_info("Testing trend_analysis endpoint...")
        response = requests.post(
            f"{API_V1}/analysis/trend_analysis",
            json={"period": "month"},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            results.add_result("Advanced - Trend Analysis", True, f"Trend: {data['result'].get('overall_trend', '')}")
        else:
            results.add_result("Advanced - Trend Analysis", False, f"HTTP {response.status_code}")
    except Exception as e:
        results.add_result("Advanced - Trend Analysis", False, str(e))

    # Test 3: Anomaly Detection
    try:
        print_info("Testing anomaly_detection endpoint...")
        response = requests.post(
            f"{API_V1}/analysis/anomaly_detection",
            json={"method": "iqr", "sensitivity": 1.5},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            summary = data['result'].get('summary', {})
            results.add_result("Advanced - Anomaly Detection", True, f"Found {summary.get('total_anomalous_users', 0)} anomalies")
        else:
            results.add_result("Advanced - Anomaly Detection", False, f"HTTP {response.status_code}")
    except Exception as e:
        results.add_result("Advanced - Anomaly Detection", False, str(e))

    # Test 4: Rating Sentiment Analysis
    try:
        print_info("Testing rating_sentiment endpoint...")
        response = requests.post(
            f"{API_V1}/analysis/rating_sentiment",
            json={"analysis_type": "overall"},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            sentiment = data['sentiment_analysis'].get('overall_sentiment', {})
            results.add_result("Advanced - Rating Sentiment", True, f"Positive: {sentiment.get('positive', 0):.1f}%")
        else:
            results.add_result("Advanced - Rating Sentiment", False, f"HTTP {response.status_code}")
    except Exception as e:
        results.add_result("Advanced - Rating Sentiment", False, str(e))


# ================================================================================
# 5. ML RECOMMENDATIONS ENDPOINT TESTS
# ================================================================================

def test_ml_recommendations_endpoints():
    """Test ML recommendation endpoints"""
    print_section("TEST MODULE 5: ML Recommendations Endpoints")

    # Test 1: Similar Movies
    try:
        print_info("Testing similar_movies endpoint...")
        response = requests.post(
            f"{API_V1}/recommendations/similar_movies",
            json={"movie_id": 1, "limit": 10, "min_common_ratings": 50},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            results.add_result("ML - Similar Movies", True, f"Found {len(data.get('similar_movies', []))} movies")
        else:
            results.add_result("ML - Similar Movies", False, f"HTTP {response.status_code}")
    except Exception as e:
        results.add_result("ML - Similar Movies", False, str(e))

    # Test 2: User Recommendations
    try:
        print_info("Testing user_recommendations endpoint...")
        response = requests.post(
            f"{API_V1}/recommendations/user_recommendations",
            json={"user_id": 1, "limit": 10, "min_user_overlap": 50},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            results.add_result("ML - User Recommendations", True, f"Found {len(data.get('recommendations', []))} recommendations")
        else:
            results.add_result("ML - User Recommendations", False, f"HTTP {response.status_code}")
    except Exception as e:
        results.add_result("ML - User Recommendations", False, str(e))


# ================================================================================
# 6. HOME PAGE TEST
# ================================================================================

def test_home_page():
    """Test home page endpoint"""
    print_section("TEST MODULE 6: Home Page")

    try:
        print_info("Testing home page endpoint...")
        response = requests.get(BASE_URL, timeout=10)
        if response.status_code == 200:
            html = response.text
            # Verify it contains expected content
            assert "Movie Data Analysis Platform" in html, "Missing page title"
            assert "Data Processing Endpoints" in html, "Missing data processing section"
            assert "Analysis Endpoints" in html, "Missing analysis section"
            assert "ML Recommendations" in html, "Missing ML section"
            results.add_result("Home Page", True, "All sections present")
        else:
            results.add_result("Home Page", False, f"HTTP {response.status_code}")
    except Exception as e:
        results.add_result("Home Page", False, str(e))


# ================================================================================
# MAIN TEST EXECUTION
# ================================================================================

def main():
    """Run all tests"""
    print("\n" + "🚀" * 50)
    print(" " * 30 + "COMPREHENSIVE APPLICATION TEST SUITE")
    print(" " * 35 + "Testing All 20 API Endpoints")
    print("🚀" * 50)

    # Check if server is running
    try:
        print_info(f"Checking if server is running at {BASE_URL}...")
        response = requests.get(BASE_URL, timeout=5)
        print_success("Server is running!")
    except Exception as e:
        print_error(f"Server is not running at {BASE_URL}")
        print_error(f"Please start the server with: python -m uvicorn src.main:app --host 0.0.0.0 --port 8000")
        return 1

    # Run all test modules
    test_health_endpoint()                  # 1 endpoint
    test_data_processing_endpoints()        # 6 endpoints
    test_basic_analysis_endpoints()         # 6 endpoints
    test_advanced_analysis_endpoints()      # 4 endpoints
    test_ml_recommendations_endpoints()     # 2 endpoints
    test_home_page()                        # 1 endpoint
    # Total: 20 endpoints

    # Print summary
    success = results.summary()

    if success:
        print("\n🎉 All tests passed successfully! 🎉")
        return 0
    else:
        print(f"\n⚠️  Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
