#!/usr/bin/env python3
"""
COMPREHENSIVE ERROR TESTING SUITE
Tests invalid inputs, edge cases, and exception handling across all endpoints
"""

import sys
import requests
import json

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


class ErrorTestResults:
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
        print_section("ERROR TESTING SUMMARY")
        print(f"\n{'Test Name':^70} {'Result':^15} {'Details':^15}")
        print("-" * 100)

        for test_name, result in self.tests.items():
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            message = result["message"][:15] if result["message"] else ""
            print(f"{test_name:^70} {status:^15} {message:^15}")

        print("-" * 100)
        total = self.passed + self.failed
        print(f"\nTotal Error Tests: {total}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"Success Rate: {(self.passed/total)*100:.1f}%")

        return self.failed == 0


results = ErrorTestResults()


# ================================================================================
# 1. INVALID INPUT TESTS - Top Movies
# ================================================================================

def test_top_movies_errors():
    """Test top_movies with invalid inputs"""
    print_section("ERROR TEST MODULE 1: Top Movies Invalid Inputs")

    # Test 1: Negative limit
    try:
        print_info("Testing negative limit...")
        response = requests.post(
            f"{API_V1}/analysis/top_movies",
            json={"limit": -5, "min_ratings": 50},
            timeout=10
        )
        if response.status_code == 422:  # Pydantic validation error
            results.add_result("Top Movies - Negative Limit", True, "422 Validation Error")
        else:
            results.add_result("Top Movies - Negative Limit", False, f"Expected 422, got {response.status_code}")
    except Exception as e:
        results.add_result("Top Movies - Negative Limit", False, str(e))

    # Test 2: Limit exceeds maximum (>100)
    try:
        print_info("Testing limit exceeds maximum...")
        response = requests.post(
            f"{API_V1}/analysis/top_movies",
            json={"limit": 500, "min_ratings": 50},
            timeout=10
        )
        if response.status_code == 422:
            results.add_result("Top Movies - Limit Exceeds Max", True, "422 Validation Error")
        else:
            results.add_result("Top Movies - Limit Exceeds Max", False, f"Expected 422, got {response.status_code}")
    except Exception as e:
        results.add_result("Top Movies - Limit Exceeds Max", False, str(e))

    # Test 3: Invalid data type for limit
    try:
        print_info("Testing invalid data type...")
        response = requests.post(
            f"{API_V1}/analysis/top_movies",
            json={"limit": "invalid", "min_ratings": 50},
            timeout=10
        )
        if response.status_code == 422:
            results.add_result("Top Movies - Invalid Type", True, "422 Validation Error")
        else:
            results.add_result("Top Movies - Invalid Type", False, f"Expected 422, got {response.status_code}")
    except Exception as e:
        results.add_result("Top Movies - Invalid Type", False, str(e))

    # Test 4: Missing required field
    try:
        print_info("Testing with empty body...")
        response = requests.post(
            f"{API_V1}/analysis/top_movies",
            json={},
            timeout=10
        )
        # Should use defaults or return 200
        if response.status_code in [200, 422]:
            results.add_result("Top Movies - Empty Body", True, f"HTTP {response.status_code}")
        else:
            results.add_result("Top Movies - Empty Body", False, f"Unexpected {response.status_code}")
    except Exception as e:
        results.add_result("Top Movies - Empty Body", False, str(e))

    # Test 5: Malformed JSON
    try:
        print_info("Testing malformed JSON...")
        response = requests.post(
            f"{API_V1}/analysis/top_movies",
            data="{invalid json}",
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if response.status_code == 422:
            results.add_result("Top Movies - Malformed JSON", True, "422 Validation Error")
        else:
            results.add_result("Top Movies - Malformed JSON", False, f"Expected 422, got {response.status_code}")
    except Exception as e:
        results.add_result("Top Movies - Malformed JSON", False, str(e))


# ================================================================================
# 2. INVALID INPUT TESTS - User Statistics
# ================================================================================

def test_user_statistics_errors():
    """Test user_statistics with invalid inputs"""
    print_section("ERROR TEST MODULE 2: User Statistics Invalid Inputs")

    # Test 1: Non-existent user ID
    try:
        print_info("Testing non-existent user ID...")
        response = requests.post(
            f"{API_V1}/analysis/user_statistics",
            json={"user_id": 99999999},
            timeout=10
        )
        if response.status_code in [404, 400]:
            results.add_result("User Stats - Non-existent ID", True, f"HTTP {response.status_code}")
        else:
            results.add_result("User Stats - Non-existent ID", False, f"Expected 404/400, got {response.status_code}")
    except Exception as e:
        results.add_result("User Stats - Non-existent ID", False, str(e))

    # Test 2: Negative user ID
    try:
        print_info("Testing negative user ID...")
        response = requests.post(
            f"{API_V1}/analysis/user_statistics",
            json={"user_id": -1},
            timeout=10
        )
        if response.status_code in [422, 400, 404]:
            results.add_result("User Stats - Negative ID", True, f"HTTP {response.status_code}")
        else:
            results.add_result("User Stats - Negative ID", False, f"Expected 422/400/404, got {response.status_code}")
    except Exception as e:
        results.add_result("User Stats - Negative ID", False, str(e))

    # Test 3: Zero user ID
    try:
        print_info("Testing zero user ID...")
        response = requests.post(
            f"{API_V1}/analysis/user_statistics",
            json={"user_id": 0},
            timeout=10
        )
        if response.status_code in [404, 400]:
            results.add_result("User Stats - Zero ID", True, f"HTTP {response.status_code}")
        else:
            results.add_result("User Stats - Zero ID", False, f"Expected 404/400, got {response.status_code}")
    except Exception as e:
        results.add_result("User Stats - Zero ID", False, str(e))

    # Test 4: Invalid type for user_id
    try:
        print_info("Testing string for user_id...")
        response = requests.post(
            f"{API_V1}/analysis/user_statistics",
            json={"user_id": "invalid"},
            timeout=10
        )
        if response.status_code == 422:
            results.add_result("User Stats - Invalid Type", True, "422 Validation Error")
        else:
            results.add_result("User Stats - Invalid Type", False, f"Expected 422, got {response.status_code}")
    except Exception as e:
        results.add_result("User Stats - Invalid Type", False, str(e))


# ================================================================================
# 3. INVALID INPUT TESTS - Clustering
# ================================================================================

def test_clustering_errors():
    """Test clustering with invalid inputs"""
    print_section("ERROR TEST MODULE 3: Clustering Invalid Inputs")

    # Test 1: n_clusters too small (< 2)
    try:
        print_info("Testing n_clusters = 1...")
        response = requests.post(
            f"{API_V1}/analysis/clustering",
            json={"n_clusters": 1},
            timeout=10
        )
        if response.status_code == 422:
            results.add_result("Clustering - n_clusters=1", True, "422 Validation Error")
        else:
            results.add_result("Clustering - n_clusters=1", False, f"Expected 422, got {response.status_code}")
    except Exception as e:
        results.add_result("Clustering - n_clusters=1", False, str(e))

    # Test 2: n_clusters too large (> 10)
    try:
        print_info("Testing n_clusters = 50...")
        response = requests.post(
            f"{API_V1}/analysis/clustering",
            json={"n_clusters": 50},
            timeout=10
        )
        if response.status_code == 422:
            results.add_result("Clustering - n_clusters=50", True, "422 Validation Error")
        else:
            results.add_result("Clustering - n_clusters=50", False, f"Expected 422, got {response.status_code}")
    except Exception as e:
        results.add_result("Clustering - n_clusters=50", False, str(e))

    # Test 3: Negative n_clusters
    try:
        print_info("Testing negative n_clusters...")
        response = requests.post(
            f"{API_V1}/analysis/clustering",
            json={"n_clusters": -5},
            timeout=10
        )
        if response.status_code == 422:
            results.add_result("Clustering - Negative Clusters", True, "422 Validation Error")
        else:
            results.add_result("Clustering - Negative Clusters", False, f"Expected 422, got {response.status_code}")
    except Exception as e:
        results.add_result("Clustering - Negative Clusters", False, str(e))


# ================================================================================
# 4. INVALID INPUT TESTS - Anomaly Detection
# ================================================================================

def test_anomaly_detection_errors():
    """Test anomaly detection with invalid inputs"""
    print_section("ERROR TEST MODULE 4: Anomaly Detection Invalid Inputs")

    # Test 1: Invalid method
    try:
        print_info("Testing invalid method...")
        response = requests.post(
            f"{API_V1}/analysis/anomaly_detection",
            json={"method": "invalid_method", "sensitivity": 1.5},
            timeout=10
        )
        if response.status_code in [422, 400]:
            results.add_result("Anomaly - Invalid Method", True, f"HTTP {response.status_code}")
        else:
            results.add_result("Anomaly - Invalid Method", False, f"Expected 422/400, got {response.status_code}")
    except Exception as e:
        results.add_result("Anomaly - Invalid Method", False, str(e))

    # Test 2: Sensitivity out of range (> 5)
    try:
        print_info("Testing sensitivity > 5...")
        response = requests.post(
            f"{API_V1}/analysis/anomaly_detection",
            json={"method": "iqr", "sensitivity": 10.0},
            timeout=10
        )
        if response.status_code == 422:
            results.add_result("Anomaly - Sensitivity > 5", True, "422 Validation Error")
        else:
            results.add_result("Anomaly - Sensitivity > 5", False, f"Expected 422, got {response.status_code}")
    except Exception as e:
        results.add_result("Anomaly - Sensitivity > 5", False, str(e))

    # Test 3: Negative sensitivity
    try:
        print_info("Testing negative sensitivity...")
        response = requests.post(
            f"{API_V1}/analysis/anomaly_detection",
            json={"method": "iqr", "sensitivity": -1.0},
            timeout=10
        )
        if response.status_code == 422:
            results.add_result("Anomaly - Negative Sensitivity", True, "422 Validation Error")
        else:
            results.add_result("Anomaly - Negative Sensitivity", False, f"Expected 422, got {response.status_code}")
    except Exception as e:
        results.add_result("Anomaly - Negative Sensitivity", False, str(e))

    # Test 4: Zero sensitivity
    try:
        print_info("Testing zero sensitivity...")
        response = requests.post(
            f"{API_V1}/analysis/anomaly_detection",
            json={"method": "iqr", "sensitivity": 0.0},
            timeout=10
        )
        if response.status_code == 422:
            results.add_result("Anomaly - Zero Sensitivity", True, "422 Validation Error")
        else:
            results.add_result("Anomaly - Zero Sensitivity", False, f"Expected 422, got {response.status_code}")
    except Exception as e:
        results.add_result("Anomaly - Zero Sensitivity", False, str(e))


# ================================================================================
# 5. INVALID INPUT TESTS - Rating Sentiment
# ================================================================================

def test_rating_sentiment_errors():
    """Test rating sentiment with invalid inputs"""
    print_section("ERROR TEST MODULE 5: Rating Sentiment Invalid Inputs")

    # Test 1: Invalid analysis_type
    try:
        print_info("Testing invalid analysis_type...")
        response = requests.post(
            f"{API_V1}/analysis/rating_sentiment",
            json={"analysis_type": "invalid_type"},
            timeout=10
        )
        if response.status_code in [422, 400]:
            results.add_result("Sentiment - Invalid Type", True, f"HTTP {response.status_code}")
        else:
            results.add_result("Sentiment - Invalid Type", False, f"Expected 422/400, got {response.status_code}")
    except Exception as e:
        results.add_result("Sentiment - Invalid Type", False, str(e))

    # Test 2: movie_sentiment without movie_id
    try:
        print_info("Testing movie_sentiment without movie_id...")
        response = requests.post(
            f"{API_V1}/analysis/rating_sentiment",
            json={"analysis_type": "movie_sentiment"},
            timeout=10
        )
        if response.status_code in [422, 400]:
            results.add_result("Sentiment - Missing movie_id", True, f"HTTP {response.status_code}")
        else:
            results.add_result("Sentiment - Missing movie_id", False, f"Expected 422/400, got {response.status_code}")
    except Exception as e:
        results.add_result("Sentiment - Missing movie_id", False, str(e))

    # Test 3: user_sentiment without user_id
    try:
        print_info("Testing user_sentiment without user_id...")
        response = requests.post(
            f"{API_V1}/analysis/rating_sentiment",
            json={"analysis_type": "user_sentiment"},
            timeout=10
        )
        if response.status_code in [422, 400]:
            results.add_result("Sentiment - Missing user_id", True, f"HTTP {response.status_code}")
        else:
            results.add_result("Sentiment - Missing user_id", False, f"Expected 422/400, got {response.status_code}")
    except Exception as e:
        results.add_result("Sentiment - Missing user_id", False, str(e))

    # Test 4: Non-existent movie_id
    try:
        print_info("Testing non-existent movie_id...")
        response = requests.post(
            f"{API_V1}/analysis/rating_sentiment",
            json={"analysis_type": "movie_sentiment", "movie_id": 99999999},
            timeout=10
        )
        if response.status_code in [404, 400]:
            results.add_result("Sentiment - Non-existent movie", True, f"HTTP {response.status_code}")
        else:
            results.add_result("Sentiment - Non-existent movie", False, f"Expected 404/400, got {response.status_code}")
    except Exception as e:
        results.add_result("Sentiment - Non-existent movie", False, str(e))


# ================================================================================
# 6. INVALID INPUT TESTS - Trend Analysis
# ================================================================================

def test_trend_analysis_errors():
    """Test trend analysis with invalid inputs"""
    print_section("ERROR TEST MODULE 6: Trend Analysis Invalid Inputs")

    # Test 1: Invalid period
    try:
        print_info("Testing invalid period...")
        response = requests.post(
            f"{API_V1}/analysis/trend_analysis",
            json={"period": "invalid_period"},
            timeout=10
        )
        if response.status_code in [422, 400]:
            results.add_result("Trend - Invalid Period", True, f"HTTP {response.status_code}")
        else:
            results.add_result("Trend - Invalid Period", False, f"Expected 422/400, got {response.status_code}")
    except Exception as e:
        results.add_result("Trend - Invalid Period", False, str(e))

    # Test 2: Empty period
    try:
        print_info("Testing empty period...")
        response = requests.post(
            f"{API_V1}/analysis/trend_analysis",
            json={"period": ""},
            timeout=10
        )
        if response.status_code in [422, 400]:
            results.add_result("Trend - Empty Period", True, f"HTTP {response.status_code}")
        else:
            results.add_result("Trend - Empty Period", False, f"Expected 422/400, got {response.status_code}")
    except Exception as e:
        results.add_result("Trend - Empty Period", False, str(e))


# ================================================================================
# 7. INVALID INPUT TESTS - Visualization
# ================================================================================

def test_visualization_errors():
    """Test visualization with invalid inputs"""
    print_section("ERROR TEST MODULE 7: Visualization Invalid Inputs")

    # Test 1: Invalid visualization_type
    try:
        print_info("Testing invalid visualization_type...")
        response = requests.post(
            f"{API_V1}/analysis/visualize",
            json={"visualization_type": "invalid_type"},
            timeout=10
        )
        if response.status_code in [422, 400]:
            results.add_result("Viz - Invalid Type", True, f"HTTP {response.status_code}")
        else:
            results.add_result("Viz - Invalid Type", False, f"Expected 422/400, got {response.status_code}")
    except Exception as e:
        results.add_result("Viz - Invalid Type", False, str(e))

    # Test 2: Missing visualization_type
    try:
        print_info("Testing missing visualization_type...")
        response = requests.post(
            f"{API_V1}/analysis/visualize",
            json={},
            timeout=10
        )
        if response.status_code == 422:
            results.add_result("Viz - Missing Type", True, "422 Validation Error")
        else:
            results.add_result("Viz - Missing Type", False, f"Expected 422, got {response.status_code}")
    except Exception as e:
        results.add_result("Viz - Missing Type", False, str(e))


# ================================================================================
# 8. INVALID INPUT TESTS - ML Recommendations
# ================================================================================

def test_recommendations_errors():
    """Test ML recommendations with invalid inputs"""
    print_section("ERROR TEST MODULE 8: ML Recommendations Invalid Inputs")

    # Test 1: Similar movies - non-existent movie_id
    try:
        print_info("Testing similar_movies with non-existent ID...")
        response = requests.post(
            f"{API_V1}/recommendations/similar_movies",
            json={"movie_id": 99999999, "limit": 10},
            timeout=10
        )
        if response.status_code in [404, 400]:
            results.add_result("Similar Movies - Non-existent ID", True, f"HTTP {response.status_code}")
        else:
            results.add_result("Similar Movies - Non-existent ID", False, f"Expected 404/400, got {response.status_code}")
    except Exception as e:
        results.add_result("Similar Movies - Non-existent ID", False, str(e))

    # Test 2: Similar movies - negative movie_id
    try:
        print_info("Testing similar_movies with negative ID...")
        response = requests.post(
            f"{API_V1}/recommendations/similar_movies",
            json={"movie_id": -1, "limit": 10},
            timeout=10
        )
        if response.status_code in [422, 400, 404]:
            results.add_result("Similar Movies - Negative ID", True, f"HTTP {response.status_code}")
        else:
            results.add_result("Similar Movies - Negative ID", False, f"Expected 422/400/404, got {response.status_code}")
    except Exception as e:
        results.add_result("Similar Movies - Negative ID", False, str(e))

    # Test 3: User recommendations - non-existent user_id
    try:
        print_info("Testing user_recommendations with non-existent ID...")
        response = requests.post(
            f"{API_V1}/recommendations/user_recommendations",
            json={"user_id": 99999999, "limit": 10},
            timeout=10
        )
        if response.status_code in [404, 400]:
            results.add_result("User Recommendations - Non-existent ID", True, f"HTTP {response.status_code}")
        else:
            results.add_result("User Recommendations - Non-existent ID", False, f"Expected 404/400, got {response.status_code}")
    except Exception as e:
        results.add_result("User Recommendations - Non-existent ID", False, str(e))

    # Test 4: Invalid limit (> 100)
    try:
        print_info("Testing limit > 100...")
        response = requests.post(
            f"{API_V1}/recommendations/similar_movies",
            json={"movie_id": 1, "limit": 500},
            timeout=10
        )
        if response.status_code == 422:
            results.add_result("Recommendations - Limit > 100", True, "422 Validation Error")
        else:
            results.add_result("Recommendations - Limit > 100", False, f"Expected 422, got {response.status_code}")
    except Exception as e:
        results.add_result("Recommendations - Limit > 100", False, str(e))


# ================================================================================
# 9. EDGE CASE TESTS - Data Processing
# ================================================================================

def test_data_processing_edge_cases():
    """Test data processing with edge cases"""
    print_section("ERROR TEST MODULE 9: Data Processing Edge Cases")

    # Test 1: Filter with invalid rating range
    try:
        print_info("Testing filter with rating > 5.0...")
        response = requests.post(
            f"{API_V1}/dataprocess/filter_data",
            json={"min_rating": 6.0, "max_rating": 10.0},
            timeout=10
        )
        if response.status_code in [422, 400]:
            results.add_result("Filter - Rating > 5.0", True, f"HTTP {response.status_code}")
        else:
            results.add_result("Filter - Rating > 5.0", False, f"Expected 422/400, got {response.status_code}")
    except Exception as e:
        results.add_result("Filter - Rating > 5.0", False, str(e))

    # Test 2: Export with invalid format
    try:
        print_info("Testing export with invalid format...")
        response = requests.post(
            f"{API_V1}/dataprocess/export_data",
            json={"dataset": "ratings", "format": "invalid_format"},
            timeout=10
        )
        if response.status_code in [422, 400]:
            results.add_result("Export - Invalid Format", True, f"HTTP {response.status_code}")
        else:
            results.add_result("Export - Invalid Format", False, f"Expected 422/400, got {response.status_code}")
    except Exception as e:
        results.add_result("Export - Invalid Format", False, str(e))


# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main():
    """Run all error tests"""
    print_section("COMPREHENSIVE ERROR TESTING SUITE")
    print_info("Testing invalid inputs, edge cases, and exception handling")
    print_info(f"Target API: {BASE_URL}")

    # Check if server is running
    try:
        response = requests.get(f"{API_V1}/health", timeout=5)
        if response.status_code != 200:
            print_error("Server is not healthy!")
            print_error(f"Please start: python -m uvicorn src.main:app --host 127.0.0.1 --port 8000")
            sys.exit(1)
    except Exception as e:
        print_error(f"Server is not running at {BASE_URL}")
        print_error(f"Error: {str(e)}")
        print_error(f"Please start: python -m uvicorn src.main:app --host 127.0.0.1 --port 8000")
        sys.exit(1)

    print_success("Server is running!")

    # Run all error test modules
    test_top_movies_errors()
    test_user_statistics_errors()
    test_clustering_errors()
    test_anomaly_detection_errors()
    test_rating_sentiment_errors()
    test_trend_analysis_errors()
    test_visualization_errors()
    test_recommendations_errors()
    test_data_processing_edge_cases()

    # Print summary
    all_passed = results.summary()

    if all_passed:
        print("\n✅ All error tests passed! Excellent error handling.")
        sys.exit(0)
    else:
        print("\n⚠️  Some error tests failed. Review error handling logic.")
        sys.exit(1)


if __name__ == "__main__":
    main()
