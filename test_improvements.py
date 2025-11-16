#!/usr/bin/env python3
import sys
import time
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.services.data_processor import DataProcessor
from src.core.config import settings

def test_vectorized_genre_analysis():
    """Test the vectorized genre analysis performance and correctness"""
    print("=" * 60)
    print("Testing Vectorized Genre Analysis")
    print("=" * 60)

    processor = DataProcessor()

    # Load movies data
    movies_file = Path(settings.data_processed_path) / "movies.csv"
    if not movies_file.exists():
        print(f"❌ Movies file not found: {movies_file}")
        print("Please run the data processing pipeline first")
        return False

    print(f"\n✓ Loading movies data from: {movies_file}")
    df = processor.load_data(str(movies_file))
    print(f"✓ Loaded {len(df)} movies")

    # Test genre analysis
    print("\n📊 Running genre analysis...")
    start_time = time.time()
    stats = processor.aggregate_statistics(df)
    end_time = time.time()

    # Check results
    if 'genre_analysis' in stats:
        genre_analysis = stats['genre_analysis']
        print(f"\n✅ Genre analysis completed in {end_time - start_time:.3f} seconds")
        print(f"   - Total genres: {genre_analysis['total_genres']}")
        print(f"   - Total genre tags: {genre_analysis['total_genre_tags']}")
        print(f"\n   Top 5 genres:")
        for i, (genre, count) in enumerate(list(genre_analysis['top_10_genres'].items())[:5], 1):
            print(f"   {i}. {genre}: {count} movies")
        return True
    else:
        print("❌ Genre analysis not found in statistics")
        return False


def test_export_methods():
    """Test the new export methods"""
    print("\n" + "=" * 60)
    print("Testing Export Methods")
    print("=" * 60)

    processor = DataProcessor()

    # Create a small test dataset
    test_df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Test 1', 'Test 2', 'Test 3'],
        'value': [10.5, 20.3, 30.7]
    })

    # Test CSV export
    print("\n📁 Testing CSV export...")
    csv_path = Path(settings.data_processed_path) / "test_export.csv"
    try:
        result_path = processor.export_to_csv(test_df, str(csv_path), index=False)
        if Path(result_path).exists():
            print(f"✅ CSV export successful: {result_path}")
            # Clean up
            Path(result_path).unlink()
        else:
            print("❌ CSV file not created")
            return False
    except Exception as e:
        print(f"❌ CSV export failed: {e}")
        return False

    # Test JSON export
    print("\n📁 Testing JSON export...")
    json_path = Path(settings.data_processed_path) / "test_export.json"
    try:
        result_path = processor.export_to_json(test_df, str(json_path), orient='records', indent=2)
        if Path(result_path).exists():
            print(f"✅ JSON export successful: {result_path}")
            # Verify JSON content
            with open(result_path, 'r') as f:
                content = f.read()
                if '"id"' in content and '"name"' in content:
                    print("✅ JSON content validated")
            # Clean up
            Path(result_path).unlink()
        else:
            print("❌ JSON file not created")
            return False
    except Exception as e:
        print(f"❌ JSON export failed: {e}")
        return False

    return True


def main():
    """Run all tests"""
    print("\n🚀 Running improvement tests...\n")

    results = []

    # Test 1: Vectorized genre analysis
    results.append(("Vectorized Genre Analysis", test_vectorized_genre_analysis()))

    # Test 2: Export methods
    results.append(("Export Methods", test_export_methods()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(result[1] for result in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
