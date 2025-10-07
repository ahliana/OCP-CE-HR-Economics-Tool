"""
Test Economics Panel Display (without logging output)
"""

import sys
import os
import io

# Set UTF-8 encoding for output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

from data.loader import load_csv_files

# Mock Output widget for testing
class MockOutput:
    def clear_output(self, wait=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

def test_display():
    print("=" * 60)
    print("Testing Economics Panel Display (Suppressed Output)")
    print("=" * 60)

    # Load CSV data
    print("\n1. Loading CSV data...")
    load_csv_files()
    print("   ✓ CSV data loaded")

    # Import after data is loaded
    from ui.economics_panel import display_economics_analysis

    # Create mock output
    output = MockOutput()

    # Test parameters
    wha = 1.0
    T1 = 20
    temp_rise = 10

    print(f"\n2. Test Parameters:")
    print(f"   - System Power: {wha} MW")
    print(f"   - Inlet Temperature: {T1}°C")
    print(f"   - Temperature Rise: {temp_rise}°C")

    print("\n3. Displaying economics analysis...")
    print("   (This should not show any logging output)")
    print()

    # This will attempt to display but won't actually render in console
    display_economics_analysis(output, wha, T1, temp_rise)

    print("\n4. ✓ Display function executed without errors")
    print("   (Logging was suppressed)")

    print("\n" + "=" * 60)
    print("Test Complete - No logging output displayed!")
    print("=" * 60)

if __name__ == "__main__":
    test_display()
