"""Test valve cost calculation"""
import sys
sys.path.insert(0, 'python')

from data.loader import load_csv_files, get_csv_data, is_csv_loaded
from data.converter import universal_float_convert
from core.original_calculations import get_PipeSize_Suggested

# Load data
load_csv_files()

# Test data
F1 = 1503.6

# Get pipe size
pipe_size = get_PipeSize_Suggested(F1)
print(f"Pipe size for F1={F1}: {pipe_size}")
print(f"Pipe size as int: {int(pipe_size)}")
print(f"Pipe size as str: {str(int(pipe_size))}")

# Check CVALV
print("\n" + "="*60)
print("CVALV DATA:")
print("="*60)

if is_csv_loaded('CVALV'):
    cvalv_df = get_csv_data('CVALV')
    print(f"CVALV loaded: {cvalv_df is not None}")
    if cvalv_df is not None:
        print(f"CVALV shape: {cvalv_df.shape}")
        print(f"CVALV columns: {cvalv_df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(cvalv_df.head(10))

        # Try to find pipe size 150
        pipe_size_str = str(int(pipe_size))
        print(f"\nLooking for pipe size: '{pipe_size_str}'")

        cvalv_df = cvalv_df.copy()
        print(f"\nColumn 0 values (first 10):")
        for idx, row in cvalv_df.head(10).iterrows():
            col0_value = str(row.iloc[0]).strip()
            print(f"  Row {idx}: '{col0_value}' (type: {type(row.iloc[0])})")

        # Try matching
        for idx, row in cvalv_df.iterrows():
            col0_str = str(row.iloc[0]).strip()
            if col0_str == pipe_size_str:
                print(f"\nMATCH FOUND at row {idx}!")
                print(f"  Column 0: {row.iloc[0]}")
                print(f"  Column 1 (raw): {row.iloc[1]}")
                cvalv_df.iloc[:, 1] = cvalv_df.iloc[:, 1].apply(universal_float_convert)
                print(f"  Column 1 (converted): {cvalv_df.iloc[idx, 1]}")
                break
        else:
            print("\nNO MATCH FOUND")

# Check IVALV
print("\n" + "="*60)
print("IVALV DATA:")
print("="*60)

if is_csv_loaded('IVALV'):
    ivalv_df = get_csv_data('IVALV')
    print(f"IVALV loaded: {ivalv_df is not None}")
    if ivalv_df is not None:
        print(f"IVALV shape: {ivalv_df.shape}")
        print(f"IVALV columns: {ivalv_df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(ivalv_df.head(10))

        # Try to find pipe size 150
        pipe_size_str = str(int(pipe_size))
        print(f"\nLooking for pipe size: '{pipe_size_str}'")

        ivalv_df = ivalv_df.copy()
        print(f"\nColumn 0 values (first 10):")
        for idx, row in ivalv_df.head(10).iterrows():
            col0_value = str(row.iloc[0]).strip()
            print(f"  Row {idx}: '{col0_value}' (type: {type(row.iloc[0])})")

        # Try matching
        for idx, row in ivalv_df.iterrows():
            col0_str = str(row.iloc[0]).strip()
            if col0_str == pipe_size_str:
                print(f"\nMATCH FOUND at row {idx}!")
                print(f"  Column 0: {row.iloc[0]}")
                print(f"  Column 1 (raw): {row.iloc[1]}")
                ivalv_df.iloc[:, 1] = ivalv_df.iloc[:, 1].apply(universal_float_convert)
                print(f"  Column 1 (converted): {ivalv_df.iloc[idx, 1]}")
                break
        else:
            print("\nNO MATCH FOUND")
