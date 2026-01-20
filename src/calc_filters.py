"""
Apply all CJS (2013) filters to SPX options data.
Converts combined_filters.ipynb to a script.
"""

import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, "./src")

import pandas as pd

import chartbook
import level_1_filters as f1
import level_2_filters as f2
import level_3_filters as f3

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"
OUTPUT_DIR = BASE_DIR / "_output"

# Date ranges
START_DATE_01 = date(1996, 1, 1)
END_DATE_01 = date(2012, 1, 31)
START_DATE_02 = date(2012, 2, 1)
END_DATE_02 = date(2019, 12, 31)


def compare_filtered_data(filtered_df, orig_df, filter_name="Filter"):
    """Print summary of filter effect."""
    removed = orig_df.shape[0] - filtered_df.shape[0]
    pct = removed / orig_df.shape[0] if orig_df.shape[0] > 0 else 0
    print(f"| {filter_name}:")
    print(f">> Records removed: {removed:,.0f} out of {orig_df.shape[0]:,.0f} ({pct:.2%})")
    print(f">> Filtered data shape: {filtered_df.shape[0]:,.0f} rows")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    DATE_RANGE = f"{pd.Timestamp(START_DATE_01):%Y-%m}_{pd.Timestamp(END_DATE_02):%Y-%m}"

    # Load combined raw data
    input_file = DATA_DIR / f"data_{DATE_RANGE}.parquet"
    if not input_file.exists():
        print(f">> Input file not found: {input_file}")
        print(">> Please run pull_option_data.py first")
        return

    print(f">> Loading raw data from {input_file}...")
    raw_option_data = pd.read_parquet(input_file)

    # Preprocess
    raw_option_data["mid_price"] = (
        raw_option_data["best_bid"] + raw_option_data["best_offer"]
    ) / 2
    # Note: strike_price already divided by 1000 in pull_option_data.py
    raw_option_data = f1.calc_moneyness(raw_option_data)
    raw_option_data = raw_option_data.rename(columns={"impl_volatility": "IV"})

    print(f">> Raw data shape: {raw_option_data.shape}")

    # === Level 1 Filters ===
    print("\n=== Applying Level 1 Filters ===")

    # Identical filter
    spx_filtered = f1.identical_filter(raw_option_data)
    compare_filtered_data(spx_filtered, raw_option_data, "Identical Filter")

    # Identical except price filter
    spx_filtered_2 = f1.identical_but_price_filter(spx_filtered)
    compare_filtered_data(spx_filtered_2, spx_filtered, "Identical Except Price Filter")

    # Bid = 0 filter
    spx_filtered_3 = f1.delete_zero_bid_filter(spx_filtered_2)
    compare_filtered_data(spx_filtered_3, spx_filtered_2, "Delete Zero Bid Filter")

    # Don't apply volume filter per CJS Appendix B
    spx_l1_filtered = spx_filtered_3.copy()
    spx_l1_filtered.to_parquet(DATA_DIR / f"L1_filtered_{DATE_RANGE}.parquet", index=False)
    print(f">> Saved L1 filtered data to L1_filtered_{DATE_RANGE}.parquet")

    # === Level 2 Filters ===
    print("\n=== Applying Level 2 Filters ===")

    # Days to maturity filter
    spx_filtered_5 = f2.days_to_maturity_filter(spx_l1_filtered, min_days=7, max_days=180)
    compare_filtered_data(spx_filtered_5, spx_l1_filtered, "Days to Maturity Filter")

    # IV range filter
    spx_filtered_6 = f2.iv_range_filter(spx_filtered_5, min_iv=0.05, max_iv=1.0)
    compare_filtered_data(spx_filtered_6, spx_filtered_5, "IV Range Filter")

    # Moneyness filter
    spx_filtered_7 = f2.moneyness_filter(spx_filtered_6, min_moneyness=0.8, max_moneyness=1.2)
    compare_filtered_data(spx_filtered_7, spx_filtered_6, "Moneyness Filter")

    # Implied interest rate filter
    spx_filtered_8 = f2.implied_interest_rate_filter(spx_filtered_7)
    compare_filtered_data(spx_filtered_8, spx_filtered_7, "Implied Interest Rate Filter")

    # Unable to compute IV filter
    spx_filtered_9 = f2.unable_to_compute_iv_filter(spx_filtered_8)
    compare_filtered_data(spx_filtered_9, spx_filtered_8, "Unable to Compute IV Filter")

    spx_l2_filtered = spx_filtered_9.copy()
    spx_l2_filtered.to_parquet(DATA_DIR / f"L2_filtered_{DATE_RANGE}.parquet", index=False)
    print(f">> Saved L2 filtered data to L2_filtered_{DATE_RANGE}.parquet")

    # === Level 3 Filters ===
    print("\n=== Applying Level 3 Filters ===")

    # IV filter
    l2_data_with_fit, spx_filtered_10 = f3.IV_filter(spx_l2_filtered, DATE_RANGE, data_dir=DATA_DIR)
    compare_filtered_data(spx_filtered_10, spx_l2_filtered, "IV Filter")

    # Put-call parity filter
    spx_filtered_11 = f3.put_call_filter(spx_filtered_10, DATE_RANGE)
    compare_filtered_data(spx_filtered_11, spx_filtered_10, "Put-Call Parity Filter")

    spx_l3_filtered = spx_filtered_11.copy()
    spx_l3_filtered.to_parquet(DATA_DIR / f"L3_filtered_{DATE_RANGE}.parquet", index=False)
    print(f">> Saved L3 filtered data to L3_filtered_{DATE_RANGE}.parquet")

    # Save final filtered data
    spx_filtered_final = spx_filtered_11.copy()
    final_savefile = DATA_DIR / f"spx_filtered_final_{DATE_RANGE}.parquet"
    spx_filtered_final.to_parquet(final_savefile, index=True)
    print(f">> Final filtered data saved to {final_savefile}")

    print("\n=== Filter Summary ===")
    print(f"Raw data:     {raw_option_data.shape[0]:,.0f} records")
    print(f"L1 filtered:  {spx_l1_filtered.shape[0]:,.0f} records")
    print(f"L2 filtered:  {spx_l2_filtered.shape[0]:,.0f} records")
    print(f"L3 filtered:  {spx_l3_filtered.shape[0]:,.0f} records")
    total_removed = raw_option_data.shape[0] - spx_l3_filtered.shape[0]
    print(f"Total removed: {total_removed:,.0f} ({total_removed/raw_option_data.shape[0]:.2%})")


if __name__ == "__main__":
    main()
