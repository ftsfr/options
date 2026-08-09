"""
Apply all CJS (2013) filters to SPX options data.
Converts combined_filters.ipynb to a script.
"""

import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, "./src")

import pandas as pd

# Copy-on-write makes rename/assign lazy, so the filter stages below don't
# each duplicate the ~19M-row frame. Peak memory would otherwise exceed 25 GB.
pd.options.mode.copy_on_write = True

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
    df = pd.read_parquet(input_file)

    # Preprocess
    df["mid_price"] = (df["best_bid"] + df["best_offer"]) / 2
    # Note: strike_price already divided by 1000 in pull_option_data.py
    df = f1.calc_moneyness(df)
    df = df.rename(columns={"impl_volatility": "IV"})

    print(f">> Raw data shape: {df.shape}")
    n_raw = len(df)

    # Only `df` (current stage) and `prev` (stage before it) are kept alive;
    # each rebinding frees the earlier frames so the whole run fits in laptop
    # RAM. Row counts for the final summary are captured as plain ints.

    # === Level 1 Filters ===
    print("\n=== Applying Level 1 Filters ===")

    # Identical filter
    prev = df
    df = f1.identical_filter(prev)
    compare_filtered_data(df, prev, "Identical Filter")

    # Identical except price filter
    prev = df
    df = f1.identical_but_price_filter(prev)
    compare_filtered_data(df, prev, "Identical Except Price Filter")

    # Bid = 0 filter
    prev = df
    df = f1.delete_zero_bid_filter(prev)
    compare_filtered_data(df, prev, "Delete Zero Bid Filter")
    del prev

    # Don't apply volume filter per CJS Appendix B
    n_l1 = len(df)
    df.to_parquet(DATA_DIR / f"L1_filtered_{DATE_RANGE}.parquet", index=False)
    print(f">> Saved L1 filtered data to L1_filtered_{DATE_RANGE}.parquet")

    # === Level 2 Filters ===
    print("\n=== Applying Level 2 Filters ===")

    # Days to maturity filter
    prev = df
    df = f2.days_to_maturity_filter(prev, min_days=7, max_days=180)
    compare_filtered_data(df, prev, "Days to Maturity Filter")

    # IV range filter
    prev = df
    df = f2.iv_range_filter(prev, min_iv=0.05, max_iv=1.0)
    compare_filtered_data(df, prev, "IV Range Filter")

    # Moneyness filter
    prev = df
    df = f2.moneyness_filter(prev, min_moneyness=0.8, max_moneyness=1.2)
    compare_filtered_data(df, prev, "Moneyness Filter")

    # Implied interest rate filter
    prev = df
    df = f2.implied_interest_rate_filter(prev)
    compare_filtered_data(df, prev, "Implied Interest Rate Filter")

    # Unable to compute IV filter
    prev = df
    df = f2.unable_to_compute_iv_filter(prev)
    compare_filtered_data(df, prev, "Unable to Compute IV Filter")
    del prev

    n_l2 = len(df)
    df.to_parquet(DATA_DIR / f"L2_filtered_{DATE_RANGE}.parquet", index=False)
    print(f">> Saved L2 filtered data to L2_filtered_{DATE_RANGE}.parquet")

    # === Level 3 Filters ===
    print("\n=== Applying Level 3 Filters ===")

    # IV filter
    prev = df
    l2_with_fit, df = f3.IV_filter(prev, DATE_RANGE, data_dir=DATA_DIR)
    del l2_with_fit  # only returned for run_filter API compatibility; ~1 GB
    compare_filtered_data(df, prev, "IV Filter")

    # Put-call parity filter
    prev = df
    df = f3.put_call_filter(prev, DATE_RANGE)
    compare_filtered_data(df, prev, "Put-Call Parity Filter")
    del prev

    n_l3 = len(df)
    df.to_parquet(DATA_DIR / f"L3_filtered_{DATE_RANGE}.parquet", index=False)
    print(f">> Saved L3 filtered data to L3_filtered_{DATE_RANGE}.parquet")

    # Save final filtered data
    final_savefile = DATA_DIR / f"spx_filtered_final_{DATE_RANGE}.parquet"
    df.to_parquet(final_savefile, index=True)
    print(f">> Final filtered data saved to {final_savefile}")

    print("\n=== Filter Summary ===")
    print(f"Raw data:     {n_raw:,.0f} records")
    print(f"L1 filtered:  {n_l1:,.0f} records")
    print(f"L2 filtered:  {n_l2:,.0f} records")
    print(f"L3 filtered:  {n_l3:,.0f} records")
    total_removed = n_raw - n_l3
    print(f"Total removed: {total_removed:,.0f} ({total_removed/n_raw:.2%})")


if __name__ == "__main__":
    main()
