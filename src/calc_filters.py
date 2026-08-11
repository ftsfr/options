"""
Apply all CJS (2013) filters to SPX options data.
Converts combined_filters.ipynb to a script.

The sample is ~44M raw rows, so the chain runs in two stages:

1. Per-year: the Level 1 filters and the row-wise Level 2 filters. Every filter
   here is either row-wise or keyed on a subset containing `date`, so a
   partition that keeps each trading day intact gives identical results to a
   single pass. Peak memory is one chunk plus one year rather than the whole
   panel.
2. Global, after concatenation: `implied_interest_rate_filter` (medians by
   maturity across the sample) and all of Level 3 (IV outlier thresholds are
   standard deviations by moneyness bin, and the put-call parity threshold is a
   standard deviation, both computed over the whole panel). These cannot be
   partitioned without changing their thresholds.
"""

import gc
import sys
from pathlib import Path

sys.path.insert(0, "./src")

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Copy-on-write makes rename/assign lazy, so the filter stages below don't
# each duplicate the frame. Peak memory would otherwise exceed 25 GB.
pd.options.mode.copy_on_write = True

import chartbook
import level_1_filters as f1
import level_2_filters as f2
import level_3_filters as f3
from date_config import (
    DATE_RANGE,
    FINAL_FILENAME,
    L1_FILENAME,
    L2_FILENAME,
    L3_FILENAME,
    PULL_CHUNKS,
    raw_filename,
)
from pull_option_data import clean_optm_data

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"
OUTPUT_DIR = BASE_DIR / "_output"


def preprocess(df):
    """Derived columns the filters expect. Assumes `clean_optm_data` has run."""
    df["mid_price"] = (df["best_bid"] + df["best_offer"]) / 2
    df = f1.calc_moneyness(df)
    return df.rename(columns={"impl_volatility": "IV"})


def filter_partition(df):
    """Level 1 plus the row-wise Level 2 filters for one partition.

    Returns the post-L1 frame (a pipeline output in its own right) and the
    post-row-wise-L2 frame, along with per-stage row counts.
    """
    counts = {"raw": len(df)}

    df = f1.identical_filter(df)
    df = f1.identical_but_price_filter(df)
    df = f1.delete_zero_bid_filter(df)
    l1 = df
    counts["l1"] = len(l1)

    df = f2.days_to_maturity_filter(l1, min_days=7, max_days=180)
    counts["after_dtm"] = len(df)
    df = f2.iv_range_filter(df, min_iv=0.05, max_iv=1.0)
    counts["after_iv_range"] = len(df)
    df = f2.moneyness_filter(df, min_moneyness=0.8, max_moneyness=1.2)
    counts["after_moneyness"] = len(df)

    return l1, df, counts


class IncrementalParquet:
    """Append row groups to one parquet file without holding the whole frame.

    The L1 output is roughly 600 MB across the full sample; streaming it keeps
    it out of memory while still producing the single file the pipeline
    declares as a target.
    """

    def __init__(self, path):
        self.path = path
        self._writer = None
        self._schema = None

    def write(self, df):
        if df.empty:
            return
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._writer is None:
            self._schema = table.schema
            self._writer = pq.ParquetWriter(self.path, self._schema)
        elif not table.schema.equals(self._schema):
            table = table.cast(self._schema)
        self._writer.write_table(table)

    def close(self):
        if self._writer is not None:
            self._writer.close()


def run_partitioned_stage():
    """Stage 1: per-year L1 and row-wise L2 over every raw chunk."""
    l1_writer = IncrementalParquet(DATA_DIR / L1_FILENAME)
    l2_parts = []
    per_year = []

    for start, end in PULL_CHUNKS:
        path = DATA_DIR / raw_filename(start, end)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing raw chunk {path}. Run src/pull_option_data.py first."
            )

        print(f"\n>> Loading {path.name} ...")
        # The per-chunk cache is pre-clean: strike_price is 1000x and tb_m3 is
        # a percent. Clean the whole chunk before partitioning, because the
        # ffill of tb_m3 is order dependent and must not see year boundaries
        # that the original two-chunk pipeline did not have.
        chunk = clean_optm_data(pd.read_parquet(path))
        chunk = preprocess(chunk)
        print(f"   {len(chunk):,} raw rows")

        for year, sub in chunk.groupby(chunk["date"].dt.year, sort=True):
            l1, l2, counts = filter_partition(sub)
            l1_writer.write(l1)
            l2_parts.append(l2)
            counts["year"] = year
            per_year.append(counts)
            print(
                f"   {year}: raw {counts['raw']:>9,} "
                f"-> L1 {counts['l1']:>9,} -> L2 row-wise {counts['after_moneyness']:>9,}"
            )
            del l1, l2, sub

        del chunk
        gc.collect()

    l1_writer.close()
    print(f"\n>> Saved L1 filtered data to {L1_FILENAME}")

    df = pd.concat(l2_parts, ignore_index=True)
    del l2_parts
    gc.collect()
    return df, pd.DataFrame(per_year)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Stage 1: per-year Level 1 and row-wise Level 2 ===")
    df, per_year = run_partitioned_stage()
    n_raw = int(per_year["raw"].sum())
    n_l1 = int(per_year["l1"].sum())
    print(f"\n>> Post row-wise Level 2: {len(df):,} rows")

    print("\n=== Stage 2: global Level 2 filters ===")
    before = len(df)
    df = f2.implied_interest_rate_filter(df)
    print(f">> Implied interest rate filter: {before:,} -> {len(df):,}")

    before = len(df)
    df = f2.unable_to_compute_iv_filter(df)
    print(f">> Unable to compute IV filter:  {before:,} -> {len(df):,}")

    n_l2 = len(df)
    df.to_parquet(DATA_DIR / L2_FILENAME, index=False)
    print(f">> Saved L2 filtered data to {L2_FILENAME}")

    print("\n=== Stage 3: global Level 3 filters ===")
    before = len(df)
    l2_with_fit, df = f3.IV_filter(df, DATE_RANGE, data_dir=DATA_DIR)
    del l2_with_fit  # only returned for run_filter API compatibility; ~1 GB
    gc.collect()
    print(f">> IV filter: {before:,} -> {len(df):,}")

    before = len(df)
    df = f3.put_call_filter(df, DATE_RANGE)
    print(f">> Put-call parity filter: {before:,} -> {len(df):,}")

    n_l3 = len(df)
    df.to_parquet(DATA_DIR / L3_FILENAME, index=False)
    print(f">> Saved L3 filtered data to {L3_FILENAME}")

    final_savefile = DATA_DIR / FINAL_FILENAME
    df.to_parquet(final_savefile, index=True)
    print(f">> Final filtered data saved to {final_savefile}")

    # Per-year survival rates: the report compares 1996-2019 against 2020-2024,
    # so how much of each year the filters remove is a diagnostic the reader
    # needs in order to judge whether the two periods are comparable.
    per_year["l1_survival"] = (per_year["l1"] / per_year["raw"]).round(4)
    per_year["iv_cap_dropped"] = per_year["after_dtm"] - per_year["after_iv_range"]
    per_year["iv_cap_share"] = (
        per_year["iv_cap_dropped"] / per_year["after_dtm"]
    ).round(4)
    per_year.to_csv(OUTPUT_DIR / f"filter_diagnostics_{DATE_RANGE}.csv", index=False)
    print(f">> Wrote per-year filter diagnostics to _output/filter_diagnostics_{DATE_RANGE}.csv")

    print("\n=== Filter Summary ===")
    print(f"Raw data:     {n_raw:,.0f} records")
    print(f"L1 filtered:  {n_l1:,.0f} records")
    print(f"L2 filtered:  {n_l2:,.0f} records")
    print(f"L3 filtered:  {n_l3:,.0f} records")
    total_removed = n_raw - n_l3
    print(f"Total removed: {total_removed:,.0f} ({total_removed/n_raw:.2%})")


if __name__ == "__main__":
    main()
