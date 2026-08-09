"""
This module pulls SPX options data from WRDS OptionMetrics database.
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, "./src")

import pandas as pd
import wrds

import chartbook

from date_config import PULL_CHUNKS, raw_filename

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"

WRDS_USERNAME = chartbook.env.get("WRDS_USERNAME")


def sql_query(year, start, end):
    """
    Build SQL query for a specific year's option data.
    Uses optionm_all schema which has year-specific tables.
    """
    return f"""
        SELECT
            b.secid, b.date,
            b.open, b.close,
            a.cp_flag,
            a.exdate, a.impl_volatility,
            c.dtb3 as tb_m3,
            a.volume, a.open_interest,
            a.best_bid, a.best_offer, a.strike_price, a.contract_size
        FROM
            optionm_all.opprcd{year} AS a
        JOIN
            optionm_all.secprd{year} AS b ON a.date = b.date AND a.secid = b.secid
        JOIN
            frb_all.rates_daily AS c ON c.date = a.date
        WHERE
            (a.secid = 108105)
        AND
            (a.date >= '{start}')
        AND
            (a.date <= '{end}')
    """


def pull_year_range(
    wrds_username: str,
    year_start: int,
    year_end: int,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Pull SPX option data for a range of years.
    Queries each year's table separately and concatenates results.
    """
    print(f">> Connecting to WRDS as {wrds_username}...")
    db = wrds.Connection(wrds_username=wrds_username, verbose=False)

    dlist = []
    for year in range(year_start, year_end + 1):
        t0 = time.time()
        sql = sql_query(year=year, start=start_date, end=end_date)
        dftemp = db.raw_sql(sql, date_cols=["date", "exdate"])
        dlist.append(dftemp)
        t1 = round(time.time() - t0, 2)
        print(f"   {year}: {len(dftemp):,} records ({t1}s)")

    df = pd.concat(dlist, axis=0, ignore_index=True)
    db.close()

    print(f">> Total records: {len(df):,}")
    return df


def clean_optm_data(df):
    """Convert WRDS raw units to the conventions the rest of the pipeline assumes.

    OptionMetrics quotes `strike_price` at 1000x the real strike, and
    `frb_all.rates_daily.dtb3` is a percent. Everything downstream expects a
    real strike and a decimal annual rate. The cached files written by
    `load_optm_data` are pre-clean, so any code reading them directly must call
    this first.
    """
    df = df.copy()
    df["strike_price"] = df["strike_price"] / 1000
    df["tb_m3"] = df["tb_m3"] / 100
    df["tb_m3"] = df["tb_m3"].ffill()
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_optm_data(start_date: str, end_date: str, force_reload: bool = False):
    """
    Load OptionMetrics data, pulling from WRDS if not cached.
    """
    year_start = int(start_date[:4])
    year_end = int(end_date[:4])

    file_path = DATA_DIR / f"data_{start_date[:7]}_{end_date[:7]}.parquet"

    if file_path.exists() and not force_reload:
        print(f">> Reading from cache: {file_path}")
        df = pd.read_parquet(file_path)
    else:
        print(f">> Pulling data from WRDS: {start_date} to {end_date}")
        df = pull_year_range(
            wrds_username=WRDS_USERNAME,
            year_start=year_start,
            year_end=year_end,
            start_date=start_date,
            end_date=end_date,
        )
        df.to_parquet(file_path, index=False)
        print(f">> Saved to {file_path}")

    df = clean_optm_data(df)
    return df


def main():
    """Populate the raw per-chunk cache for every chunk in PULL_CHUNKS.

    No combined file is written. `calc_filters` reads the chunks directly and
    processes them a year at a time, which keeps peak memory bounded now that
    the full sample is ~44M rows. A combined file was also a unit trap: it was
    written post-`clean_optm_data` while the per-chunk caches are pre-clean, so
    two files following the same naming convention held different units.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    for start, end in PULL_CHUNKS:
        print(f"\n=== Pulling {start:%Y-%m} to {end:%Y-%m} ===")
        df = load_optm_data(start_date=str(start), end_date=str(end))
        total += len(df)
        print(f">> {raw_filename(start, end)}: {len(df):,} records")
        del df

    print(f"\n>> Total records across {len(PULL_CHUNKS)} chunks: {total:,}")


if __name__ == "__main__":
    main()
