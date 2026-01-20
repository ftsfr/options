"""
This module pulls SPX options data from WRDS OptionMetrics database.
"""

import sys
from pathlib import Path
from datetime import date
import time

sys.path.insert(0, "./src")

import pandas as pd
import wrds

import chartbook

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"

WRDS_USERNAME = chartbook.env.get("WRDS_USERNAME")

# Date ranges for data pulls
START_DATE_01 = date(1996, 1, 1)
END_DATE_01 = date(2012, 1, 31)
START_DATE_02 = date(2012, 2, 1)
END_DATE_02 = date(2019, 12, 31)


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
    """Clean and standardize option data."""
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
    start_ym = start_date[:7]
    end_ym = end_date[:7]

    file_name = f"data_{start_ym}_{end_ym}.parquet"
    file_path = DATA_DIR / file_name

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
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Pull data for both date ranges
    print("=== Pulling 1996-01 to 2012-01 ===")
    df1 = load_optm_data(
        start_date=str(START_DATE_01),
        end_date=str(END_DATE_01),
    )

    print("\n=== Pulling 2012-02 to 2019-12 ===")
    df2 = load_optm_data(
        start_date=str(START_DATE_02),
        end_date=str(END_DATE_02),
    )

    # Combine the two files
    combined_date_range = f"{str(START_DATE_01)[:7]}_{str(END_DATE_02)[:7]}"
    combined_file = DATA_DIR / f"data_{combined_date_range}.parquet"

    print(f"\n>> Combining data files...")
    df_combined = pd.concat([df1, df2], ignore_index=True)
    df_combined.to_parquet(combined_file, index=False)
    print(f">> Saved combined file to {combined_file}")
    print(f">> Total combined records: {len(df_combined):,}")


if __name__ == "__main__":
    main()
