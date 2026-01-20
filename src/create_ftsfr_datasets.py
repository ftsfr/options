"""
Create FTSFR standardized datasets from option portfolio returns.
Outputs: ftsfr_hkm_option_returns.parquet, ftsfr_cjs_option_returns.parquet
"""

import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, "./src")

import polars as pl

import chartbook

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"

# Date ranges
START_DATE_01 = date(1996, 1, 1)
END_DATE_02 = date(2019, 12, 31)
DATE_RANGE = f"{START_DATE_01.strftime('%Y-%m')}_{END_DATE_02.strftime('%Y-%m')}"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load HKM portfolio returns
    hkm_file = DATA_DIR / f"hkm_portfolio_returns_{DATE_RANGE}.parquet"
    if hkm_file.exists():
        print(f">> Loading HKM portfolio returns from {hkm_file}...")
        df_hkm = pl.read_parquet(hkm_file)
        # Rename columns to FTSFR standard
        df_hkm = df_hkm.rename({"ftfsa_id": "unique_id", "return": "y", "date": "ds"})
        df_hkm = df_hkm.select(["unique_id", "ds", "y"])
        df_hkm = df_hkm.drop_nulls()

        # Save FTSFR format
        output_file = DATA_DIR / "ftsfr_hkm_option_returns.parquet"
        df_hkm.write_parquet(output_file)
        print(f">> Saved {output_file}")
        print(f"   Shape: {df_hkm.shape}")
        print(f"   Portfolios: {df_hkm['unique_id'].n_unique()}")
    else:
        print(f">> HKM file not found: {hkm_file}")

    # Load CJS portfolio returns
    cjs_file = DATA_DIR / f"cjs_portfolio_returns_{DATE_RANGE}.parquet"
    if cjs_file.exists():
        print(f">> Loading CJS portfolio returns from {cjs_file}...")
        df_cjs = pl.read_parquet(cjs_file)
        # Rename columns to FTSFR standard
        df_cjs = df_cjs.rename({"ftfsa_id": "unique_id", "return": "y", "date": "ds"})
        df_cjs = df_cjs.select(["unique_id", "ds", "y"])
        df_cjs = df_cjs.drop_nulls()

        # Save FTSFR format
        output_file = DATA_DIR / "ftsfr_cjs_option_returns.parquet"
        df_cjs.write_parquet(output_file)
        print(f">> Saved {output_file}")
        print(f"   Shape: {df_cjs.shape}")
        print(f"   Portfolios: {df_cjs['unique_id'].n_unique()}")
    else:
        print(f">> CJS file not found: {cjs_file}")


if __name__ == "__main__":
    main()
