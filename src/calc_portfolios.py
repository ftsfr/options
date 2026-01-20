"""
Construct CJS (2013) and HKM (2017) option portfolios from filtered data.
Converts portfolios.ipynb to a script.
"""

import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, "./src")

import numpy as np
import pandas as pd
from scipy.stats import norm

import chartbook

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"
OUTPUT_DIR = BASE_DIR / "_output"

# Date ranges
START_DATE_01 = date(1996, 1, 1)
END_DATE_01 = date(2012, 1, 31)
START_DATE_02 = date(2012, 2, 1)
END_DATE_02 = date(2019, 12, 31)


def parse_interval_string(s):
    """Parse string interval like '(0.9, 0.95]' into pandas Interval."""
    if pd.isnull(s) or not isinstance(s, str):
        return pd.NA
    s = s.strip().replace("(", "").replace("]", "")
    try:
        left, right = map(float, s.split(","))
        return pd.Interval(left, right, closed="right")
    except ValueError:
        return pd.NA


def kernel_weights(m_grid, ttm_grid, k_s, ttm, bw_m=0.0125, bw_t=10):
    """Gaussian kernel weights for moneyness and time to maturity."""
    m_grid = np.asarray(m_grid, dtype=float)
    ttm_grid = np.asarray(ttm_grid, dtype=float)
    x = (m_grid - k_s) / bw_m
    y = (ttm_grid - ttm) / bw_t
    dist_sq = x**2 + y**2
    weights = np.exp(-0.5 * dist_sq)
    return weights / weights.sum() if weights.sum() > 0 else np.zeros_like(weights)


def calc_option_delta_elasticity(df):
    """Calculate BSM delta and elasticity for options."""
    df = df.copy()

    T = df["days_to_maturity"].dt.days / 365.0
    S = df["close"]
    K = df["strike_price"]
    r = df["tb_m3"] / 100
    sigma = df["IV"]
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    df["option_delta"] = np.where(df["cp_flag"] == "C", norm.cdf(d1), norm.cdf(d1) - 1)
    df["option_elasticity"] = df["option_delta"] * df["close"] / df["mid_price"]

    return df


def calc_kernel_weights(spx_mod):
    """Calculate kernel weights for each option based on moneyness and maturity targets."""
    moneyness_targets = [0.90, 0.925, 0.950, 0.975, 1.000, 1.025, 1.050, 1.075, 1.100]
    maturity_targets = [30, 60, 90]
    cp_flags = ["C", "P"]

    spx_mod = spx_mod.copy()
    spx_mod["days_to_maturity_int"] = spx_mod["days_to_maturity"].dt.days
    spx_mod = spx_mod.reset_index()
    spx_mod["original_index"] = spx_mod.index

    weight_results = []

    for cp_flag in cp_flags:
        for target_moneyness in moneyness_targets:
            for target_ttm in maturity_targets:
                candidate_options = spx_mod[
                    (spx_mod["cp_flag"] == cp_flag)
                    & (spx_mod["moneyness_id"] == target_moneyness)
                    & (spx_mod["maturity_id"] == target_ttm)
                ].copy()

                if candidate_options.empty:
                    continue

                candidate_options["kernel_weight"] = np.nan

                for date_val, g in candidate_options.groupby("date"):
                    idx = g.index
                    weights = kernel_weights(
                        g["moneyness"].values,
                        g["days_to_maturity_int"].values,
                        k_s=target_moneyness,
                        ttm=target_ttm,
                    )
                    candidate_options.loc[idx, "kernel_weight"] = weights

                weight_results.append(
                    candidate_options[["original_index", "kernel_weight"]]
                )

    if weight_results:
        all_weights = pd.concat(weight_results).set_index("original_index")
        spx_mod.set_index("original_index", inplace=True)
        spx_mod["kernel_weight"] = all_weights["kernel_weight"]
        spx_mod.reset_index(inplace=True)

    spx_mod.drop(columns=["original_index"], inplace=True, errors="ignore")
    return spx_mod


def compute_cjs_return_leverage_investment(spx_mod):
    """Compute CJS leverage-adjusted portfolio returns."""
    df = spx_mod.copy()
    df = df.sort_values(["ftfsa_id", "date"])

    # Lag price
    df["mid_price_lag"] = df.groupby("ftfsa_id")["mid_price"].shift(1)

    # Return and daily risk-free rate
    df["option_return"] = (df["mid_price"] - df["mid_price_lag"]) / df["mid_price_lag"]
    df["daily_rf"] = df["tb_m3"] / 100 / 252

    # Weighted dollar investment and return contribution
    df["inv_weight"] = df["kernel_weight"] / df["option_elasticity"]
    df["inv_return"] = df["inv_weight"] * df["option_return"]

    # Group and aggregate
    grouped = df.groupby(["date", "ftfsa_id"])

    port = grouped.agg(
        total_inv_weight=("inv_weight", "sum"),
        total_inv_return=("inv_return", "sum"),
        daily_rf=("daily_rf", "first"),
        cp_flag=("cp_flag", "first"),
    ).reset_index()

    # Apply CJS logic
    def adjusted_return(row):
        if row["cp_flag"] == "C":
            return row["total_inv_return"] + (1 - row["total_inv_weight"]) * row["daily_rf"]
        elif row["cp_flag"] == "P":
            return -row["total_inv_return"] + (1 + row["total_inv_weight"]) * row["daily_rf"]
        else:
            return np.nan

    port["portfolio_return"] = port.apply(adjusted_return, axis=1)

    return port


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    DATE_RANGE = f"{pd.Timestamp(START_DATE_01):%Y-%m}_{pd.Timestamp(END_DATE_02):%Y-%m}"

    # Load filtered data
    source_file = DATA_DIR / f"spx_filtered_final_{DATE_RANGE}.parquet"
    if not source_file.exists():
        print(f">> Input file not found: {source_file}")
        print(">> Please run calc_filters.py first")
        return

    print(f">> Loading filtered data from {source_file}...")
    spx_filtered = pd.read_parquet(source_file)
    spx_filtered = spx_filtered.reset_index()

    # Restore moneyness_bin as intervals
    if "moneyness_bin" in spx_filtered.columns:
        spx_filtered["moneyness_bin"] = spx_filtered["moneyness_bin"].apply(parse_interval_string)

    # Create moneyness ID from the moneyness_bin column
    spx_filtered["moneyness_id"] = spx_filtered["moneyness_bin"].apply(
        lambda x: x.right if pd.notnull(x) else np.nan
    )
    spx_filtered = spx_filtered.dropna(subset=["moneyness_id"])

    print(f">> Filtered data shape: {spx_filtered.shape}")

    # === Build FTFSA ID for each portfolio ===
    print(">> Building portfolio IDs...")
    maturity_id = pd.concat(
        (
            abs(spx_filtered["days_to_maturity"].dt.days - 30),
            abs(spx_filtered["days_to_maturity"].dt.days - 60),
            abs(spx_filtered["days_to_maturity"].dt.days - 90),
        ),
        axis=1,
    )
    maturity_id.columns = [30, 60, 90]
    spx_filtered["maturity_id"] = maturity_id.idxmin(axis=1)
    spx_filtered["ftfsa_id"] = (
        spx_filtered["cp_flag"]
        + "_"
        + (spx_filtered["moneyness_id"] * 1000)
        .apply(lambda x: str(int(x)) if pd.notnull(x) and x == int(x) else str(x))
        + "_"
        + spx_filtered["maturity_id"].astype(str)
    )

    spx_filtered.set_index(["ftfsa_id", "date"], inplace=True)

    portfolio_ids = spx_filtered.index.get_level_values("ftfsa_id").unique()
    print(f">> Number of portfolios: {len(portfolio_ids)}")

    # === Calculate option elasticity and kernel weights ===
    print(">> Calculating option elasticity and kernel weights...")
    spx_mod = spx_filtered.copy()
    spx_mod = calc_option_delta_elasticity(spx_mod)
    spx_mod = calc_kernel_weights(spx_mod)

    # Remove options with weights lower than 1%
    spx_mod = spx_mod[spx_mod["kernel_weight"] >= 0.01].reset_index(drop=True)
    print(f">> Options after weight filter: {len(spx_mod):,}")

    # === Calculate portfolio returns ===
    print(">> Calculating portfolio returns...")
    portfolio_returns = compute_cjs_return_leverage_investment(spx_mod)

    portfolio_returns.set_index(["date", "ftfsa_id"], inplace=True)
    daily_returns = portfolio_returns.pivot_table(
        index="date", columns="ftfsa_id", values="portfolio_return"
    )

    # === Compound daily returns to monthly (CJS 54 portfolios) ===
    print(">> Compounding daily returns to monthly...")
    cjs_returns = daily_returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
    cjs_returns = cjs_returns.reset_index().melt(
        id_vars="date", var_name="ftfsa_id", value_name="return"
    )
    cjs_returns["ftfsa_id"] = "cjs_" + cjs_returns["ftfsa_id"]
    cjs_returns = cjs_returns[["ftfsa_id", "date", "return"]].set_index(["ftfsa_id", "date"])

    cjs_output = DATA_DIR / f"cjs_portfolio_returns_{DATE_RANGE}.parquet"
    cjs_returns.to_parquet(cjs_output, index=True)
    print(f">> Saved CJS portfolio returns to {cjs_output}")
    print(f">> CJS portfolios: {cjs_returns.index.get_level_values('ftfsa_id').nunique()}")

    # === Construct HKM 18 portfolios ===
    print(">> Constructing HKM 18 portfolios...")
    hkm_returns = cjs_returns.copy().reset_index()
    hkm_returns["type"] = hkm_returns["ftfsa_id"].apply(lambda x: x.split("_")[1])
    hkm_returns["moneyness_id"] = hkm_returns["ftfsa_id"].apply(lambda x: x.split("_")[2])
    hkm_returns["maturity_id"] = hkm_returns["ftfsa_id"].apply(lambda x: x.split("_")[3])
    hkm_returns.drop(columns=["ftfsa_id"], inplace=True)
    hkm_returns.set_index(["date", "type", "moneyness_id", "maturity_id"], inplace=True)
    hkm_returns = hkm_returns.groupby(["date", "type", "moneyness_id"]).mean()
    hkm_returns["ftfsa_id"] = (
        "hkm_"
        + hkm_returns.index.get_level_values("type")
        + "_"
        + hkm_returns.index.get_level_values("moneyness_id")
    )
    hkm_returns = (
        hkm_returns.reset_index()
        .drop(columns=["type", "moneyness_id"])
        .set_index(["ftfsa_id", "date"])
        .sort_index()
    )

    hkm_output = DATA_DIR / f"hkm_portfolio_returns_{DATE_RANGE}.parquet"
    hkm_returns.to_parquet(hkm_output, index=True)
    print(f">> Saved HKM portfolio returns to {hkm_output}")
    print(f">> HKM portfolios: {hkm_returns.index.get_level_values('ftfsa_id').nunique()}")

    print("\n=== Portfolio Construction Complete ===")


if __name__ == "__main__":
    main()
