"""
Level 1 Filters for options data following CJS (2013) Appendix B.
"""

import pandas as pd
import numpy as np


def calc_moneyness(df):
    """
    Calculate moneyness as strike price / underlying price.
    """
    df = df.copy()
    df["moneyness"] = df["strike_price"] / df["close"]
    return df


def identical_filter(df):
    """
    Remove duplicate quotes with identical option type, strike price,
    expiration date, and price. Keep only the first occurrence.
    """
    df = df.drop_duplicates(
        subset=["date", "exdate", "cp_flag", "strike_price", "best_bid", "best_offer"],
        keep="first",
    )
    return df


def identical_but_price_filter(df):
    """
    For quotes with identical terms (type, strike, maturity) but different prices,
    keep the quote whose T-bill-based implied volatility is closest to its moneyness neighbors.
    """
    df = df.copy()

    # Find duplicates
    dup_cols = ["date", "exdate", "cp_flag", "strike_price"]
    df["is_dup"] = df.duplicated(subset=dup_cols, keep=False)

    # For non-duplicates, keep them
    non_dups = df[~df["is_dup"]].copy()

    # For duplicates, keep the one with IV closest to neighbors
    dups = df[df["is_dup"]].copy()

    if len(dups) > 0:
        # Group by date, exdate, cp_flag and calculate median IV per moneyness bin
        dups["moneyness_bin"] = pd.cut(
            dups["moneyness"], bins=np.arange(0.5, 1.5, 0.05)
        )

        # For each duplicate set, find the quote with IV closest to the median
        def select_best_iv(group):
            if len(group) == 1:
                return group
            # Handle case where all IV values are NaN
            if group["IV"].isna().all():
                return group.iloc[[0]]
            median_iv = group["IV"].median()
            if pd.isna(median_iv):
                return group.iloc[[0]]
            best_idx = (group["IV"] - median_iv).abs().idxmin()
            if pd.isna(best_idx):
                return group.iloc[[0]]
            return group.loc[[best_idx]]

        dups = dups.groupby(dup_cols, group_keys=False).apply(select_best_iv)

    df = pd.concat([non_dups, dups], ignore_index=True)
    df = df.drop(columns=["is_dup", "moneyness_bin"], errors="ignore")

    return df


def delete_zero_bid_filter(df):
    """
    Remove quotes with a bid price of zero.
    Zero bids indicate low-valued options or censored negative bids.
    """
    df = df[df["best_bid"] > 0]
    return df


def delete_zero_volume_filter(df):
    """
    Remove quotes with zero volume.
    Note: CJS Appendix B does not explicitly detail this filter,
    but it appears in Table B.1.
    """
    df = df[df["volume"] > 0]
    return df


def apply_l1_filters(df, include_volume_filter=False):
    """
    Apply all Level 1 filters to the dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw options data
    include_volume_filter : bool
        Whether to include the zero volume filter (default False per CJS Appendix B)

    Returns:
    --------
    pd.DataFrame
        Filtered options data
    """
    # Preprocess
    df = df.copy()
    df["mid_price"] = (df["best_bid"] + df["best_offer"]) / 2
    df["strike_price"] = df["strike_price"] / 1000  # Adjust strike price
    df = calc_moneyness(df)
    df = df.rename(columns={"IV": "IV"})

    # Apply filters
    df = identical_filter(df)
    df = identical_but_price_filter(df)
    df = delete_zero_bid_filter(df)

    if include_volume_filter:
        df = delete_zero_volume_filter(df)

    return df
