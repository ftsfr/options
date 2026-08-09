"""
Level 1 Filters for options data following CJS (2013) Appendix B.
"""

import pandas as pd
import numpy as np


def calc_moneyness(df):
    """
    Calculate moneyness as strike price / underlying price.
    """
    return df.assign(moneyness=df["strike_price"] / df["close"])


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
    keep the quote whose T-bill-based implied volatility is closest to the group's
    median IV. Ties keep the first-occurring quote; a NaN-IV quote survives only
    if every quote in its group has NaN IV (then the first quote wins).
    """
    # Find duplicates
    dup_cols = ["date", "exdate", "cp_flag", "strike_price"]
    is_dup = df.duplicated(subset=dup_cols, keep=False)

    # For non-duplicates, keep them
    non_dups = df[~is_dup]

    # For duplicates, keep the one with IV closest to the group median. The
    # stable sort on distance + drop_duplicates(keep="first") reproduces
    # idxmin's first-occurrence tie-breaking; the final sort on the group keys
    # keeps the surviving rows in group-key order.
    dups = df[is_dup]

    if len(dups) > 0:
        median_iv = dups.groupby(dup_cols)["IV"].transform("median")
        iv_dist = (dups["IV"] - median_iv).abs().fillna(np.inf)
        dups = (
            dups.assign(_iv_dist=iv_dist)
            .sort_values("_iv_dist", kind="stable")
            .drop_duplicates(subset=dup_cols, keep="first")
            .sort_values(dup_cols, kind="stable")
            .drop(columns="_iv_dist")
        )

    return pd.concat([non_dups, dups], ignore_index=True)


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
