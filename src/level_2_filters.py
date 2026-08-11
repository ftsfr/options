"""
Level 2 Filters for options data following CJS (2013) Appendix B.
"""

import pandas as pd
import numpy as np


def calc_days_to_maturity(df):
    """
    Calculate days to maturity as exdate - date.
    """
    return df.assign(days_to_maturity=df["exdate"] - df["date"])


def days_to_maturity_filter(df, min_days=7, max_days=180):
    """
    Remove options with fewer than min_days or more than max_days to expiration.
    """
    df = calc_days_to_maturity(df)
    df = df[
        (df["days_to_maturity"] >= pd.Timedelta(days=min_days))
        & (df["days_to_maturity"] <= pd.Timedelta(days=max_days))
    ]
    return df


def iv_range_filter(df, min_iv=0.05, max_iv=1.00):
    """
    Filter options based on implied volatility range.
    Default is 5% to 100% (0.05 to 1.00).
    """
    df = df[(df["IV"] >= min_iv) & (df["IV"] <= max_iv)]
    return df


def moneyness_filter(df, min_moneyness=0.8, max_moneyness=1.2):
    """
    Filter options based on moneyness range.
    Default is 0.8 to 1.2.
    Moneyness is defined as the ratio of the option's strike price to the underlying price.
    """
    if "moneyness" not in df.columns:
        df = df.copy()
        df["moneyness"] = df["strike_price"] / df["close"]

    df = df[
        (df["moneyness"] > min_moneyness) & (df["moneyness"] < max_moneyness)
    ].reset_index(drop=True)
    return df


def calc_implied_interest_rate(matched):
    """
    Calculate implied interest rate from put-call parity.

    For a European index option paying a dividend yield q,

        C - P = S e^{-qT} - K e^{-rT}   =>   S - C + P = S(1 - e^{-qT}) + K e^{-rT}

    so the rate is  r = -log((S - C + P) / K) / T. The leading minus sign is
    what makes this a rate rather than its negative; without it the expression
    returns approximately q - r, which is anti-correlated with the actual short
    rate (-0.99 against the 3-month T-bill across 1996-2024).

    Note the result is r - q, not r, because no dividend adjustment is
    available here. Callers must therefore not compare it against zero or
    against a T-bill quote; see `implied_interest_rate_filter`.
    """
    import datetime

    S = matched["close_C"] if "close_C" in matched.columns else matched["close"]
    K = (
        matched["strike_price_C"]
        if "strike_price_C" in matched.columns
        else matched["strike_price"]
    )

    T_inv = np.power(
        (matched.reset_index()["exdate"] - matched.reset_index()["date"])
        / datetime.timedelta(days=365),
        -1,
    )
    T_inv.index = matched.index

    C_mid = matched["mid_price_C"]
    P_mid = matched["mid_price_P"]

    matched = matched.copy()
    matched["pc_parity_int_rate"] = -np.log((S - C_mid + P_mid) / K) * T_inv
    return matched


def implied_interest_rate_filter(df, outlier_threshold=2.0):
    """
    Remove quotes whose put-call parity implied rate is out of line with the
    other quotes observed on the same day.

    This previously removed quotes whose implied rate was negative. That test
    does not survive a change of rate regime. `calc_implied_interest_rate`
    returns r - q rather than r, so a zero threshold asks whether the short
    rate exceeds the dividend yield, which is a macroeconomic question rather
    than a data quality one. Measured over 1996-2024 it removed 95-98% of
    matched pairs in 1996-2000 and 99.7% in 2024, when the short rate was near
    5%, against 1.5-5% during the years when it was near zero. Correcting only
    the sign inverts the problem and deletes the zero-rate years instead.

    Referencing each quote to the median implied rate on the same day removes
    the regime, because the level of rates is common to the cross-section on
    any given day. That flags a stable 3.2-6.1% of pairs in every year of the
    sample. It is also the test CJS already apply at Level 3, where quotes are
    compared to a daily reference rather than to a fixed threshold.
    """
    df = df.assign(mid_price=(df["best_bid"] + df["best_offer"]) / 2)

    # Split calls and puts
    calls = df[df["cp_flag"] == "C"]
    puts = df[df["cp_flag"] == "P"]

    # Match by date, exdate, moneyness
    calls.set_index(["date", "exdate", "moneyness"], inplace=True)
    puts.set_index(["date", "exdate", "moneyness"], inplace=True)
    common = calls.index.intersection(puts.index)
    c, p = calls.loc[common].reset_index(), puts.loc[common].reset_index()

    # Merge and compute implied interest rate
    matched = pd.merge(
        c, p, on=["date", "exdate", "moneyness"], suffixes=("_C", "_P")
    )
    matched = calc_implied_interest_rate(matched)

    # Deviation from the same day's median implied rate, in rate units. An
    # absolute deviation rather than a relative one, because the reference
    # crosses zero whenever the dividend yield meets the short rate.
    daily_median = matched.groupby("date")["pc_parity_int_rate"].transform("median")
    deviation = matched["pc_parity_int_rate"] - daily_median
    is_outlier = deviation.abs() > outlier_threshold * deviation.std()

    outliers = matched[is_outlier][
        ["date", "exdate", "strike_price_C", "close_C"]
    ].drop_duplicates()
    df = df.merge(
        outliers,
        left_on=["date", "exdate", "strike_price", "close"],
        right_on=["date", "exdate", "strike_price_C", "close_C"],
        how="outer",
        indicator=True,
    )
    df = df[df["_merge"] == "left_only"].drop(
        columns=["_merge", "strike_price_C", "close_C"]
    )

    # Impute missing rates using median from ATM calls. Conditioning on a
    # non-negative rate here would reintroduce the regime dependence, since
    # r - q is legitimately negative whenever the short rate sits below the
    # dividend yield, so the surviving quotes are used instead.
    atm = matched[matched["moneyness"].between(0.95, 1.05) & ~is_outlier]
    if "days_to_maturity_C" in matched.columns:
        med = (
            atm.groupby("days_to_maturity_C")["pc_parity_int_rate"]
            .median()
            .reset_index()
        )
        df = df.merge(
            med, left_on="days_to_maturity", right_on="days_to_maturity_C", how="left"
        )
        df["pc_parity_int_rate"] = df["pc_parity_int_rate"].ffill()
        df.drop(columns="days_to_maturity_C", inplace=True, errors="ignore")

    return df


def unable_to_compute_iv_filter(df):
    """
    Removes options where the time value is negative and IV cannot be computed.
    Time value = market price - intrinsic value.
    For calls: intrinsic = max(S - K, 0)
    For puts:  intrinsic = max(K - S, 0)
    """
    df = df.assign(mid_price=(df["best_bid"] + df["best_offer"]) / 2)

    # Float, not int: the assignments below write floats into this column, and
    # under pandas 3 an int64 column will raise rather than silently upcast.
    df["intrinsic"] = 0.0
    call_mask = df["cp_flag"] == "C"
    put_mask = df["cp_flag"] == "P"
    df.loc[call_mask, "intrinsic"] = (
        df.loc[call_mask, "close"] - df.loc[call_mask, "strike_price"]
    ).clip(lower=0)
    df.loc[put_mask, "intrinsic"] = (
        df.loc[put_mask, "strike_price"] - df.loc[put_mask, "close"]
    ).clip(lower=0)

    # Filter out rows where time value is negative
    df = df[df["mid_price"] >= df["intrinsic"]]

    return df


def apply_l2_filters(df):
    """Apply all level 2 filters to the dataframe."""
    df = days_to_maturity_filter(df)
    df = iv_range_filter(df)
    df = moneyness_filter(df)
    df = implied_interest_rate_filter(df)
    df = unable_to_compute_iv_filter(df)
    return df
