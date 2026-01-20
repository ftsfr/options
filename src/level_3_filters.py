"""
Level 3 Filters for options data following CJS (2013) Appendix B.
"""

import numpy as np
import pandas as pd
import datetime


def fit_and_store_curve(group):
    """
    Fit a quadratic curve to the given group of data points and store the fitted values.
    """
    try:
        coefficients = np.polyfit(group["moneyness"], group["log_iv"], 2)
        group["fitted_iv"] = np.polyval(coefficients, group["moneyness"])
    except np.RankWarning:
        print("Polyfit may be poorly conditioned")
    return group


def apply_quadratic_iv_fit(l2_data):
    """
    Apply quadratic curve fitting to the input data.
    """
    l2_data = (
        l2_data.dropna(subset=["moneyness", "log_iv"])
        .groupby(["date", "exdate", "cp_flag"])
        .filter(lambda group: len(group) >= 3)
    )

    l2_data = l2_data.groupby(["date", "exdate", "cp_flag"], group_keys=False).apply(
        fit_and_store_curve
    )

    return l2_data


def calc_relative_distance(series1, series2, method="percent"):
    """
    Calculate the relative distance between two series.
    """
    if method == "percent":
        result = (series1 - series2) / series2 * 100
    elif method == "manhattan":
        result = abs(series1 - series2)
    elif method == "euclidean":
        result = np.sqrt((series1 - series2) ** 2)
    else:
        raise ValueError("Method must be 'percent', 'manhattan', or 'euclidean'")

    result = np.where(np.isinf(result), np.nan, result)
    return result


def iv_filter_outliers(l2_data, iv_distance_method="percent", iv_outlier_threshold=2.0):
    """
    Filter out outliers based on the relative distance of log_iv and fitted_iv.
    """
    l2_data = l2_data.copy()
    l2_data["rel_distance_iv"] = calc_relative_distance(
        l2_data["log_iv"], l2_data["fitted_iv"], method=iv_distance_method
    )

    # Define moneyness bins
    bins = np.arange(0.875, 1.125, 0.025)
    l2_data["moneyness_bin"] = pd.cut(l2_data["moneyness"], bins=bins)

    # Compute standard deviation of relative distances within each moneyness bin
    std_devs = (
        l2_data.groupby("moneyness_bin", observed=True)["rel_distance_iv"]
        .std()
        .reset_index(name="std_dev")
    )

    l2_data["stdev_iv_moneyness_bin"] = l2_data["moneyness_bin"].map(
        std_devs.set_index("moneyness_bin")["std_dev"]
    ).astype(float)

    # Flag outliers based on the threshold
    l2_data["is_outlier_iv"] = l2_data["rel_distance_iv"].abs() > (
        l2_data["stdev_iv_moneyness_bin"].values * iv_outlier_threshold
    )

    # Filter out the outliers
    l3_data_iv_only = l2_data[~l2_data["is_outlier_iv"]]

    return l3_data_iv_only


def build_put_call_pairs(call_options, put_options):
    """
    Builds pairs of call and put options based on the same date, expiration date, and moneyness.
    """
    call_options = call_options.copy()
    put_options = put_options.copy()

    call_options.set_index(["date", "exdate", "moneyness"], inplace=True)
    put_options.set_index(["date", "exdate", "moneyness"], inplace=True)

    common_index = call_options.index.intersection(put_options.index)

    matching_calls = call_options.loc[common_index]
    matching_puts = put_options.loc[common_index]

    return (matching_calls, matching_puts)


def calc_implied_interest_rate_matched(matched_options):
    """
    Calculates the implied interest rate assuming put-call parity.
    """
    try:
        S = matched_options["close_C"]
    except KeyError:
        S = matched_options["close"]

    try:
        K = matched_options["strike_price_C"]
    except KeyError:
        K = matched_options["strike_price"]

    T_inv = np.power(
        (
            matched_options.reset_index()["exdate"]
            - matched_options.reset_index()["date"]
        )
        / datetime.timedelta(days=365),
        -1,
    )
    T_inv.index = matched_options.index

    C_mid = matched_options["mid_price_C"]
    P_mid = matched_options["mid_price_P"]

    matched_options = matched_options.copy()
    matched_options["pc_parity_int_rate"] = np.log((S - C_mid + P_mid) / K) * T_inv
    return matched_options


def pcp_filter_outliers(
    matched_options, int_rate_rel_distance_func="percent", outlier_threshold=2.0
):
    """
    Filters out outliers based on the relative distance of interest rates.
    """
    matched_options = matched_options.copy()
    matched_options["rel_distance_int_rate"] = calc_relative_distance(
        matched_options["pc_parity_int_rate"],
        matched_options["daily_median_rate"],
        method=int_rate_rel_distance_func,
    )
    matched_options["rel_distance_int_rate"] = matched_options[
        "rel_distance_int_rate"
    ].fillna(0.0)

    # Calculate the standard deviation of the relative distances
    stdev_int_rate_rel_distance = matched_options["rel_distance_int_rate"].std()

    # Flag outliers based on the threshold
    matched_options["is_outlier_int_rate"] = (
        matched_options["rel_distance_int_rate"].abs()
        > outlier_threshold * stdev_int_rate_rel_distance
    )

    # Filter out the outliers
    l3_filtered_options = matched_options[~matched_options["is_outlier_int_rate"]]

    # Make the dataframe long-form
    _calls = l3_filtered_options.filter(like="_C").rename(
        columns=lambda x: x.replace("_C", "")
    )
    _puts = l3_filtered_options.filter(like="_P").rename(
        columns=lambda x: x.replace("_P", "")
    )
    l3_filtered_options = pd.concat((_calls, _puts), axis=0)

    return l3_filtered_options


def IV_filter(l2_data, date_range="", data_dir=None):
    """
    Applies log(IV), fits quadratic curve, filters IV outliers.
    """
    print(" >> Running IV filter...")

    l2_data = l2_data.copy()
    l2_data["log_iv"] = np.log(l2_data["IV"])

    print(" |-- IV filter: applying quadratic fit...")
    l2_data = apply_quadratic_iv_fit(l2_data)

    print(" |-- IV filter: filtering outliers...")
    l3_data_iv_only = iv_filter_outliers(l2_data, "percent", 2.0)
    l3_data_iv_only = l3_data_iv_only.copy()
    l3_data_iv_only["moneyness_bin"] = l3_data_iv_only["moneyness_bin"].astype(str)

    if data_dir is not None:
        print(" |-- IV filter: saving L3 IV-filtered data...")
        l3_data_iv_only.to_parquet(
            data_dir / f"L3_IV_filter_only_{date_range}.parquet"
        )

    return l2_data, l3_data_iv_only


def put_call_filter(df, date_range=""):
    """
    Filters option data using the put-call parity filter.
    """
    print(" >> Running PCP filter...")

    df = df.copy()
    df["mid_price"] = (df["best_bid"] + df["best_offer"]) / 2

    # Extract all the call options
    call_options = df[df["cp_flag"] == "C"].copy()
    put_options = df[df["cp_flag"] == "P"].copy()

    print(" |-- PCP filter: building put-call pairs...")
    matching_calls, matching_puts = build_put_call_pairs(
        call_options.reset_index(drop=True), put_options.reset_index(drop=True)
    )

    # Match the puts and calls
    matched_options = pd.merge(
        matching_calls,
        matching_puts,
        on=["date", "exdate", "moneyness"],
        suffixes=("_C", "_P"),
    )

    # Calculate the PCP implied interest rate
    print(" |-- PCP filter: calculating PCP implied interest rate...")
    matched_options = calc_implied_interest_rate_matched(matched_options)

    # Calculate the daily median implied interest rate from the T-Bill data
    daily_median_int_rate = (
        matched_options.groupby("date")["tb_m3_C"]
        .median()
        .reset_index(name="daily_median_rate")
    )
    matched_options = matched_options.join(
        daily_median_int_rate.set_index("date"), on="date"
    )

    print(" |-- PCP filter: filtering outliers...")
    l3_filtered_options = pcp_filter_outliers(matched_options, "percent", 2.0)

    # Build chart
    print(" |-- PCP filter complete.")
    l3_filtered_options = l3_filtered_options.copy()
    l3_filtered_options["log_iv"] = np.log(
        l3_filtered_options["IV"].where(l3_filtered_options["IV"] > 0)
    )

    return l3_filtered_options


def run_filter(df, date_range, iv_only=False, data_dir=None):
    """
    Run the L3 filter on option data.
    """
    print(">> L3 filter running...")

    if iv_only:
        print(" >> Running IV filter only...")
        l2_with_fit, l3_data_iv_only = IV_filter(df, date_range=date_range, data_dir=data_dir)
        l3_filtered_options = None
    else:
        l2_with_fit, l3_data_iv_only = IV_filter(df, date_range=date_range, data_dir=data_dir)
        l3_filtered_options = put_call_filter(l3_data_iv_only, date_range=date_range)

    return l2_with_fit, l3_data_iv_only, l3_filtered_options
