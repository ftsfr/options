"""
This module provides Black-Scholes-Merton option pricing functions
and implied volatility calculations.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# === Black-Scholes-Merton Pricing ===


def bsm_call_price(S, K, T, r, sigma):
    """
    Calculate Black-Scholes-Merton call option price.

    Parameters:
    -----------
    S : float or array
        Current stock price
    K : float or array
        Strike price
    T : float or array
        Time to expiration (in years)
    r : float or array
        Risk-free interest rate
    sigma : float or array
        Volatility

    Returns:
    --------
    float or array
        Call option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def bsm_put_price(S, K, T, r, sigma):
    """
    Calculate Black-Scholes-Merton put option price.

    Parameters:
    -----------
    S : float or array
        Current stock price
    K : float or array
        Strike price
    T : float or array
        Time to expiration (in years)
    r : float or array
        Risk-free interest rate
    sigma : float or array
        Volatility

    Returns:
    --------
    float or array
        Put option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price


def bsm_price(S, K, T, r, sigma, option_type="call"):
    """
    Calculate Black-Scholes-Merton option price.

    Parameters:
    -----------
    S : float or array
        Current stock price
    K : float or array
        Strike price
    T : float or array
        Time to expiration (in years)
    r : float or array
        Risk-free interest rate
    sigma : float or array
        Volatility
    option_type : str
        'call' or 'put'

    Returns:
    --------
    float or array
        Option price
    """
    if option_type.lower() == "call":
        return bsm_call_price(S, K, T, r, sigma)
    elif option_type.lower() == "put":
        return bsm_put_price(S, K, T, r, sigma)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


# === Greeks ===


def bsm_delta(S, K, T, r, sigma, option_type="call"):
    """
    Calculate Black-Scholes-Merton delta.

    Parameters:
    -----------
    S : float or array
        Current stock price
    K : float or array
        Strike price
    T : float or array
        Time to expiration (in years)
    r : float or array
        Risk-free interest rate
    sigma : float or array
        Volatility
    option_type : str
        'call' or 'put'

    Returns:
    --------
    float or array
        Option delta
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type.lower() == "call":
        return norm.cdf(d1)
    elif option_type.lower() == "put":
        return norm.cdf(d1) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def bsm_elasticity(S, K, T, r, sigma, option_type="call"):
    """
    Calculate Black-Scholes-Merton elasticity (omega).
    Elasticity = Delta * S / Price

    Parameters:
    -----------
    S : float or array
        Current stock price
    K : float or array
        Strike price
    T : float or array
        Time to expiration (in years)
    r : float or array
        Risk-free interest rate
    sigma : float or array
        Volatility
    option_type : str
        'call' or 'put'

    Returns:
    --------
    float or array
        Option elasticity
    """
    delta = bsm_delta(S, K, T, r, sigma, option_type)
    price = bsm_price(S, K, T, r, sigma, option_type)
    return delta * S / price


# === Implied Volatility ===


def bsm_implied_vol(
    market_price, S, K, T, r, option_type="call", lower=0.001, upper=5.0, tol=1e-8
):
    """
    Calculate implied volatility using Brent's method.

    Parameters:
    -----------
    market_price : float
        Observed market price
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    r : float
        Risk-free interest rate
    option_type : str
        'call' or 'put'
    lower : float
        Lower bound for volatility search
    upper : float
        Upper bound for volatility search
    tol : float
        Tolerance for convergence

    Returns:
    --------
    float
        Implied volatility, or NaN if not found
    """
    if T <= 0 or market_price <= 0:
        return np.nan

    def objective(sigma):
        return bsm_price(S, K, T, r, sigma, option_type) - market_price

    try:
        # Check bounds
        f_lower = objective(lower)
        f_upper = objective(upper)

        if f_lower * f_upper > 0:
            return np.nan

        iv = brentq(objective, lower, upper, xtol=tol)
        return iv
    except (ValueError, RuntimeError):
        return np.nan


def calc_implied_vol_vectorized(df, price_col="mid_price", option_type_col="cp_flag"):
    """
    Calculate implied volatility for a DataFrame of options.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: close (underlying), strike_price, days_to_maturity, tb_m3
    price_col : str
        Column name for option price
    option_type_col : str
        Column name for option type (C/P)

    Returns:
    --------
    pd.Series
        Implied volatilities
    """
    import pandas as pd

    results = []
    for idx, row in df.iterrows():
        try:
            option_type = "call" if row[option_type_col] == "C" else "put"
            T = row["days_to_maturity"].days / 365.0 if hasattr(row["days_to_maturity"], "days") else row["days_to_maturity"] / 365.0
            iv = bsm_implied_vol(
                market_price=row[price_col],
                S=row["close"],
                K=row["strike_price"],
                T=T,
                r=row["tb_m3"] / 100,
                option_type=option_type,
            )
            results.append(iv)
        except Exception:
            results.append(np.nan)

    return pd.Series(results, index=df.index)
