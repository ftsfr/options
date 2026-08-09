"""Single source of truth for the pipeline's sample period.

Every module derives its dates and output filenames from here. Before this
existed, `START_DATE_01` through `END_DATE_02` were redeclared independently in
four modules and the resulting date range was hardcoded into twelve filenames in
`dodo.py`, so extending the sample meant editing sixteen places and any missed
one produced a silent filename mismatch rather than an error.

To change the sample period, edit `START_DATE`, `END_DATE`, and `PULL_CHUNKS`.
"""

from datetime import date

START_DATE = date(1996, 1, 1)
END_DATE = date(2024, 12, 31)

# WRDS is queried in chunks: `optionm_all` stores one table per year and the
# full range does not fit in memory as a single frame. Chunk boundaries also
# define the raw cache filenames, so changing them invalidates that cache.
PULL_CHUNKS = [
    (date(1996, 1, 1), date(2012, 1, 31)),
    (date(2012, 2, 1), date(2019, 12, 31)),
    (date(2020, 1, 1), date(2024, 12, 31)),
]

YEARS = list(range(START_DATE.year, END_DATE.year + 1))


def ym(d):
    """Format a date as the YYYY-MM stamp used throughout the filenames."""
    return f"{d:%Y-%m}"


def range_tag(start, end):
    """Filename tag for a date range, e.g. '1996-01_2024-12'."""
    return f"{ym(start)}_{ym(end)}"


DATE_RANGE = range_tag(START_DATE, END_DATE)


def raw_filename(start, end):
    """Name of the raw per-chunk WRDS cache file for one pull chunk.

    Note these files hold data as WRDS returns it: `strike_price` is 1000x the
    real strike and `tb_m3` is a percent, not a decimal. `clean_optm_data`
    converts both, and callers reading these files directly must apply it.
    """
    return f"data_{range_tag(start, end)}.parquet"


RAW_FILENAMES = [raw_filename(s, e) for s, e in PULL_CHUNKS]

L1_FILENAME = f"L1_filtered_{DATE_RANGE}.parquet"
L2_FILENAME = f"L2_filtered_{DATE_RANGE}.parquet"
L3_FILENAME = f"L3_filtered_{DATE_RANGE}.parquet"
L3_IV_ONLY_FILENAME = f"L3_IV_filter_only_{DATE_RANGE}.parquet"
FINAL_FILENAME = f"spx_filtered_final_{DATE_RANGE}.parquet"
CJS_FILENAME = f"cjs_portfolio_returns_{DATE_RANGE}.parquet"
HKM_FILENAME = f"hkm_portfolio_returns_{DATE_RANGE}.parquet"
