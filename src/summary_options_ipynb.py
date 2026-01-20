# %% [markdown]
# # SPX Option Portfolio Returns Summary

# %%
import sys
sys.path.insert(0, "./src")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import chartbook

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"

# %% [markdown]
# ## Data Overview
#
# This pipeline produces leverage-adjusted SPX option portfolio returns following:
# - CJS (2013): 54 portfolios (9 moneyness x 3 maturities x 2 option types)
# - HKM (2017): 18 portfolios (average across maturities)
#
# ### Data Sources
#
# - WRDS OptionMetrics (SPX options, secid=108105)
# - T-Bill rates for risk-free rate

# %% [markdown]
# ## HKM 18 Portfolios

# %%
df_hkm = pd.read_parquet(DATA_DIR / "ftsfr_hkm_option_returns.parquet")
print(f"Shape: {df_hkm.shape}")
print(f"Columns: {df_hkm.columns.tolist()}")
print(f"\nDate range: {df_hkm['ds'].min()} to {df_hkm['ds'].max()}")
print(f"Number of portfolios: {df_hkm['unique_id'].nunique()}")

# %%
df_hkm.describe()

# %%
# Show portfolio IDs
print("HKM Portfolio IDs:")
print(df_hkm['unique_id'].unique())

# %% [markdown]
# ### HKM Portfolio Return Statistics

# %%
# Pivot to wide format for analysis
hkm_wide = df_hkm.pivot(index='ds', columns='unique_id', values='y')
hkm_stats = hkm_wide.describe().T
hkm_stats['skewness'] = hkm_wide.skew()
hkm_stats['kurtosis'] = hkm_wide.kurtosis()
print(hkm_stats[['mean', 'std', 'min', 'max', 'skewness', 'kurtosis']].to_string())

# %%
# Time series plot of select HKM portfolios
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Select representative portfolios
portfolios = [
    ('hkm_C_1000', 'ATM Call'),
    ('hkm_P_1000', 'ATM Put'),
    ('hkm_C_950', 'OTM Call (0.95)'),
    ('hkm_P_1050', 'OTM Put (1.05)')
]

for ax, (port_id, label) in zip(axes.flat, portfolios):
    if port_id in hkm_wide.columns:
        ax.plot(hkm_wide.index, hkm_wide[port_id], alpha=0.7)
        ax.set_title(f'{label} ({port_id})')
        ax.set_ylabel('Monthly Return')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(DATA_DIR.parent / "_output" / "hkm_portfolio_returns.png", dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## CJS 54 Portfolios

# %%
df_cjs = pd.read_parquet(DATA_DIR / "ftsfr_cjs_option_returns.parquet")
print(f"Shape: {df_cjs.shape}")
print(f"Columns: {df_cjs.columns.tolist()}")
print(f"\nDate range: {df_cjs['ds'].min()} to {df_cjs['ds'].max()}")
print(f"Number of portfolios: {df_cjs['unique_id'].nunique()}")

# %%
df_cjs.describe()

# %%
# Show portfolio IDs
print("CJS Portfolio IDs:")
for i, pid in enumerate(sorted(df_cjs['unique_id'].unique())):
    print(pid, end='\t')
    if (i + 1) % 6 == 0:
        print()

# %% [markdown]
# ### CJS Portfolio Return Statistics

# %%
# Pivot to wide format for analysis
cjs_wide = df_cjs.pivot(index='ds', columns='unique_id', values='y')
cjs_stats = cjs_wide.describe().T
cjs_stats['skewness'] = cjs_wide.skew()
cjs_stats['kurtosis'] = cjs_wide.kurtosis()
print(cjs_stats[['mean', 'std', 'min', 'max', 'skewness', 'kurtosis']].round(4).head(20).to_string())

# %% [markdown]
# ### Correlation Heatmap (HKM)

# %%
# Correlation matrix for HKM portfolios
fig, ax = plt.subplots(figsize=(12, 10))
corr = hkm_wide.corr()
sns.heatmap(corr, annot=False, cmap='RdBu_r', center=0, ax=ax)
ax.set_title('HKM Portfolio Correlations')
plt.tight_layout()
plt.savefig(DATA_DIR.parent / "_output" / "hkm_correlation.png", dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Data Definitions
#
# ### HKM Portfolio Returns (ftsfr_hkm_option_returns)
#
# | Variable | Description |
# |----------|-------------|
# | unique_id | Portfolio identifier (e.g., hkm_C_1000 = call, moneyness 1.0) |
# | ds | Month-end date |
# | y | Monthly leverage-adjusted return |
#
# ### CJS Portfolio Returns (ftsfr_cjs_option_returns)
#
# | Variable | Description |
# |----------|-------------|
# | unique_id | Portfolio identifier (e.g., cjs_C_1000_30 = call, moneyness 1.0, 30-day) |
# | ds | Month-end date |
# | y | Monthly leverage-adjusted return |
#
# ### Portfolio Naming Convention
#
# - Format: `{source}_{type}_{moneyness}_{maturity}`
# - type: C (call) or P (put)
# - moneyness: Strike/Spot * 1000 (e.g., 1000 = ATM, 950 = 5% OTM call)
# - maturity: 30, 60, or 90 days (CJS only)
