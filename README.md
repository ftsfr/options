# SPX Options Pipeline

This pipeline constructs leverage-adjusted SPX option portfolio returns following:
- **CJS (2013)**: Constantinides, Jackwerth, Savov - 54 portfolios
- **HKM (2017)**: He, Kelly, Manela - 18 portfolios

## Data Source

- WRDS OptionMetrics (SPX options, secid=108105)
- T-Bill rates for risk-free rate calculations

## Outputs

- `ftsfr_hkm_option_returns.parquet`: HKM 18 portfolio monthly returns
- `ftsfr_cjs_option_returns.parquet`: CJS 54 portfolio monthly returns

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up credentials:
   ```bash
   cp .env.example .env
   # Edit .env with your WRDS username
   ```

3. Run the pipeline:
   ```bash
   doit
   ```

4. View the generated documentation in `docs/index.html`

## Data Coverage

- Time period: January 1996 - December 2019
- Frequency: Monthly
- Underlying: S&P 500 Index (SPX)

## Portfolio Construction

### CJS 54 Portfolios

54 portfolios = 9 moneyness levels × 3 maturities × 2 option types

- **Moneyness**: 0.90, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.10
- **Maturities**: 30, 60, 90 days to expiration
- **Types**: Calls and Puts

### HKM 18 Portfolios

18 portfolios = average of CJS portfolios across maturities

### Data Filtration (CJS Appendix B)

**Level 1 Filters:**
- Remove identical quotes
- Remove zero bid quotes

**Level 2 Filters:**
- Days to maturity: 7-180 days
- Implied volatility: 5-100%
- Moneyness: 0.8-1.2

**Level 3 Filters:**
- IV outlier filter (quadratic fit)
- Put-call parity filter

## Academic References

### Primary Papers

- **Constantinides, Jackwerth, and Savov (2013)** - "The Puzzle of Index Option Returns"
  - Review of Asset Pricing Studies
  - Methodology for 54-portfolio construction with leverage adjustment

- **He, Kelly, and Manela (2017)** - "Intermediary Asset Pricing: New Evidence from Many Asset Classes"
  - Journal of Financial Economics 126.1 (2017): 1-35
  - 18-portfolio averaging methodology

### Key Findings Replicated

- Crisis-related factors significantly reduce option return pricing errors
- Daily adjustment for beta, maturity, and moneyness reduces variance/skewness
- Short-maturity OTM puts are particularly sensitive to market conditions
