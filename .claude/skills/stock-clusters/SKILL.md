---
name: stock-clusters
description: Analyze stock tickers by return and volatility using K-means clustering. Use when exploring investment opportunities, identifying risk profiles, comparing market segments, finding high-performers, or analyzing portfolio positioning.
---

Cluster stocks by annualized return and volatility using K-means to identify high-performers, stable assets, and outliers worth investigating. Works with custom tickers or major market indexes (S&P 500, NASDAQ-100, Dow Jones).

## Quick Start

Analyze specific stocks:

```bash
python scripts/stock_clusters.py --tickers "AAPL,MSFT,GOOGL,NVDA,META"
```

## Using in Claude

Ask Claude to analyze stocks:

```
cluster Apple, Microsoft, Google, Nvidia, and Meta by return and volatility
```

```
analyze these tech stocks and show me high performers with low risk: AAPL, MSFT, GOOGL, NVDA, META
```

```
cluster NASDAQ-100 stocks and identify the best risk-adjusted performers
```

Claude will:
1. Convert company names to tickers using `/ticker` if needed
2. Run the clustering analysis
3. Generate visualizations
4. Help interpret the clusters

## Workflow

### Step 1: Get Ticker Symbols

**If the user provides company names** (not ticker symbols), use `/ticker` to convert them:

```bash
python ../ticker/scripts/ticker.py "Apple"
```

Handle disambiguation if multiple matches are found—ask the user to choose.

**If the user provides ticker symbols directly**, proceed to Step 2.

**If analyzing an index**, the script fetches tickers automatically:

```bash
python scripts/stock_clusters.py --index nasdaq100
```

### Step 2: Load Price Data and Calculate Metrics

Run the script to download price data and calculate return/volatility metrics:

```bash
python scripts/stock_clusters.py --tickers "AAPL,MSFT,GOOGL,NVDA,META" --elbow --elbow-output elbow.png
```

The script:
1. Downloads 1 year of daily prices from Yahoo Finance
2. Calculates **annualized return**: `mean(daily_returns) * 252`
3. Calculates **annualized volatility**: `std(daily_returns) * sqrt(252)`
4. Generates the elbow curve and saves it to `elbow.png`

### Step 3: View and Analyze the Elbow Curve

**You MUST view the elbow curve image** (`elbow.png`) to determine the optimal number of clusters.

The elbow curve shows distortion (y-axis) vs number of clusters k (x-axis). Look for:
- The **"elbow" point**—where the curve bends sharply
- This is where adding more clusters yields diminishing returns
- The k value at the elbow is the optimal cluster count

**After viewing the curve, identify:**
1. The k value where the elbow occurs
2. Report this value before proceeding to Step 4

**Fallback guidelines** (if elbow is unclear):
- Small ticker sets (< 10): Use k=2 or k=3
- Medium sets (10-50): Use k=3 to k=5
- Large sets (50+): Use k=4 to k=7

### Step 4: Create Clusters with the Determined k

Run with the k value you identified from the elbow curve:

```bash
python scripts/stock_clusters.py --tickers "AAPL,MSFT,GOOGL,NVDA,META" --clusters <k> --output clusters.html
```

Replace `<k>` with the elbow value (e.g., `--clusters 3` if the elbow was at k=3)

The script automatically labels each cluster based on its characteristics:
- **Strong Return, Low Vol**: Best risk-adjusted performers
- **Strong Return, High Vol**: High risk/reward opportunities
- **Moderate Return, Low Vol**: Stable performers
- **Low Return, Low Vol**: Defensive stocks
- **Negative Return**: Underperformers to investigate

### Step 5: Present the Cluster Visualization

**You MUST present the `clusters.html` file to the user** before interpreting the results.

This allows the user to:
- See the interactive scatter plot
- Hover over data points to explore individual stocks
- Form their own observations before your interpretation

### Step 6: Interpret the Cluster Visualization

After the user has seen the plot, provide your interpretation:

**Scatter plot axes:**
- **X-axis**: Annualized return (higher = better performance)
- **Y-axis**: Annualized volatility (higher = more risk)

**Hover info shows:**
- Ticker symbol and company name
- Industry/sector classification
- Exact return and volatility percentages
- Cluster profile label

**Key patterns to identify:**
- Upper-left quadrant: High return, low volatility (ideal candidates)
- Upper-right quadrant: High return, high volatility (aggressive/risky)
- Lower-left quadrant: Low return, low volatility (defensive/stable)
- Outliers: May indicate special situations worth investigating

**Summarize for the user:**
1. Which cluster contains the best risk-adjusted performers
2. Notable stocks in each cluster
3. Any surprising outliers or patterns

### Step 7: Get More Details on Specific Stocks

For deeper analysis of any ticker found in the results, use `/ticker`:

```bash
python ../ticker/scripts/ticker.py "NVDA" --details
```

This provides company name, sector, industry, and market cap.

## Common Tasks

| Task | Command |
|------|---------|
| Analyze specific stocks | `python scripts/stock_clusters.py -t "AAPL,MSFT,GOOGL"` |
| Analyze with elbow curve | `python scripts/stock_clusters.py -t "..." --elbow` |
| Analyze NASDAQ-100 | `python scripts/stock_clusters.py -i nasdaq100` |
| Analyze Dow Jones | `python scripts/stock_clusters.py -i dow` |
| Analyze S&P 500 (default) | `python scripts/stock_clusters.py` |
| Custom cluster count | `python scripts/stock_clusters.py -t "..." -k 4` |
| Save interactive chart | `python scripts/stock_clusters.py -t "..." -o chart.html` |
| Export cluster data | `python scripts/stock_clusters.py -t "..." --csv results.csv` |
| Skip company info (faster) | `python scripts/stock_clusters.py -t "..." --no-info` |

## Output Format

### Console Output

```
=== Cluster Summary ===

Strong Return, Low Vol (2 stocks): Avg Return=45.6%, Avg Volatility=31.8%
Strong Return, High Vol (1 stocks): Avg Return=56.8%, Avg Volatility=45.4%
Low Return, Low Vol (4 stocks): Avg Return=5.7%, Avg Volatility=31.7%

=== Top Performers by Return ===

  GOOGL (Alphabet Inc.): Return=59.4%, Volatility=32.1% [Strong Return, Low Vol]
  NVDA (NVIDIA Corporation): Return=56.8%, Volatility=45.4% [Strong Return, High Vol]
  ...
```

### CSV Export

```csv
Ticker,Returns,Volatility,Cluster,Name,Sector,ClusterLabel
AAPL,0.127,0.320,1,Apple Inc.,Consumer Electronics,"Strong Return, Low Vol"
MSFT,0.100,0.243,0,Microsoft Corporation,Software—Infrastructure,"Low Return, Low Vol"
```

### Interactive HTML

Plotly scatter plot with hover tooltips showing ticker, company name, sector, and metrics.

## Data Sources

- **Ticker lists**: Custom input, or Wikipedia (S&P 500, NASDAQ-100, Dow Jones)
- **Price data**: Yahoo Finance Chart API (1 year daily)
- **Company info**: Via `/ticker` skill (batch lookup)

## Dependencies

**Skill dependency:**
- `/ticker` - Required for company name/sector lookup

**Python packages** (pre-installed in Claude environments):
- `pandas`, `numpy`, `scipy`, `matplotlib`

**Auto-installed** (for interactive charts):
- `plotly` - Automatically installed on first use if not present

No `yfinance` package needed—uses Yahoo Finance API directly.

## Rate Limiting

Yahoo Finance has aggressive rate limiting. If you see errors:

1. **Increase delay**: `--delay 2.0` or higher
2. **Test with subset first**: `--limit 50`
3. **Wait and retry**: Rate limits reset after 10-15 minutes

See [reference.md](reference.md) for technical details.

## See Also

- `/ticker` - Look up ticker symbols and company details
- [reference.md](reference.md) - Yahoo Finance API and clustering algorithm details
