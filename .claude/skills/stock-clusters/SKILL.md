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

Analyze S&P 500 stocks (default):

```bash
python scripts/stock_clusters.py
```

Analyze NASDAQ-100:

```bash
python scripts/stock_clusters.py --index nasdaq100
```

Customize clusters and output:

```bash
python scripts/stock_clusters.py --clusters 5 --output clusters.html
```

## Using in Claude

Ask Claude to analyze stock performance:

```
cluster AAPL, MSFT, GOOGL, NVDA, META, NFLX, and IBM by return and volatility
```

```
analyze tech stocks and show me the highest performing ones with low volatility
```

```
cluster NASDAQ-100 stocks by risk and return
```

Claude will run the analysis, generate an interactive scatter plot, and help interpret the clusters.

## Common Tasks

| Task | Command |
|------|---------|
| Analyze specific stocks | `python scripts/stock_clusters.py -t "AAPL,MSFT,GOOGL"` |
| Analyze NASDAQ-100 | `python scripts/stock_clusters.py -i nasdaq100` |
| Analyze Dow Jones | `python scripts/stock_clusters.py -i dow` |
| Default analysis (S&P 500, 5 clusters) | `python scripts/stock_clusters.py` |
| Custom cluster count | `python scripts/stock_clusters.py -k 7` |
| Save interactive chart | `python scripts/stock_clusters.py -o chart.html` |
| Show elbow curve | `python scripts/stock_clusters.py --elbow` |
| Export cluster data | `python scripts/stock_clusters.py --csv results.csv` |
| Test with fewer tickers | `python scripts/stock_clusters.py --limit 50` |
| Slower requests (rate limiting) | `python scripts/stock_clusters.py --delay 2.0` |

## Workflow

### 1. Run the Analysis

```bash
python scripts/stock_clusters.py --clusters 5 --output analysis.html
```

The script:
1. Gets tickers from custom list, or fetches from index (S&P 500, NASDAQ-100, Dow Jones)
2. Downloads 1 year of daily prices from Yahoo Finance
3. Calculates annualized return and volatility for each stock
4. Clusters stocks using K-means
5. Generates an interactive scatter plot

### 2. Interpret the Clusters

The scatter plot shows:
- **X-axis**: Annualized return (higher = better performance)
- **Y-axis**: Annualized volatility (higher = more risk)

Hover over dots to see ticker symbols. Look for:

- **High return, low volatility**: Ideal candidates (upper-left quadrant)
- **High return, high volatility**: High risk/reward (upper-right)
- **Low return, low volatility**: Stable but slow growth (lower-left)
- **Outliers**: May indicate special situations worth investigating

### 3. Find Optimal Cluster Count

Use the elbow method to determine the optimal number of clusters:

```bash
python scripts/stock_clusters.py --elbow
```

The elbow point (where diminishing returns start) suggests the optimal k value.

### 4. Disambiguate Tickers

If you need more information about a specific ticker found in the results, use `/ticker`:

```bash
python ../ticker/scripts/ticker.py --ticker NVDA
```

## Understanding the Metrics

### Annualized Return
Daily returns averaged and scaled to 252 trading days:
```
Return = mean(daily_pct_change) * 252
```

### Annualized Volatility
Standard deviation of daily returns scaled to annual:
```
Volatility = std(daily_pct_change) * sqrt(252)
```

Higher volatility means larger price swings. A stock with 30% volatility might swing 30% up or down in a year under normal conditions.

## Output Format

### Console Output
```
Cluster 0 (45 stocks): Avg Return=12.3%, Avg Volatility=25.1%
Cluster 1 (89 stocks): Avg Return=8.7%, Avg Volatility=18.4%
...

Top performers by return:
  NVDA: Return=125.4%, Volatility=45.2%, Cluster=2
  META: Return=89.3%, Volatility=38.1%, Cluster=2
  ...
```

### CSV Export
```csv
Ticker,Returns,Volatility,Cluster
AAPL,0.234,0.281,1
MSFT,0.187,0.245,1
...
```

### Interactive HTML
Plotly scatter plot with hover tooltips showing ticker details.

## Data Sources

- **Ticker lists**:
  - Custom tickers via `--tickers` argument
  - S&P 500 from Wikipedia (default)
  - NASDAQ-100 from Wikipedia
  - Dow Jones from Wikipedia
- **Price data**: Yahoo Finance Chart API (1 year daily)

## Dependencies

**Required packages** (pre-installed in Claude environments):
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scipy` - K-means clustering (scipy.cluster.vq)
- `matplotlib` - Static plotting (fallback)

**Optional** (for interactive charts):
- `plotly` - Interactive HTML charts

The script uses Yahoo Finance API directly (no `yfinance` package needed).

## Limitations

- 1-year lookback period (fixed)
- Stocks with insufficient price history are skipped
- Yahoo Finance rate limits may affect large batch downloads
- Index tickers are fetched from Wikipedia (may not be 100% up-to-date)

## Rate Limiting

Yahoo Finance has aggressive CDN-level rate limiting. If you see "Edge: Too Many Requests" errors or high failure rates:

1. **Increase delay**: `--delay 2.0` or higher
2. **Test with subset first**: `--limit 50` to verify connectivity
3. **Wait and retry**: Rate limits reset after 10-15 minutes

See [reference.md](reference.md) for technical details on rate limiting mitigations.

## See Also

- `/ticker` - Look up ticker details and disambiguate symbols
- [reference.md](reference.md) - Technical details on Yahoo Finance API and clustering algorithm
