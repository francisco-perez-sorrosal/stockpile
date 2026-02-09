# Technical Reference

## Return and Volatility Calculation

### Annualized Return

```python
daily_returns = prices.pct_change()
annualized_return = daily_returns.mean() * 252
```

Where 252 is the number of trading days per year (excluding weekends and holidays).

### Annualized Volatility

```python
annualized_volatility = daily_returns.std() * sqrt(252)
```

The square root scaling converts daily standard deviation to annual using the property that variance scales linearly with time (under certain assumptions).

## K-means Clustering

### Algorithm

1. **Data preparation**: Convert return/volatility pairs to numpy array
2. **Whitening**: Normalize data by dividing by standard deviation (equalizes feature importance)
3. **Clustering**: Use `scipy.cluster.vq.kmeans` to find centroids
4. **Assignment**: Use `scipy.cluster.vq.vq` to assign each stock to nearest centroid

### Why scipy instead of sklearn?

- `scipy.cluster.vq` is lighter weight
- Pre-installed in more environments
- Sufficient for basic K-means without advanced features

### Elbow Method

The elbow curve plots distortion (sum of squared distances to centroids) vs. number of clusters:

```python
for k in range(2, 20):
    centroids, _ = kmeans(data, k)
    idx, _ = vq(data, centroids)
    distortion = sum((data - centroids[idx]) ** 2).sum()
```

The "elbow" point where diminishing returns begin suggests optimal k.

## S&P 500 Ticker Source

Tickers are scraped from Wikipedia's S&P 500 constituents table:

```
https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
```

The first table contains current constituents with columns:
- Symbol
- Security (company name)
- GICS Sector
- GICS Sub-Industry

### Ticker Cleaning

Some Wikipedia tickers need adjustment for Yahoo Finance:
- Replace `.` with `-` (e.g., `BRK.B` -> `BRK-B`)
- Remove whitespace and newlines

## Output Formats

### Interactive HTML (Plotly)

Requires `plotly` package. Creates a standalone HTML file with:
- Zoomable scatter plot
- Hover tooltips showing ticker symbols
- Color-coded clusters
- No server required

### Static Image (Matplotlib)

Fallback when Plotly unavailable. Supports:
- PNG (raster, good for web)
- PDF (vector, good for print)
- SVG (vector, good for editing)

### CSV Export

Standard CSV format for further analysis:

```csv
Ticker,Returns,Volatility,Cluster
AAPL,0.234,0.281,1
MSFT,0.187,0.245,1
```

## Error Handling

| Error | Behavior |
|-------|----------|
| Ticker download fails | Skip and continue |
| Insufficient price data (<20 days) | Skip ticker |
| NaN/Inf in calculations | Remove from analysis |
| Plotly not available | Fall back to matplotlib |

## Interpretation Guidelines

### Cluster Characteristics

| Position | Interpretation |
|----------|----------------|
| High return, low volatility | Ideal performers - investigate for fundamentals |
| High return, high volatility | Momentum/growth stocks - higher risk |
| Low return, low volatility | Defensive/dividend stocks - stable |
| Low return, high volatility | Underperformers - avoid or investigate turnarounds |
| Outliers | Special situations - mergers, restructuring, etc. |

### Caveats

- Past performance does not predict future results
- Volatility can indicate both risk and opportunity
- High recent returns may indicate a stock is already overvalued
- Always verify with fundamental analysis

---

Note: Yahoo Finance API documentation (rate limiting, endpoints, curl examples) was removed from this file. That content belongs with the `ticker-cache` MCP server, which handles all Yahoo Finance interactions.
