---
name: stock-clusters
description: Analyze stock tickers by return and volatility using K-means clustering. Use when exploring investment opportunities, identifying risk profiles, comparing market segments, finding high-performers, or analyzing portfolio positioning.
---

Cluster stocks by annualized return and volatility using K-means to identify high-performers, stable assets, and outliers worth investigating.

## Workflow Overview

1. **Get tickers** - Use MCP `lookup` for tickers/indexes, or user provides them
2. **Ensure metrics** - Use MCP `refresh_metrics` to calculate return/volatility
3. **Generate elbow curve** - Run script with `--elbow` to determine optimal k
4. **Analyze elbow image** - View the image to identify the elbow point
5. **Run clustering** - Execute with chosen k value
6. **Present visualization** - Show the clusters.html to user
7. **Interpret results** - Explain cluster characteristics

## Step-by-Step Guide

### Step 1: Get Ticker Symbols

Use the **ticker-cache MCP server** to get tickers:

```
# Index tickers
lookup("sp500")
lookup("nasdaq100")
lookup("dow")

# Specific tickers
lookup("AAPL,MSFT,GOOGL,NVDA,META")

# Company names (auto-resolves)
lookup("Apple,Microsoft,Google")
```

### Step 2: Calculate Metrics via MCP

Ensure return/volatility metrics are calculated:

```
# For specific tickers
refresh_metrics("AAPL,MSFT,GOOGL,NVDA,META")

# For an index
refresh_metrics("nasdaq100")
```

This fetches 1-year price history and calculates annualized metrics.

### Step 3: Generate Elbow Curve

Run the clustering script with `--elbow` to find optimal cluster count:

```bash
python scripts/stock_clusters.py --tickers "AAPL,MSFT,GOOGL,NVDA,META" --elbow --elbow-output elbow.png
```

For index analysis:
```bash
python scripts/stock_clusters.py --index nasdaq100 --elbow --elbow-output elbow.png
```

### Step 4: Analyze the Elbow Curve

**You MUST view the elbow curve image** (`elbow.png`) to determine optimal k.

Look for:
- The **"elbow" point** where the curve bends sharply
- This is where adding more clusters yields diminishing returns
- The k value at the elbow is the optimal cluster count

**Fallback guidelines** (if elbow is unclear):
- Small sets (< 10): k=2 or k=3
- Medium sets (10-50): k=3 to k=5
- Large sets (50+): k=4 to k=7

### Step 5: Run Clustering with Chosen k

```bash
python scripts/stock_clusters.py --tickers "AAPL,MSFT,GOOGL,NVDA,META" --clusters <k> --output clusters.html
```

Replace `<k>` with the elbow value (e.g., `--clusters 3`).

Cluster labels are assigned automatically:
- **Strong Return, Low Vol**: Best risk-adjusted performers
- **Strong Return, High Vol**: High risk/reward opportunities
- **Moderate Return, Low Vol**: Stable performers
- **Low Return, Low Vol**: Defensive stocks
- **Negative Return**: Underperformers

### Step 6: Present the Visualization

**Show `clusters.html` to the user** before interpreting. This allows them to:
- Explore the interactive scatter plot
- Hover over points for stock details
- Form observations before your analysis

### Step 7: Interpret Results

**Scatter plot axes:**
- X-axis: Annualized return (higher = better performance)
- Y-axis: Annualized volatility (higher = more risk)

**Key patterns:**
- Upper-left: High return, low volatility (ideal)
- Upper-right: High return, high volatility (aggressive)
- Lower-left: Low return, low volatility (defensive)
- Outliers: Special situations worth investigating

**Summarize:**
1. Which cluster has best risk-adjusted performers
2. Notable stocks in each cluster
3. Surprising outliers or patterns

## Script Commands

| Task | Command |
|------|---------|
| Analyze custom tickers | `python scripts/stock_clusters.py -t "AAPL,MSFT,GOOGL"` |
| Analyze with elbow | `python scripts/stock_clusters.py -t "..." --elbow --elbow-output elbow.png` |
| Analyze NASDAQ-100 | `python scripts/stock_clusters.py -i nasdaq100` |
| Analyze S&P 500 | `python scripts/stock_clusters.py -i sp500` |
| Analyze Dow Jones | `python scripts/stock_clusters.py -i dow` |
| Set cluster count | `python scripts/stock_clusters.py -t "..." -k 4` |
| Save interactive chart | `python scripts/stock_clusters.py -t "..." -o clusters.html` |
| Export to CSV | `python scripts/stock_clusters.py -t "..." --csv results.csv` |

## Output Formats

### Console

```
=== Cluster Summary ===

Strong Return, Low Vol (2 stocks): Avg Return=45.6%, Avg Volatility=31.8%
Strong Return, High Vol (1 stocks): Avg Return=56.8%, Avg Volatility=45.4%

=== Top Performers by Return ===

  GOOGL (Alphabet Inc.): Return=59.4%, Volatility=32.1% [Strong Return, Low Vol]
```

### CSV Export

```csv
Ticker,Returns,Volatility,Cluster,Name,Sector,ClusterLabel
AAPL,0.127,0.320,1,Apple Inc.,Consumer Electronics,"Strong Return, Low Vol"
```

### Interactive HTML

Plotly scatter plot with hover tooltips showing ticker, company, sector, and metrics.

## Data Flow

```
MCP Server (ticker-cache)          Skill Script
       │                                 │
       │ refresh_metrics("sp500")        │
       ├────────────────────────────────>│
       │                                 │
       │ (writes to cache file)          │
       │                                 │
       │                                 │ reads cache
       │                                 │ clusters with scipy
       │                                 │ generates plotly viz
       │                                 │
       │                                 ├──> elbow.png
       │                                 ├──> clusters.html
       │                                 └──> console output
```

## Dependencies

**MCP Server** (required for data):
```bash
cd .claude/mcps/ticker-cache && make install
```

**Python packages** (pre-installed in Claude environments):
- pandas, numpy, scipy, matplotlib

**Auto-installed** (for interactive charts):
- plotly

## See Also

- `/ticker` - Look up ticker symbols and company details
