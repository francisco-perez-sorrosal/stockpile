# stock-clusters

Cluster stocks by annualized return and volatility using K-means to identify high-performers, stable assets, and outliers.

## When to Use

- Exploring investment opportunities across a set of tickers or an index
- Identifying risk profiles (high return/low volatility vs. aggressive growth)
- Comparing market segments or sectors by return/volatility characteristics
- Finding outliers worth investigating in a portfolio or watchlist
- Visualizing how stocks group by performance metrics

## Activation

Triggered by `/stockpile:stock-clusters` or when the user asks about clustering stocks, grouping stocks by return/volatility, or identifying stock risk profiles.

## Skill Contents

| File | Description |
|------|-------------|
| `SKILL.md` | Skill instructions loaded into Claude's context (workflow, MCP usage, script commands) |
| `reference.md` | Technical reference for clustering math, output formats, and interpretation |
| `scripts/stock_clusters.py` | K-means clustering script (reads from ticker cache or JSON data file, outputs console/HTML/CSV) |
| `tests/test_cache_reading.py` | Tests for cache and data file reading |
| `tests/test_cli.py` | Tests for CLI argument parsing and integration |
| `tests/test_clustering.py` | Tests for clustering logic and label generation |

## Related Skills

- `/stockpile:ticker` -- Look up ticker symbols and company details via the ticker-cache MCP server
