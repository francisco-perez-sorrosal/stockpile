---
name: ticker
description: Look up stock ticker symbols from company names using Yahoo Finance. Use when researching stocks, needing ticker symbols, validating tickers, or getting company details like sector and market cap.
---

Look up stock ticker symbols by company name or validate existing tickers. Auto-detects whether input is a ticker or company name. Caches results for performance.

## Quick Start

```bash
# Single ticker
python scripts/ticker.py "AAPL"

# Company name (auto-resolved to ticker)
python scripts/ticker.py "Apple"

# Mixed tickers and names
python scripts/ticker.py "AAPL,Microsoft,GOOGL"

# Index tickers (with info)
python scripts/ticker.py --index sp500

# Multiple indexes
python scripts/ticker.py --index "dow,nasdaq100"

# Combine query with index
python scripts/ticker.py "NVDA,Tesla" --index dow

# Skip cache (fresh data)
python scripts/ticker.py "AAPL" --no-cache

# Refresh market data for specific tickers
python scripts/ticker.py "AAPL" --refresh-market

# Refresh market data for ALL cached tickers
python scripts/ticker.py --refresh-market

# Calculate/refresh metrics (returns, volatility)
python scripts/ticker.py "AAPL,MSFT" --refresh-metrics

# Refresh metrics for ALL cached tickers
python scripts/ticker.py --refresh-metrics
```

## Common Tasks

| Task | Command |
|------|---------|
| Look up ticker | `python scripts/ticker.py "AAPL"` |
| Search company | `python scripts/ticker.py "Apple"` |
| Mixed lookups | `python scripts/ticker.py "AAPL,Microsoft,GOOGL"` |
| Index tickers | `python scripts/ticker.py --index sp500` |
| Multiple indexes | `python scripts/ticker.py --index "dow,nasdaq100"` |
| Query + index | `python scripts/ticker.py "NVDA" --index dow` |
| Fresh data | `python scripts/ticker.py "AAPL" --no-cache` |
| Refresh prices | `python scripts/ticker.py "AAPL" --refresh-market` |
| Refresh all cached | `python scripts/ticker.py --refresh-market` |
| Refresh metrics | `python scripts/ticker.py "AAPL" --refresh-metrics` |
| Refresh all metrics | `python scripts/ticker.py --refresh-metrics` |

## Output Format

All outputs are dict of symbol → info:

```json
{
  "AAPL": {"name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics", "returns": 0.127, "volatility": 0.320},
  "MSFT": {"name": "Microsoft Corporation", "sector": "Technology", "industry": "Software—Infrastructure", "returns": 0.100, "volatility": 0.243}
}
```

Metrics fields (`returns`, `volatility`) appear after using `--refresh-metrics`.

## Cache

Ticker info is cached to `~/.cache/ticker/tickers.json` with three refresh policies:

**Static data** (name, sector, industry, exchange): Cached 30 days
**Market data** (price, volume, day_high/low, week_52_high/low): Cached 24 hours
**Metrics data** (returns, volatility): Cached 7 days

- `--no-cache`: Bypass cache entirely, fetch all data fresh
- `--refresh-market`: Keep static data from cache, refresh only market data
- `--refresh-metrics`: Fetch 1-year historical prices and calculate annualized return/volatility
- Cache stores complete ticker info for use by other skills (e.g., `/stock-clusters`)

## Auto-Detection

The script automatically detects whether input is a ticker or company name:

- **Ticker**: Uppercase alphanumeric, ≤10 chars (e.g., `AAPL`, `BRK.B`, `META`)
- **Company name**: Everything else (e.g., `Apple`, `Microsoft Corporation`)

## Dependencies

**No external dependencies.** Uses Python standard library only.

For API endpoint details and exchange codes, see [reference.md](reference.md).
