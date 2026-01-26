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

## Output Format

All outputs are dict of symbol → info:

```json
{
  "AAPL": {"name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics"},
  "MSFT": {"name": "Microsoft Corporation", "sector": "Technology", "industry": "Software—Infrastructure"}
}
```

## Cache

Ticker info is cached to `~/.cache/ticker/tickers.json` with two refresh policies:

**Static data** (name, sector, industry, exchange): Cached 30 days
**Market data** (price, volume, day_high/low, week_52_high/low): Cached 24 hours

- `--no-cache`: Bypass cache entirely, fetch all data fresh
- `--refresh-market`: Keep static data from cache, refresh only market data
- Cache stores complete ticker info for future calculations and database migration

## Auto-Detection

The script automatically detects whether input is a ticker or company name:

- **Ticker**: Uppercase alphanumeric, ≤10 chars (e.g., `AAPL`, `BRK.B`, `META`)
- **Company name**: Everything else (e.g., `Apple`, `Microsoft Corporation`)

## Dependencies

**No external dependencies.** Uses Python standard library only.

For API endpoint details and exchange codes, see [reference.md](reference.md).
