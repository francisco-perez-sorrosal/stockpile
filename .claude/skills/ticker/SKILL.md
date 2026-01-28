---
name: ticker
description: Look up stock ticker symbols from company names using Yahoo Finance. Use when researching stocks, needing ticker symbols, validating tickers, or getting company details like sector and market cap.
---

Look up stock ticker symbols using the **ticker-cache MCP server**. All data access goes through MCP tools and resources.

## Quick Reference

| Task | MCP Tool |
|------|----------|
| Look up ticker(s) | `lookup("AAPL")` or `lookup("AAPL,MSFT,GOOGL")` |
| Search company name | `lookup("Apple")` |
| Get index tickers | `lookup("sp500")` or `lookup("nasdaq100")` or `lookup("dow")` |
| Calculate metrics | `refresh_metrics("AAPL,MSFT")` |
| Browse cached data | Resource: `ticker://cache` |

## Tools

### `lookup(query)`

Unified lookup supporting multiple input types:

- **Single ticker**: `lookup("AAPL")`
- **Multiple tickers**: `lookup("AAPL,MSFT,GOOGL")`
- **Index name**: `lookup("sp500")`, `lookup("nasdaq100")`, `lookup("dow")`
- **Company name**: `lookup("Apple")` - searches and caches all matches

Returns dict of symbol -> full ticker info.

### `refresh_metrics(symbols)`

Calculate annualized return and volatility from 1-year prices:

```
refresh_metrics("AAPL,MSFT")   # Specific tickers
refresh_metrics("nasdaq100")    # All index tickers
```

Metrics are cached for 7 days.

## Resources

Read-only views of cached data (no auto-fetching):

| Resource | Description |
|----------|-------------|
| `ticker://cache` | Summary of all cached tickers |
| `ticker://cache/stats` | Cache statistics |
| `ticker://ticker/{symbol}` | Full data for a cached ticker |
| `ticker://ticker/{symbol}/metrics` | Just return/volatility |
| `ticker://indexes` | List index names |
| `ticker://indexes/{index}` | Cached info for index tickers |

## Output Format

All tools return JSON dict of symbol -> info:

```json
{
  "AAPL": {
    "symbol": "AAPL",
    "name": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "exchange": "NMS",
    "returns": 0.127,
    "volatility": 0.320
  }
}
```

Metrics fields (`returns`, `volatility`) appear after using `refresh_metrics`.

## Cache

Ticker info cached to `~/.cache/ticker/tickers.json`:

- **Static data** (name, sector): 30 days
- **Market data** (price, volume): 24 hours
- **Metrics** (returns, volatility): 7 days

## Dependencies

Requires **ticker-cache MCP server** to be installed:

```bash
cd .claude/mcps/ticker-cache && make install
```
