---
name: ticker
description: Look up stock ticker symbols from company names using Yahoo Finance. Use when researching stocks, needing ticker symbols, validating tickers, or getting company details like sector and market cap.
---

Look up stock ticker symbols by company name, validate existing tickers, and retrieve company details like sector, industry, and market cap.

## Quick Start

Search for a company's ticker symbol:

```bash
python scripts/ticker.py "Apple"
```

Output:
```json
{
  "status": "ambiguous",
  "query": "Apple",
  "results": [
    {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NMS", "type": "EQUITY", "sector": "Technology"},
    {"symbol": "APLE", "name": "Apple Hospitality REIT, Inc.", "exchange": "NYQ", "type": "EQUITY"}
  ]
}
```

Validate a known ticker:

```bash
python scripts/ticker.py --ticker MSFT
```

## Using in Claude

Simply ask Claude to look up a ticker:

```
find me reliable information about apple's ticker
```

Claude will automatically invoke the ticker skill, search for matches, and handle disambiguation if multiple companies are found.

## Common Tasks

| Task | Command |
|------|---------|
| Search by company name | `python scripts/ticker.py "company name"` |
| Include market cap | `python scripts/ticker.py "company name" --details` |
| Validate/lookup ticker | `python scripts/ticker.py --ticker SYMBOL` |
| Limit results | `python scripts/ticker.py "company" -n 5` |

## Workflow

### 1. User asks for a ticker

Run the search:
```bash
python scripts/ticker.py "Tesla"
```

**If `status` is `found`** (single match): Return the ticker directly.

**If `status` is `ambiguous`** (multiple matches): Present all options to the user and ask them to choose. **Never assume which one they mean**—the user is responsible for disambiguation.

**If `status` is `not_found`**: Try alternative spellings or ask for clarification.

### Disambiguation (IMPORTANT)

When results are ambiguous, you MUST ask the user to select. Common ambiguity cases:

- Same company on different exchanges (e.g., AAPL on NASDAQ vs APC.DU in Frankfurt)
- Companies with similar names (e.g., Apple Inc. vs Apple Hospitality REIT)
- ETFs tracking a company (e.g., leveraged ETFs like NVDX for NVIDIA)

Example prompt to user:
> I found multiple matches for "Apple". Which one did you mean?
> 1. **AAPL** - Apple Inc. (NASDAQ, Technology)
> 2. **APLE** - Apple Hospitality REIT, Inc. (NYSE, Real Estate)

### 2. User asks about a ticker symbol

Validate and get details:
```bash
python scripts/ticker.py --ticker NVDA
```

Returns company name, exchange, sector, and industry.

### 3. User needs detailed company info

Include market cap with `--details`:
```bash
python scripts/ticker.py "Microsoft" --details
```

## Output Format

All responses are JSON with consistent structure:

```json
{
  "status": "found|ambiguous|not_found",
  "query": "the search term",
  "results": [
    {
      "symbol": "AAPL",
      "name": "Apple Inc.",
      "exchange": "NMS",
      "type": "EQUITY",
      "sector": "Technology",
      "industry": "Consumer Electronics",
      "market_cap": "$3.50T"
    }
  ]
}
```

| Field | Description |
|-------|-------------|
| `status` | `found` (1 result), `ambiguous` (multiple), `not_found` (none) |
| `symbol` | Stock ticker symbol |
| `name` | Company name |
| `exchange` | Exchange code (NMS=NASDAQ, NYQ=NYSE, etc.) |
| `type` | Security type (EQUITY, ETF, INDEX) |
| `sector` | Business sector (Technology, Healthcare, etc.) |
| `industry` | Specific industry within sector |
| `market_cap` | Market capitalization (with `--details` or `--ticker`) |

## Dependencies

**No external dependencies.** Uses Python standard library only (`urllib`, `json`, `argparse`).

For API endpoint details and exchange codes, see [reference.md](reference.md).
