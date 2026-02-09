# Ticker Skill

Look up stock ticker symbols from company names using Yahoo Finance data via the ticker-cache MCP server.

## When to Use

- Looking up ticker symbols for one or more companies
- Validating whether a ticker symbol exists
- Retrieving company details (sector, industry, exchange, market cap)
- Getting tickers for a market index (S&P 500, NASDAQ 100, Dow Jones)
- Calculating annualized return and volatility metrics for stocks

## Activation

Triggered automatically when Claude detects intent to look up stock tickers, research company symbols, or access Yahoo Finance data. Can also be invoked explicitly via `/stockpile:ticker`.

## Skill Contents

| File | Purpose |
|------|---------|
| `SKILL.md` | Skill instructions loaded into Claude's context |
| `reference.md` | Yahoo Finance API endpoint details, exchange codes, quote types |
| `README.md` | This file (human-facing, not loaded into context) |
