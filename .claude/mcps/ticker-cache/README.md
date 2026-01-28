# Ticker Cache MCP Server

MCP server for stock ticker data. Provides cached access to Yahoo Finance data with resources for discovery and tools for data operations.

## Installation

### Claude Code

```bash
make install
```

This registers the server with Claude Code at user scope.

### Claude Desktop (MCPB Bundle)

Build the MCP bundle:

```bash
make build-mcpb
```

This creates `.claude/dist/ticker-cache-mcp-0.1.0.mcpb` which can be installed in Claude Desktop by double-clicking or dragging to the app.

## Resources

Resources are read-only views of cached data (no auto-fetching):

| URI | Description |
|-----|-------------|
| `ticker://cache` | Summary of all cached tickers |
| `ticker://cache/stats` | Comprehensive statistics (see below) |
| `ticker://ticker/{symbol}` | Full data for a cached ticker |
| `ticker://ticker/{symbol}/metrics` | Just metrics (returns, volatility) |
| `ticker://indexes` | List supported index names |
| `ticker://indexes/{index}` | Cached ticker info for an index |

## Tools

Tools perform actions (fetch, compute, cache):

### `lookup(query: str)`

Unified lookup supporting multiple input types:

```
lookup("AAPL")            # Single ticker
lookup("AAPL,MSFT,GOOGL") # Multiple tickers
lookup("sp500")           # Index name (expands to all tickers)
lookup("Apple")           # Company name (searches and caches matches)
```

### `refresh_metrics(symbols: str)`

Calculate annualized return and volatility from 1-year price history:

```
refresh_metrics("AAPL,MSFT")  # Specific tickers
refresh_metrics("nasdaq100")   # All index tickers
```

## Cache Location

`~/.cache/ticker/tickers.json`

Shared with skills that need ticker data.

## Development

```bash
# Verify server imports
make test

# Run MCP inspector for interactive testing
make inspect

# Run server directly
make run
```

## Build Process

The MCPB bundle build:

1. `make update-deps` - Syncs and exports dependencies to `requirements.txt`
2. `make bundle-deps` - Installs dependencies to `lib/` for bundling
3. `make pack` - Creates the `.mcpb` file using `npx @anthropic-ai/mcpb`

Or run all steps: `make build-mcpb`

## Cleanup

```bash
make clean      # Remove build artifacts
make uninstall  # Remove from Claude Code
```
