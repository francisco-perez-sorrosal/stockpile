# Ticker Cache MCP Server

MCP server for stock ticker data. Provides cached access to Yahoo Finance data with resources for discovery and tools for data operations.

## Installation

### Claude Code (via plugin)

The MCP server auto-starts when the Stockpile plugin is active. No manual installation needed -- `.mcp.json` at the project root configures auto-start.

### Claude Code (manual)

```bash
make install
```

This registers the server with Claude Code at user scope.

### Claude Desktop (MCPB Bundle)

Build the MCP bundle from the project root:

```bash
make mcp-pack
```

This creates `dist/ticker-cache-mcp-<version>.mcpb` which can be installed in Claude Desktop by double-clicking or dragging to the app.

## Module Structure

The server is decomposed into focused modules:

| Module | Responsibility |
|--------|---------------|
| `app.py` | Shared instances (FastMCP, TickerCache, CLI args) |
| `main.py` | Entry point, side-effect imports for registration |
| `resources.py` | MCP resource handlers |
| `tools.py` | MCP tool handlers |
| `models.py` | Pydantic models (MarketData, MetricsData, TickerInfo) |
| `cache.py` | TickerCache class and operation stats |
| `constants.py` | URLs, paths, refresh policies |
| `http_helpers.py` | HTTP fetch helpers (JSON, HTML) |
| `yahoo.py` | Yahoo Finance API functions |
| `scraping.py` | Wikipedia index scraping |

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

# Run server directly (stdio transport)
make run
```

Or from the project root:

```bash
make mcp-test       # Verify imports
make mcp-run        # Run server
make mcp-inspect    # Interactive inspector
```

### Running Tests

```bash
uv run --with pytest pytest tests/ -v
```

## Remote Access (HTTP)

The server supports HTTP transports for remote access:

```bash
# Start HTTP server on port 8000 (SSE transport)
make run-http

# Or with custom host/port
uv run main.py --transport sse --host 0.0.0.0 --port 9000

# Streamable HTTP transport (newer protocol)
uv run main.py --transport streamable-http --port 8000
```

**Endpoints:**
- SSE: `http://HOST:PORT/sse`
- Streamable HTTP: `http://HOST:PORT/mcp`

## Build Process

The MCPB bundle uses `uv` for dependency management at runtime:

1. `make sync-deps` - Syncs dependencies and creates `uv.lock`
2. `make pack` - Creates the `.mcpb` file

Or run all steps: `make build-mcpb`

From the project root: `make mcp-pack`

## Cleanup

```bash
make clean      # Remove build artifacts
make uninstall  # Remove from Claude Code
```
