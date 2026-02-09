"""MCP resource handlers for read-only views of cached ticker data.

Resources:
    ticker://cache              - Summary of all cached tickers
    ticker://cache/stats        - Cache statistics
    ticker://ticker/{symbol}    - Full data for a cached ticker
    ticker://ticker/{symbol}/metrics - Just metrics for a cached ticker
    ticker://indexes            - List of supported index names
    ticker://indexes/{index}    - All cached ticker info for an index
"""

import json

from app import cache, mcp
from constants import INDEX_ALIASES, INDEX_URLS
from scraping import fetch_index_tickers


@mcp.resource("ticker://cache")
def get_cache() -> str:
    """List all cached tickers with summary info."""
    all_tickers = cache.all()
    return json.dumps(
        {s: info.to_summary() for s, info in all_tickers.items()}, indent=2
    )


@mcp.resource("ticker://cache/stats")
def get_cache_stats() -> str:
    """Get comprehensive cache statistics including index coverage and operations."""
    index_tickers = {}
    for index_name in INDEX_URLS:
        try:
            tickers = fetch_index_tickers(index_name)
            if tickers:
                index_tickers[index_name] = tickers
        except Exception:
            pass  # Skip index if fetch fails

    return json.dumps(cache.stats(index_tickers), indent=2)


@mcp.resource("ticker://ticker/{symbol}")
def get_ticker(symbol: str) -> str:
    """Get full data for a cached ticker. Returns error if not cached."""
    info = cache.get(symbol.upper())
    if info:
        return json.dumps(info.to_flat_dict(), indent=2)
    return json.dumps({"error": "not cached", "symbol": symbol.upper()})


@mcp.resource("ticker://ticker/{symbol}/metrics")
def get_ticker_metrics(symbol: str) -> str:
    """Get only metrics (returns, volatility) for a cached ticker."""
    info = cache.get(symbol.upper())
    if info:
        if info.metrics.has_data():
            return json.dumps(
                {
                    "symbol": symbol.upper(),
                    "returns": info.metrics.returns,
                    "volatility": info.metrics.volatility,
                    "updated_at": (
                        info.metrics.updated_at.isoformat()
                        if info.metrics.updated_at
                        else None
                    ),
                },
                indent=2,
            )
        return json.dumps({"error": "no metrics", "symbol": symbol.upper()})
    return json.dumps({"error": "not cached", "symbol": symbol.upper()})


@mcp.resource("ticker://indexes")
def list_indexes() -> str:
    """List available market indexes."""
    return json.dumps(list(INDEX_URLS.keys()))


@mcp.resource("ticker://indexes/{index}")
def get_index_tickers(index: str) -> str:
    """Get all cached ticker info for an index."""
    index_name = INDEX_ALIASES.get(index.lower(), index.lower())
    if index_name not in INDEX_URLS:
        return json.dumps(
            {"error": f"Unknown index: {index}. Available: {list(INDEX_URLS.keys())}"}
        )

    tickers = fetch_index_tickers(index_name)
    result = {}
    for ticker in tickers:
        info = cache.get(ticker)
        if info:
            result[ticker] = info.to_flat_dict()

    return json.dumps(result, indent=2)
