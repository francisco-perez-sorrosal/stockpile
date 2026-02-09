#!/usr/bin/env python3
"""MCP server for stock ticker data cache.

Provides Resources for data discovery and Tools for data access.
All Yahoo Finance API calls are handled here - skills use MCP for data.

Resources (read-only views of cached data):
    ticker://cache              - Summary of all cached tickers
    ticker://cache/stats        - Cache statistics
    ticker://ticker/{symbol}    - Full data for a cached ticker
    ticker://ticker/{symbol}/metrics - Just metrics for a cached ticker
    ticker://indexes            - List of supported index names
    ticker://indexes/{index}    - All cached ticker info for an index

Tools (actions that fetch/compute):
    lookup(query)           - Unified: ticker, comma-list, index name, or company name
    refresh_metrics(symbols) - Calculate return/volatility from 1y prices
"""

from app import args, mcp

# Side-effect imports: register resource and tool handlers with the mcp instance
import resources  # noqa: F401, E402
import tools  # noqa: F401, E402


def main():
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
