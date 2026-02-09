"""Shared application instances for the ticker-cache MCP server.

Centralizes the FastMCP server and TickerCache instances so that
main.py, resources.py, and tools.py can all import from here
without circular dependencies.
"""

from mcp.server.fastmcp import FastMCP

from cache import TickerCache


def _parse_args():
    """Parse CLI arguments early for server configuration."""
    import argparse

    parser = argparse.ArgumentParser(description="Ticker Cache MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol: stdio (default), sse, or streamable-http",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for SSE transport (default: 8000)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for SSE transport (default: 0.0.0.0)",
    )
    # Parse known args to avoid errors from other flags
    args, _ = parser.parse_known_args()
    return args


args = _parse_args()

mcp = FastMCP("ticker-cache", host=args.host, port=args.port)

cache = TickerCache()
