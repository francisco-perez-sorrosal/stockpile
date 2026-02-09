"""Characterization tests: verify MCP server imports and registration."""

import sys
from pathlib import Path

# Ensure the MCP server package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_mcp_instance_imports():
    """The 'mcp' object should be importable from main."""
    from main import mcp

    assert mcp is not None


def test_mcp_is_fastmcp_instance():
    """The 'mcp' object should be a FastMCP instance."""
    from mcp.server.fastmcp import FastMCP

    from main import mcp

    assert isinstance(mcp, FastMCP)


def test_mcp_has_lookup_tool():
    """The server should register a 'lookup' tool."""
    from main import mcp

    tool_names = [tool.name for tool in mcp._tool_manager.list_tools()]
    assert "lookup" in tool_names


def test_mcp_has_refresh_metrics_tool():
    """The server should register a 'refresh_metrics' tool."""
    from main import mcp

    tool_names = [tool.name for tool in mcp._tool_manager.list_tools()]
    assert "refresh_metrics" in tool_names


def test_mcp_has_resources():
    """The server should register at least one resource."""
    from main import mcp

    resources = mcp._resource_manager.list_resources()
    assert len(resources) > 0


def test_mcp_resource_uris():
    """The server should register the expected resource URI patterns."""
    from main import mcp

    resources = mcp._resource_manager.list_resources()
    uris = {str(r.uri) for r in resources}

    expected_static_uris = {
        "ticker://cache",
        "ticker://cache/stats",
        "ticker://indexes",
    }
    assert expected_static_uris.issubset(uris), (
        f"Missing static URIs: {expected_static_uris - uris}"
    )


def test_mcp_has_resource_templates():
    """The server should register resource templates for parameterized URIs."""
    from main import mcp

    templates = mcp._resource_manager.list_templates()
    template_uris = {str(t.uri_template) for t in templates}

    expected_templates = {
        "ticker://ticker/{symbol}",
        "ticker://ticker/{symbol}/metrics",
        "ticker://indexes/{index}",
    }
    assert expected_templates.issubset(template_uris), (
        f"Missing template URIs: {expected_templates - template_uris}"
    )
