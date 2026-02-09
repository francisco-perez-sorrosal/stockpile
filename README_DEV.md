# Stockpile — Development Guide

Development setup, build targets, and contribution workflow for Stockpile.

## Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| [uv](https://docs.astral.sh/uv/) | Python package management, MCP server runtime | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [Node.js](https://nodejs.org/) | MCP inspector, MCPB bundling (via npx) | `brew install node` |
| [jq](https://jqlang.github.io/jq/) | JSON manipulation for config files | `brew install jq` |

## Setup

```bash
git clone git@github.com:francisco-perez-sorrosal/stockpile.git
cd stockpile
```

Skills and MCP server are auto-discovered from the plugin manifest — no installation step needed. Open Claude Code from the repo directory and everything works.

To verify the MCP server imports correctly:

```bash
make mcp-test
```

## Project Structure

```
├── .claude-plugin/
│   └── plugin.json             # Plugin manifest (skills + MCP declaration)
├── skills/
│   ├── ticker/
│   │   ├── SKILL.md            # Skill instructions
│   │   └── reference.md        # Technical reference
│   └── stock-clusters/
│       ├── SKILL.md
│       ├── reference.md
│       └── scripts/
│           └── stock_clusters.py
├── mcp/
│   └── ticker-cache/
│       ├── app.py              # Shared instances (FastMCP, TickerCache)
│       ├── main.py             # Entry point
│       ├── resources.py        # MCP resource handlers
│       ├── tools.py            # MCP tool handlers
│       ├── models.py           # Pydantic models
│       ├── cache.py            # TickerCache class
│       ├── constants.py        # Constants and config
│       ├── http_helpers.py     # HTTP fetch helpers
│       ├── yahoo.py            # Yahoo Finance API
│       ├── scraping.py         # Wikipedia scraping
│       ├── pyproject.toml      # Server dependencies
│       ├── manifest.json       # MCPB metadata
│       ├── Makefile            # Server-level targets
│       └── README.md
├── install.sh                  # Interactive installer
├── Makefile                    # Root build targets
├── .mcp.json                   # MCP server launch config (project-level)
├── CLAUDE.md                   # Project instructions for Claude
├── README.md                   # User documentation
└── README_DEV.md               # This file
```

## MCP Server Development

### Running the server

```bash
# stdio transport (default, used by Claude Code)
make mcp-run

# HTTP/SSE transport (for remote access)
cd mcp/ticker-cache && make run-http

# Streamable HTTP transport
cd mcp/ticker-cache && make run-streamable
```

### Interactive testing with MCP Inspector

```bash
# Inspect via stdio
make mcp-inspect

# Inspect running HTTP server (start with make run-http first)
cd mcp/ticker-cache && make inspect-http
```

The MCP Inspector provides a web UI to call tools and read resources interactively.

### MCP resources and tools

| Resources (read-only) | Description |
|----------------------|-------------|
| `ticker://cache` | All cached tickers |
| `ticker://cache/stats` | Cache statistics |
| `ticker://ticker/{symbol}` | Single ticker data |
| `ticker://ticker/{symbol}/metrics` | Metrics only |
| `ticker://indexes` | Available index names |
| `ticker://indexes/{index}` | Cached tickers for index |

| Tools (actions) | Description |
|-----------------|-------------|
| `lookup(query)` | Unified: ticker, list, index, or name |
| `refresh_metrics(symbols)` | Calculate return/volatility |

### Server modules

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

## Skill Development

### Skill structure

```
<skill_name>/
├── SKILL.md        # Required: instructions with YAML frontmatter
├── reference.md    # Optional: technical reference
└── scripts/        # Optional: Python scripts
    └── tool.py
```

### SKILL.md format

```markdown
---
name: my-skill
description: Brief description of when Claude should use this skill.
---

## Quick Start
...
```

Constraints:
- `name`: max 64 characters
- `description`: max 200 characters

### Adding a new skill

1. Create directory: `skills/<skill_name>/`
2. Create `SKILL.md` with YAML frontmatter (`name`, `description`)
3. Add `scripts/` for computation (if needed)
4. Reference MCP tools/resources for data access

The skill is auto-discovered as `/stockpile:<skill_name>` when the plugin is active.

### Dependency management

Skills externalize dependency management — no `requirements.txt` in skill packages.

| Environment | How Dependencies Work |
|-------------|----------------------|
| **Claude API** | 184+ pre-installed packages only |
| **Claude Code** | Auto-runs `pip install` on import error |
| **Claude.ai** | Platform-managed packages |

Prefer pre-installed packages (pandas, numpy, scipy, matplotlib) and standard library for portability. Use MCP for external API access instead of direct HTTP in skills.

## Building Artifacts

### Skill ZIPs (for Claude Desktop upload)

```bash
make SKILL=ticker skill-build
make SKILL=stock-clusters skill-build
```

Output: `dist/<skill_name>.zip`

### MCPB bundle (for Claude Desktop MCP installation)

```bash
make mcp-pack
```

Output: `dist/ticker-cache-mcp-<version>.mcpb`

The MCPB bundle is a self-contained package that Claude Desktop can install by double-clicking.

## Makefile Reference

### Root Makefile

```bash
make mcp-test                        # Verify MCP server imports
make mcp-run                         # Run server (stdio)
make mcp-inspect                     # MCP Inspector (interactive)
make mcp-pack                        # Build MCPB bundle to dist/
make SKILL=<name> skill-build        # Build skill ZIP to dist/
make install                         # Run interactive installer (Claude Code)
make install-desktop                 # Run interactive installer (Claude Desktop)
make uninstall                       # Run interactive uninstaller (Claude Code)
make uninstall-desktop               # Run interactive uninstaller (Claude Desktop)
make clean                           # Remove dist/
```

### Server Makefile (`mcp/ticker-cache/Makefile`)

```bash
make install                         # Register MCP in Claude Code user scope
make uninstall                       # Remove MCP from Claude Code
make test                            # Verify server imports
make run                             # Run server (stdio)
make run-http                        # Run with HTTP/SSE transport
make run-streamable                  # Run with streamable-http transport
make inspect                         # MCP Inspector (stdio)
make inspect-http                    # MCP Inspector (HTTP)
make build-mcpb                      # Sync deps + build MCPB bundle
make pack                            # Build MCPB bundle only
make clean                           # Remove build artifacts
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| MCP tools not available | Verify `.mcp.json` exists at project root; run `make mcp-test` |
| Skills not discovered | Verify `.claude-plugin/plugin.json` exists; check `skills/` directory |
| `uv` not found | Install uv: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| MCP Inspector won't start | Ensure Node.js is installed: `brew install node` |
| Server import errors | Run `cd mcp/ticker-cache && uv sync` to install dependencies |
| Cache issues | Delete `~/.cache/ticker/tickers.json` and retry |
