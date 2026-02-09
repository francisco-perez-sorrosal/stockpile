# Stockpile

Skills and MCP servers for investment research with Claude, packaged as a Claude Code plugin.

**Repository**: `git@github.com:francisco-perez-sorrosal/stockpile.git`

## Plugin

Stockpile is a Claude Code plugin. The `.claude-plugin/plugin.json` manifest declares the plugin identity, and Claude Code auto-discovers skills and the MCP server when the plugin is active.

- **Project-scope**: clone the repo and skills + MCP server are available automatically
- **Personal install**: `claude plugin install stockpile` makes it available across projects
- Skills are namespaced as `/stockpile:ticker` and `/stockpile:stock-clusters`

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ticker-cache MCP                         │
│  Responsibility: Cache management + Yahoo Finance access     │
│  Location: mcp/ticker-cache/                                 │
│  Modules: app.py, main.py, resources.py, tools.py,          │
│           models.py, cache.py, constants.py,                 │
│           http_helpers.py, yahoo.py, scraping.py             │
├─────────────────────────────────────────────────────────────┤
│  Resources: ticker://cache, ticker://ticker/{symbol}, ...    │
│  Tools: lookup(), refresh_metrics()                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Skills Layer                          │
│  Location: skills/                                           │
├─────────────────────────────────────────────────────────────┤
│  ticker/SKILL.md: Instructions for MCP data access           │
│  stock-clusters/SKILL.md: Clustering workflow orchestration  │
│  stock-clusters/scripts/: Clustering math + visualization    │
└─────────────────────────────────────────────────────────────┘
```

**Design principles:**
- MCP server handles all external API calls (Yahoo Finance)
- MCP server is auto-started when the plugin is active (via `.mcp.json`)
- Skills are instruction documents that guide Claude
- Scripts contain domain logic only (clustering, visualization)
- Shared cache at `~/.cache/ticker/tickers.json`

## MCP Server

The `ticker-cache` MCP server at `mcp/ticker-cache/` provides stock ticker data via Resources and Tools. It is decomposed into focused modules:

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

### Installation

The MCP server auto-starts when the plugin is active. No manual installation needed.

For manual registration with Claude Code:
```bash
cd mcp/ticker-cache && make install
```

### Resources (read-only views)

| URI | Description |
|-----|-------------|
| `ticker://cache` | All cached tickers |
| `ticker://cache/stats` | Cache statistics |
| `ticker://ticker/{symbol}` | Single ticker data |
| `ticker://ticker/{symbol}/metrics` | Metrics only |
| `ticker://indexes` | Available index names |
| `ticker://indexes/{index}` | Cached tickers for index |

### Tools (actions)

| Tool | Description |
|------|-------------|
| `lookup(query)` | Unified: ticker, list, index, or name |
| `refresh_metrics(symbols)` | Calculate return/volatility |

## Skills

### Ticker

Stock ticker lookup using the MCP server. See `skills/ticker/SKILL.md`.

### Stock Clusters

Cluster stocks by return/volatility using K-means. Orchestrates MCP data access + scipy clustering + plotly visualization. Supports `--data-file` for sandbox-friendly data input. See `skills/stock-clusters/SKILL.md`.

## Structure

```
├── .claude-plugin/
│   └── plugin.json             # Plugin manifest
├── skills/
│   ├── ticker/
│   │   ├── SKILL.md
│   │   └── reference.md
│   └── stock-clusters/
│       ├── SKILL.md
│       ├── reference.md
│       └── scripts/
│           └── stock_clusters.py
├── mcp/
│   └── ticker-cache/
│       ├── app.py              # Shared instances
│       ├── main.py             # Entry point
│       ├── resources.py        # MCP resource handlers
│       ├── tools.py            # MCP tool handlers
│       ├── models.py           # Pydantic models
│       ├── cache.py            # TickerCache class
│       ├── constants.py        # Constants and config
│       ├── http_helpers.py     # HTTP fetch helpers
│       ├── yahoo.py            # Yahoo Finance API
│       ├── scraping.py         # Wikipedia scraping
│       ├── pyproject.toml      # Self-contained dependencies
│       ├── manifest.json       # MCPB metadata
│       ├── Makefile            # Server-level build targets
│       └── README.md
├── dist/                       # Built artifacts (gitignored)
├── CLAUDE.md                   # Project instructions (this file)
├── README.md                   # User documentation
├── Makefile                    # Root build targets
├── pyproject.toml              # uv workspace root
├── uv.lock                     # Workspace lockfile
├── .mcp.json                   # MCP server launch config
└── .claude/
    └── settings.local.json     # Claude Code local settings
```

## Build System

The root `Makefile` consolidates all build targets:

```bash
# Skills
make SKILL=stock-clusters skill-build     # Build skill ZIP to dist/
make SKILL=stock-clusters skill-install   # Install to ~/.claude/skills/

# MCP Server
make mcp-test                             # Verify server imports
make mcp-run                              # Run server (stdio)
make mcp-inspect                          # Interactive MCP inspector
make mcp-pack                             # Build MCPB bundle to dist/

# Cleanup
make clean                                # Remove dist/
```

## Adding a New Skill

1. Create directory: `skills/<skill_name>/`
2. Create `SKILL.md` with YAML frontmatter (`name`, `description`)
3. Add `scripts/` for computation (if needed)
4. Reference MCP tools/resources for data access
5. Build and install: `make SKILL=<skill_name> skill-build && make SKILL=<skill_name> skill-install`

The skill is auto-discovered as `/stockpile:<skill_name>` when the plugin is active.

## Skill Dependency Management

Skills externalize dependency management -- no `requirements.txt` in skill packages.

| Environment | How Dependencies Work |
|-------------|----------------------|
| **Claude API** | 184+ pre-installed packages only |
| **Claude Code** | Auto-runs `pip install` on import error |
| **Claude.ai** | Platform-managed packages |

**Best practices:**
1. Use pre-installed packages (pandas, numpy, scipy, matplotlib)
2. Prefer Python standard library for portability
3. Document dependencies in SKILL.md
4. Use MCP for external API access (not direct HTTP in skills)
