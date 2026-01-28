# Investment Project

Project containing programming artifacts related to investment analysis and tooling.

**Repository**: `git@github.com:francisco-perez-sorrosal/investment.git`

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ticker-cache MCP                         │
│  Responsibility: Cache management + Yahoo Finance access     │
│  Location: .claude/mcps/ticker-cache/                        │
├─────────────────────────────────────────────────────────────┤
│  Resources: ticker://cache, ticker://ticker/{symbol}, ...    │
│  Tools: lookup(), search(), refresh_metrics()                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Skills Layer                          │
│  Location: .claude/skills/                                   │
├─────────────────────────────────────────────────────────────┤
│  ticker/SKILL.md: Instructions for MCP data access           │
│  stock-clusters/SKILL.md: Clustering workflow orchestration  │
│  stock-clusters/scripts/: Clustering math + visualization    │
└─────────────────────────────────────────────────────────────┘
```

**Design principles:**
- MCP server handles all external API calls (Yahoo Finance)
- Skills are instruction documents that guide Claude
- Scripts contain domain logic only (clustering, visualization)
- Shared cache at `~/.cache/ticker/tickers.json`

## MCP Server

The `ticker-cache` MCP server provides stock ticker data via Resources and Tools.

### Installation

```bash
cd .claude/mcps/ticker-cache
make install
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
| `search(query, limit)` | Search company names |
| `refresh_metrics(symbols)` | Calculate return/volatility |

## Skills

### Ticker

Stock ticker lookup using the MCP server. See `.claude/skills/ticker/SKILL.md`.

### Stock Clusters

Cluster stocks by return/volatility using K-means. Orchestrates MCP data access + scipy clustering + plotly visualization. See `.claude/skills/stock-clusters/SKILL.md`.

## Structure

```
├── CLAUDE.md               # Project instructions
├── README.md               # User documentation
├── .mcp.json               # Project-level MCP configuration
└── .claude/
    ├── mcps/
    │   └── ticker-cache/   # MCP server
    │       ├── server.py
    │       ├── pyproject.toml
    │       ├── Makefile
    │       └── README.md
    └── skills/
        ├── Makefile        # Skill maintenance
        ├── ticker/
        │   └── SKILL.md
        └── stock-clusters/
            ├── SKILL.md
            └── scripts/
                └── stock_clusters.py
```

## Adding a New Skill

1. Create directory: `.claude/skills/<skill_name>/`
2. Create `SKILL.md` with YAML frontmatter (`name`, `description`)
3. Add `scripts/` for computation (if needed)
4. Reference MCP tools/resources for data access
5. Build and install: `make SKILL=<skill_name> build && make SKILL=<skill_name> install`

## Skill Dependency Management

Skills externalize dependency management—no `requirements.txt` in skill packages.

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
