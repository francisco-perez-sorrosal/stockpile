# Stockpile

Skills and MCP servers for investment research with Claude, packaged as a Claude Code plugin.

## Quick Start

**As a project-scope plugin** (clone and use):
```bash
git clone git@github.com:francisco-perez-sorrosal/stockpile.git
cd stockpile
# Skills and MCP server are auto-discovered -- no installation needed
```

**As a personal plugin** (available across projects):
```bash
claude plugin install stockpile
```

Then use the skills:
```
What's the ticker for Apple?
Cluster NASDAQ-100 stocks by return and volatility
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Skills Layer                          │
│  skills/ticker/SKILL.md, skills/stock-clusters/SKILL.md     │
├─────────────────────────────────────────────────────────────┤
│                     ticker-cache MCP                         │
│  Tools: lookup(), refresh_metrics()                          │
│  Resources: ticker://cache, ticker://ticker/{symbol}, ...    │
├─────────────────────────────────────────────────────────────┤
│               Yahoo Finance + Wikipedia APIs                 │
│               Cache: ~/.cache/ticker/tickers.json           │
└─────────────────────────────────────────────────────────────┘
```

## Plugin Structure

Stockpile uses the Claude Code plugin format. The `.claude-plugin/plugin.json` manifest declares the plugin identity. When the plugin is active:

- Skills at `skills/` are auto-discovered as `/stockpile:ticker` and `/stockpile:stock-clusters`
- The MCP server at `mcp/ticker-cache/` auto-starts via `.mcp.json`
- No manual `make install` needed for project-scope use

## Available Skills

### Ticker Skill

Look up stock ticker symbols from company names using the MCP server.

**Example usage:**

```
What's the ticker symbol for Apple?
Look up the ticker for Tesla
Is AAPL a valid ticker symbol?
What sector and industry is Amazon in?
```

The skill uses the `lookup()` MCP tool which auto-detects whether input is a ticker, company name, index name, or comma-separated list.

### Stock Clusters Skill

Cluster stocks by return and volatility using K-means analysis.

**Example usage:**

```
Cluster AAPL, MSFT, GOOGL, NVDA, and META by return and volatility
Cluster S&P 500 stocks into 7 groups
Analyze NASDAQ-100 stocks and identify the best performers
```

The skill orchestrates:
1. MCP data access via `lookup()` and `refresh_metrics()` tools
2. K-means clustering via scipy
3. Interactive visualizations via plotly (falls back to matplotlib)

Supports `--data-file` for sandbox environments where the MCP server is not available.

## Skills vs MCP

| Aspect | Skills | MCP Server |
|--------|--------|------------|
| Purpose | Teaching workflows and domain expertise | Connecting to external data/APIs |
| Format | SKILL.md files (Markdown + YAML) | JSON-RPC protocol server |
| Data access | Via MCP tools | Direct API calls + caching |

Skills in this project depend on the MCP server for data. The MCP server handles Yahoo Finance API calls and caching, while skills contain the workflow instructions and domain logic.

## Skill Structure

```
<skill_name>/
├── SKILL.md        # Required: instructions with YAML frontmatter
├── reference.md    # Optional: technical reference
└── scripts/        # Optional: Python scripts
    └── tool.py
```

### SKILL.md Format

```markdown
---
name: my-skill
description: Brief description of when Claude should use this skill.
---

## Quick Start
...
```

**Constraints:**
- `name`: max 64 characters
- `description`: max 200 characters

## Building and Installing

### Root Makefile

```bash
# Build a skill ZIP for distribution
make SKILL=stock-clusters skill-build

# Install a skill to ~/.claude/skills/
make SKILL=stock-clusters skill-install

# Test MCP server imports
make mcp-test

# Run MCP server (stdio)
make mcp-run

# Interactive MCP inspector
make mcp-inspect

# Build MCPB bundle for Claude Desktop
make mcp-pack

# Clean built artifacts
make clean
```

### Claude Desktop (MCPB Bundle)

1. Build the MCPB bundle:
   ```bash
   make mcp-pack
   ```
2. The bundle is created in `dist/`
3. Install in Claude Desktop by double-clicking the `.mcpb` file

### Claude Desktop (Skill ZIP)

1. Build the skill ZIP:
   ```bash
   make SKILL=ticker skill-build
   ```
2. The ZIP is created in `dist/<skill_name>.zip`
3. Open Claude Desktop, go to Settings, then Capabilities, then Skills
4. Click "Upload skill" and select the ZIP file

## Troubleshooting

| Problem | Solution |
|---------|----------|
| MCP tools not available | Verify `.mcp.json` exists at project root; run `make mcp-test` |
| Skills not discovered | Verify `.claude-plugin/plugin.json` exists; check `skills/` directory |
| Skills section not visible | Enable "Code execution" in Settings; requires paid plan |
| Claude not using skill | Check it's toggled on; ensure description explains when to use |
| Upload fails | Ensure ZIP has skill folder as root with valid SKILL.md frontmatter |

## Resources

- [Engineering deep-dive](https://anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [Skills GitHub examples](https://github.com/anthropics/skills)
- [Skills open standard](https://agentskills.io)
- [Help Center - Using Skills](https://support.claude.com/en/articles/12512180-using-skills-in-claude)
- [Help Center - Creating Skills](https://support.claude.com/en/articles/12512198-how-to-create-custom-skills)
