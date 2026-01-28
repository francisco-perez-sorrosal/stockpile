# Stockpile

Skills and MCP servers for investment research with Claude.

## Quick Start

1. Install the MCP server:
   ```bash
   cd .claude/mcps/ticker-cache
   make install
   ```

2. Use the skills:
   ```
   What's the ticker for Apple?
   Cluster NASDAQ-100 stocks by return and volatility
   ```

## Architecture

The project follows a layered architecture where all external data access goes through the MCP server:

```
┌─────────────────────────────────────────────────────────────┐
│                        Skills Layer                          │
│  ticker/SKILL.md, stock-clusters/SKILL.md                   │
├─────────────────────────────────────────────────────────────┤
│                     ticker-cache MCP                         │
│  Tools: lookup(), refresh_metrics()                          │
│  Resources: ticker://cache, ticker://ticker/{symbol}, ...    │
├─────────────────────────────────────────────────────────────┤
│               Yahoo Finance + Wikipedia APIs                 │
│               Cache: ~/.cache/ticker/tickers.json           │
└─────────────────────────────────────────────────────────────┘
```

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
3. Interactive visualizations via plotly

Outputs include cluster statistics (mean returns, volatility) and HTML scatter plots.

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

### MCP Server

```bash
cd .claude/mcps/ticker-cache
make install      # Register with Claude Code
make test         # Verify imports
make inspect      # Interactive inspector
```

### Skills

```bash
cd .claude/skills

# Build distributable zip
make SKILL=ticker build

# Install to ~/.claude/skills/ (Claude Code)
make SKILL=ticker install
```

### Claude Desktop / Web

1. Build the skill zip:
   ```bash
   cd .claude/skills
   make SKILL=ticker build
   ```

2. The zip is created in `.claude/dist/<skill_name>.zip`

3. Open Claude Desktop → Settings → Capabilities → Skills

4. Click "Upload skill" and select the zip file

## Troubleshooting

| Problem | Solution |
|---------|----------|
| MCP tools not available | Run `make install` in `.claude/mcps/ticker-cache/` |
| Skills section not visible | Enable "Code execution" in Settings; requires paid plan |
| Claude not using skill | Check it's toggled on; ensure description explains when to use |
| Upload fails | Ensure ZIP has skill folder as root with valid SKILL.md frontmatter |

## Resources

- [Engineering deep-dive](https://anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [Skills GitHub examples](https://github.com/anthropics/skills)
- [Skills open standard](https://agentskills.io)
- [Help Center - Using Skills](https://support.claude.com/en/articles/12512180-using-skills-in-claude)
- [Help Center - Creating Skills](https://support.claude.com/en/articles/12512198-how-to-create-custom-skills)
