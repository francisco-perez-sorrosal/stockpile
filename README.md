# Stockpile

Skills and MCP servers for investment research with Claude.

## Available Skills

This repository includes skills and MCP servers for financial analysis and investment research:

### Ticker Skill

The **ticker** skill enables Claude to look up stock market ticker symbols from company names using Yahoo Finance. It demonstrates:

- **Zero external dependencies**: Uses only Python standard library (urllib, json)
- **Universal compatibility**: Works in Claude API, Claude Code, and Claude Desktop
- **Robust search**: Handles company name disambiguation and provides detailed results
- **Reference documentation**: Includes comprehensive API documentation

**Example usage:**

Basic ticker lookup:
```
What's the ticker symbol for Apple?
Look up the stock ticker for Tesla
Find the ticker for Microsoft
```

Ticker validation and reverse lookup:
```
Is AAPL a valid ticker symbol?
Tell me about the company with ticker MSFT
What company does ticker TSLA belong to?
```

Detailed company information (sector, industry, market cap):
```
Show me details about Netflix stock
What sector and industry is Amazon in? Include market cap.
Give me comprehensive information on ticker GOOGL
```

Disambiguation when multiple matches exist:
```
Find all tickers for companies named "National Bank"
Search for tech companies with "Cloud" in their name
```

See `.claude/skills/ticker/` for the complete implementation, including the SKILL.md definition, Python scripts, and Yahoo Finance API reference documentation.

### Stock Clusters Skill

The **stock-clusters** skill performs K-means clustering analysis on stocks based on returns and volatility. It demonstrates:

- **Pre-installed packages only**: Uses pandas, numpy, scipy, and matplotlib (no yfinance dependency)
- **Direct API access**: Fetches data from Yahoo Finance API without external libraries
- **Flexible ticker selection**: Use custom tickers or fetch from major indexes (S&P 500, NASDAQ-100, Dow Jones)
- **Configurable analysis**: Adjustable cluster counts and lookback periods
- **Visual insights**: Generates scatter plots showing risk/reward profiles

**Example usage:**

Basic clustering analysis:
```
Cluster AAPL, MSFT, GOOGL, NVDA, and META by return and volatility
Analyze tech stocks and show me risk/reward clusters
Cluster NASDAQ-100 stocks by performance
```

Customized analysis:
```
Cluster S&P 500 stocks into 7 groups based on risk and return
Analyze Dow Jones stocks and identify the best performers
Show me high-return, low-volatility stocks from the NASDAQ-100
```

The skill outputs cluster statistics (mean returns, volatility, stock counts) and saves visualization plots to disk, making it easy to identify opportunities across different risk profiles.

See `.claude/skills/stock-clusters/` for the complete implementation, including the SKILL.md definition and Python clustering script.

## Skills vs MCP

| Aspect | Skills | MCP Servers |
|--------|--------|-------------|
| Primary function | Teaching workflows and domain expertise | Connecting to external data/APIs |
| Format | SKILL.md files (Markdown + YAML frontmatter) | JSON-RPC protocol servers |
| Token efficiency | ~100 tokens metadata, <5k full load | Can consume 10k+ tokens |

Anthropic's analogy: "MCP is like having access to the aisles. Skills are like an employee's expertise."

## Skill Structure

A skill is a directory containing:

```
<skill_name>/
├── SKILL.md        # Required: main instructions with YAML frontmatter
├── reference.md    # Optional: technical reference, API docs
└── scripts/        # Optional: Python scripts Claude can execute
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

## Common Tasks
...
```

**Constraints:**
- `name`: max 64 characters
- `description`: max 200 characters (Claude uses this to decide when to invoke the skill)

## Dependency Management

Skills rely on the execution environment's pre-installed packages—no `requirements.txt` or `pyproject.toml`.

| Environment | Package Installation | Dependency Approach |
|-------------|---------------------|---------------------|
| **Claude API** | Pre-installed only | Use 184+ available packages |
| **Claude Code** | Dynamic (`pip install`) | Auto-installs on `ModuleNotFoundError` |
| **Claude.ai** | Platform-managed | Pre-installed packages only |

### Pre-installed Packages (API Container)

Python 3.11.12 with 184+ packages including:

- **Data**: pandas, numpy, scipy, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Documents**: pypdf, pdfplumber, python-docx, openpyxl
- **Utilities**: pillow, pyarrow, PyYAML, tqdm

Full list: [Code Execution Tool docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool)

### Best Practices

1. **Prefer Python standard library** for maximum portability (`urllib`, `json`, `argparse`)
2. **Design for pre-installed packages** when targeting Claude API
3. **Document dependencies clearly** in SKILL.md if external packages are required
4. **Handle import errors gracefully** with try/except and helpful messages

### Example: No External Dependencies

The `ticker` skill uses only standard library to call Yahoo Finance APIs directly:

```python
import urllib.request
import json

# No pip install needed—works everywhere
```

This approach ensures the skill works in Claude API, Claude Code, and Claude.ai without modification.

## Building and Installing Skills

This project includes a Makefile in `.claude/skills/` to automate skill packaging and installation. The Makefile handles creating distribution zips for Claude Desktop/Web and installing skills locally for Claude Code.

### Makefile Commands

All commands must be run from the `.claude/skills/` directory with the `SKILL=<name>` parameter:

```bash
cd .claude/skills

# Build a distributable zip file
make SKILL=ticker build

# Install skill to ~/.claude/skills/ (for Claude Code)
make SKILL=ticker install

# Remove generated artifacts
make SKILL=ticker clean

# List contents of skill directory
make SKILL=ticker list

# Show available commands
make help
```

### Installing Skills in Claude Desktop / Web

1. Build the skill zip:
   ```bash
   cd .claude/skills
   make SKILL=ticker build
   ```

2. The zip file will be created in `.claude/dist/<skill_name>.zip`

3. Open Claude Desktop → Settings → Capabilities → Skills

4. Click "Upload skill" and select the zip file

5. Toggle the skill on to activate it

### Installing Skills in Claude Code

Skills in `~/.claude/skills/` are automatically discovered. To install:

```bash
cd .claude/skills
make SKILL=ticker install
```

This command builds the zip and extracts it to `~/.claude/skills/ticker/`, making it immediately available in Claude Code.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Skills section not visible | Enable "Code execution" in Settings; requires paid plan |
| Claude not using skill | Check it's toggled on; ensure description explains when to use it |
| Upload fails | Ensure ZIP has skill folder as root with valid SKILL.md frontmatter |
| ModuleNotFoundError | Use standard library or pre-installed packages; Claude Code auto-installs |

## Resources

- [Engineering deep-dive](https://anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [GitHub examples](https://github.com/anthropics/skills)
- [Open standard specification](https://agentskills.io)
- [Code Execution Tool docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool)
- [Help Center - Using Skills](https://support.claude.com/en/articles/12512180-using-skills-in-claude)
- [Help Center - Creating Skills](https://support.claude.com/en/articles/12512198-how-to-create-custom-skills)
