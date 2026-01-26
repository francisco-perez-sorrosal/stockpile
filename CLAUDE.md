# Investment Project

Project containing programming artifacts related to investment analysis and tooling.

**Repository**: `git@github.com:francisco-perez-sorrosal/investment.git`

## Available Skills

### Ticker

Stock ticker symbol lookup using Yahoo Finance. Converts company names to ticker symbols with zero external dependencies (uses only Python standard library). See `.claude/skills/ticker/` for implementation.

### Stock Clusters

Analyze stocks by return and volatility, clustering them with K-means to identify investment opportunities. Works with custom tickers or major market indexes (S&P 500, NASDAQ-100, Dow Jones). Uses pandas, numpy, and scipy (all pre-installed). No `yfinance` dependency—fetches data directly from Yahoo Finance API. See `.claude/skills/stock-clusters/` for implementation.

## Structure

```
├── CLAUDE.md           # Project instructions (for Claude)
├── README.md           # User documentation on skills
└── .claude/
    ├── dist/           # Generated zip files for distribution
    └── skills/
        ├── Makefile    # Unified skill maintenance
        └── <skill_name>/   # Individual skill directories
            ├── SKILL.md    # Skill definition (YAML frontmatter + instructions)
            ├── reference.md # Technical reference (optional)
            └── scripts/    # Python scripts (optional)
```

## Skills

Skills are Markdown files with YAML frontmatter that teach Claude domain-specific workflows. See `README.md` for full documentation.

### Skill File Format

```markdown
---
name: skill-name
description: When Claude should invoke this skill (max 200 chars).
---

## Instructions
...
```

### Maintenance Commands

Run from the `skills/` directory:

```bash
cd .claude/skills
make SKILL=<skill_name> build    # Create zip in dist/
make SKILL=<skill_name> install  # Copy to ~/.claude/skills/
make SKILL=<skill_name> clean    # Remove generated artifacts
make SKILL=<skill_name> list     # Show skill contents
```

## Adding a New Skill

1. Create directory: `.claude/skills/<skill_name>/`
2. Create `SKILL.md` with required YAML frontmatter (`name`, `description`)
3. Add `scripts/` directory for Python scripts (if needed)
4. Add `reference.md` for technical details (if needed)
5. Build and install: `make SKILL=<skill_name> build && make SKILL=<skill_name> install`

## Skill Dependency Management

Skills deliberately externalize dependency management to the execution environment—there are no `requirements.txt` or `pyproject.toml` files in skill packages.

| Environment | How Dependencies Work |
|-------------|----------------------|
| **Claude API** | 184+ pre-installed packages only; no runtime install |
| **Claude Code** | Auto-runs `pip install` on `ModuleNotFoundError` |
| **Claude.ai** | Platform-managed pre-installed packages |

### When Writing Skill Scripts

1. **Design for pre-installed packages** when possible (pandas, numpy, matplotlib, pypdf, etc.)
2. **Prefer Python standard library** for maximum portability (urllib, json, argparse)
3. **Document dependencies clearly** in SKILL.md with explicit `pip install` instructions
4. **Handle import errors gracefully** with try/except and helpful messages
5. **For non-pre-installed packages**: Claude Code will auto-install; API will fail

### Example Dependency Documentation in SKILL.md

**For skills with NO external dependencies (recommended):**

```markdown
## Dependencies

None. This skill uses only Python standard library (`urllib`, `json`, `argparse`), ensuring compatibility across all Claude environments (API, Code, and Desktop).
```

**For skills requiring non-pre-installed packages:**

```markdown
## Dependencies

This skill requires `package-name` for [functionality].

**Environment compatibility:**
- **Claude Code**: Auto-installs on first use
- **Claude API**: ❌ Not available (not pre-installed)
- **Claude Desktop**: Depends on platform package availability

For local testing: `pip install package-name`
```

All skills in this project prefer pre-installed packages (pandas, numpy, scipy, matplotlib) and Python standard library for maximum compatibility.

See `README.md` for the full list of pre-installed packages and detailed best practices.
