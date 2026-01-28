# Makefile for ticker-cache MCP server

.PHONY: install uninstall test run inspect build-mcpb sync-deps pack clean clean-bundle

# Get the absolute path to this directory
DIR := $(shell pwd)

# Bundle output directory (shared with skills)
MCPB_DIR := ../../dist

# =============================================================================
# Claude Code Installation
# =============================================================================

install:
	claude mcp add --scope user ticker-cache -- uv --directory $(DIR) run server.py

uninstall:
	claude mcp remove ticker-cache

# =============================================================================
# Development & Testing
# =============================================================================

test:
	uv run python -c "from server import mcp; print('Server imports OK')"

run:
	uv run server.py

# Run MCP inspector for interactive testing
inspect:
	npx @modelcontextprotocol/inspector uv --directory $(DIR) run server.py

# =============================================================================
# MCPB Bundle (for Claude Desktop)
# =============================================================================

# Build bundle: sync deps (creates uv.lock) → pack
# Note: Dependencies managed by uv at runtime, not bundled
build-mcpb: sync-deps pack
	@echo "MCPB bundle created in $(MCPB_DIR)/"

# Sync dependencies and create uv.lock
sync-deps:
	uv sync
	uv lock

# Create MCPB bundle (includes pyproject.toml + uv.lock for uv to use)
pack:
	@mkdir -p $(MCPB_DIR)
	@BUNDLE_NAME=$$(python3 -c "import json; m=json.load(open('manifest.json')); print(f\"{m['name']}-{m['version']}.mcpb\")") && \
	npx @anthropic-ai/mcpb pack . $(MCPB_DIR)/$$BUNDLE_NAME && \
	echo "Bundle created: $(MCPB_DIR)/$$BUNDLE_NAME"

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf lib/ .venv/ __pycache__/
	@# Note: Keeps uv.lock for reproducibility. Does not remove bundles from dist/

clean-bundle:
	@BUNDLE_NAME=$$(python3 -c "import json; m=json.load(open('manifest.json')); print(f\"{m['name']}-{m['version']}.mcpb\")") && \
	rm -f $(MCPB_DIR)/$$BUNDLE_NAME && echo "Removed $$BUNDLE_NAME from dist/"
