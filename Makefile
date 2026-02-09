# Stockpile Plugin Makefile

DIST_DIR := dist

# --- Skills ---
skill-build:
ifndef SKILL
	$(error SKILL is required. Usage: make SKILL=<name> skill-build)
endif
	@mkdir -p $(DIST_DIR)
	@cd skills/$(SKILL) && zip -r ../../$(DIST_DIR)/$(SKILL).zip . -x "tests/*" -x "__pycache__/*" -x "*.pyc"
	@echo "Built $(DIST_DIR)/$(SKILL).zip"

skill-install: skill-build
	@mkdir -p ~/.claude/skills/$(SKILL)
	@unzip -o $(DIST_DIR)/$(SKILL).zip -d ~/.claude/skills/$(SKILL)/

# --- MCP Server ---
mcp-test:
	cd mcp/ticker-cache && uv run python -c "from main import mcp; print('OK')"

mcp-run:
	cd mcp/ticker-cache && uv run main.py

mcp-inspect:
	cd mcp/ticker-cache && npx @modelcontextprotocol/inspector uv --directory . run main.py

mcp-pack:
	@mkdir -p $(DIST_DIR)
	cd mcp/ticker-cache && npx @anthropic-ai/mcpb pack . ../../$(DIST_DIR)/

# --- Cleanup ---
clean:
	rm -rf $(DIST_DIR)

.PHONY: skill-build skill-install mcp-test mcp-run mcp-inspect mcp-pack clean
