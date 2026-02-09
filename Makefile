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

# --- Installation (interactive) ---
install:
	./install.sh

install-desktop:
	./install.sh desktop

uninstall:
	./install.sh --uninstall

uninstall-desktop:
	./install.sh desktop --uninstall

# --- Cleanup ---
clean:
	rm -rf $(DIST_DIR)

.PHONY: skill-build mcp-test mcp-run mcp-inspect mcp-pack install install-desktop uninstall uninstall-desktop clean
