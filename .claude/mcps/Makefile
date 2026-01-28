# Makefile for managing MCP servers
#
# Usage:
#   make MCP=ticker-cache install     # Install MCP server to Claude Code
#   make MCP=ticker-cache build-mcpb  # Build MCPB bundle for Claude Desktop
#   make MCP=ticker-cache test        # Test server imports
#   make MCP=ticker-cache clean       # Clean build artifacts
#   make list                         # List available MCP servers

.PHONY: install uninstall build-mcpb test clean list help

# Default target
help:
	@echo "MCP Server Management"
	@echo ""
	@echo "Usage: make MCP=<server_name> <target>"
	@echo ""
	@echo "Targets:"
	@echo "  install      Install MCP server to Claude Code (user scope)"
	@echo "  uninstall    Remove MCP server from Claude Code"
	@echo "  build-mcpb   Build MCPB bundle for Claude Desktop"
	@echo "  test         Test server imports"
	@echo "  inspect      Run MCP inspector for interactive testing"
	@echo "  clean        Clean build artifacts"
	@echo "  list         List available MCP servers"
	@echo ""
	@echo "Example: make MCP=ticker-cache build-mcpb"

# Validate MCP variable is set
check-mcp:
ifndef MCP
	$(error MCP is not set. Usage: make MCP=<server_name> <target>)
endif
	@test -d $(MCP) || (echo "Error: MCP server '$(MCP)' not found" && exit 1)

# Install MCP server to Claude Code
install: check-mcp
	cd $(MCP) && $(MAKE) install

# Uninstall MCP server from Claude Code
uninstall: check-mcp
	cd $(MCP) && $(MAKE) uninstall

# Build MCPB bundle for Claude Desktop
build-mcpb: check-mcp
	cd $(MCP) && $(MAKE) build-mcpb

# Test server imports
test: check-mcp
	cd $(MCP) && $(MAKE) test

# Run MCP inspector
inspect: check-mcp
	cd $(MCP) && $(MAKE) inspect

# Clean build artifacts
clean: check-mcp
	cd $(MCP) && $(MAKE) clean

# List available MCP servers
list:
	@echo "Available MCP servers:"
	@for dir in */; do \
		if [ -f "$$dir/server.py" ] || [ -f "$$dir/manifest.json" ]; then \
			echo "  $${dir%/}"; \
		fi \
	done
