#!/bin/bash
# Stockpile Installer — Skills + MCP server for Claude
#
# Installs all components (skills and MCP server) into Claude Code or
# Claude Desktop. The installer walks you through method and scope choices
# interactively, defaulting to the recommended option at each step.
#
# Usage:
#   ./install.sh [code|desktop] [--uninstall]
#
# Installation overview:
#
#   ./install.sh code (default)
#   ┌──────────────────────────────────────────────────────────────────┐
#   │  What gets installed:                                            │
#   │    • ticker-cache MCP server (Yahoo Finance data + caching)      │
#   │    • /stockpile:ticker skill                                     │
#   │    • /stockpile:stock-clusters skill                             │
#   │                                                                  │
#   │  Methods (you choose during install):                            │
#   │    Plugin (recommended) ─→ marketplace add + plugin install      │
#   │      Scope: user (everywhere) or project (single directory)      │
#   │    Direct ─→ .mcp.json + plugin manifest auto-discovery          │
#   │      Scope: project-level or user-level (claude mcp add)         │
#   └──────────────────────────────────────────────────────────────────┘
#
#   ./install.sh desktop
#   ┌──────────────────────────────────────────────────────────────────┐
#   │  What gets installed:                                            │
#   │    • ticker-cache MCP server (MCPB bundle or config file)        │
#   │    • Skill ZIPs built for manual upload                          │
#   │                                                                  │
#   │  Methods (you choose during install):                            │
#   │    MCPB bundle (recommended) ─→ double-click to install          │
#   │    Direct config ─→ writes to Claude Desktop JSON config         │
#   │  Skills: always built as ZIPs, uploaded via Settings UI          │
#   └──────────────────────────────────────────────────────────────────┘

set -euo pipefail

# =============================================================================
# Constants
# =============================================================================

PLUGIN_NAME="stockpile"
MARKETPLACE_NAME="bit-agora"
MARKETPLACE_SOURCE="francisco-perez-sorrosal/bit-agora"
SERVICE_NAME="ticker-cache"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MCP_DIR="$SCRIPT_DIR/mcp/ticker-cache"
DIST_DIR="$SCRIPT_DIR/dist"
SKILLS_DIR="$SCRIPT_DIR/skills"
SKILL_NAMES=("ticker" "stock-clusters")

# =============================================================================
# Terminal formatting (disabled when not a TTY)
# =============================================================================

if [ -t 1 ]; then
    B=$'\033[1m' D=$'\033[2m' R=$'\033[0m'
else
    B='' D='' R=''
fi

# =============================================================================
# Helpers
# =============================================================================

info()   { printf "  ✓ %s\n" "$*"; }
warn()   { printf "  ⚠ %s\n" "$*"; }
fail()   { printf "  ✗ %s\n" "$*" >&2; exit 1; }
header() { printf "\n${B}%s${R}\n" "$*"; }
step()   { printf "    %s\n" "$*"; }

# Prompt for a numbered choice. Sets REPLY to the chosen number.
# Usage: ask <default> <max>
ask() {
    local default=$1 max=$2
    printf "\n"
    read -rp "  Choice [$default]: " choice
    choice="${choice:-$default}"
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "$max" ]; then
        fail "Invalid choice: $choice"
    fi
    REPLY="$choice"
}

require() {
    local cmd=$1 msg=$2
    command -v "$cmd" &>/dev/null || fail "$msg"
}

# =============================================================================
# Overview banner
# =============================================================================

show_overview() {
    cat <<EOF

${B}Stockpile Installer${R}

  Components:
    • ticker-cache MCP server   (Yahoo Finance data + caching)
    • ticker skill              (/stockpile:ticker)
    • stock-clusters skill      (/stockpile:stock-clusters)
EOF
}

# =============================================================================
# Claude Code — Plugin install
# =============================================================================

install_code_plugin() {
    require "claude" "claude CLI is required. Install: https://docs.anthropic.com/en/docs/claude-code"

    header "Plugin scope"
    cat <<EOF

  ${B}[1] User scope${R}
      ${D}Available in every Claude Code session. Install once, use everywhere.${R}

  ${B}[2] Project scope${R}
      ${D}Only available in a specific project directory. Useful for testing${R}
      ${D}or when you want to isolate the plugin from other projects.${R}
EOF
    ask 1 2
    local scope
    if [ "$REPLY" -eq 1 ]; then scope="user"; else scope="project"; fi

    header "Installing plugin ($scope scope)..."
    printf "  Adding marketplace: %s\n" "$MARKETPLACE_SOURCE"
    claude plugin marketplace add "$MARKETPLACE_SOURCE" 2>/dev/null || true
    printf "  Installing plugin: %s\n" "$PLUGIN_NAME"
    claude plugin install "$PLUGIN_NAME" --scope "$scope"

    printf "\n"
    info "Installation complete"
    step "Skills: /stockpile:ticker, /stockpile:stock-clusters"
    step "MCP server: ticker-cache (auto-started)"
    step "Manage with: claude plugin list, claude plugin update $PLUGIN_NAME"
}

# =============================================================================
# Claude Code — Direct registration (from cloned repo)
# =============================================================================

install_code_direct() {
    [ -d "$MCP_DIR" ] || fail "MCP server not found at $MCP_DIR — clone the repo first"

    header "MCP server registration"
    cat <<EOF

  ${B}[1] Project-level .mcp.json${R}
      ${D}Registers in this project's .mcp.json. Skills are auto-discovered${R}
      ${D}from the plugin manifest. Everything works when you open Claude${R}
      ${D}Code in this directory. Already configured if you just cloned.${R}

  ${B}[2] User scope${R}
      ${D}Registers globally via 'claude mcp add --scope user'. The MCP${R}
      ${D}server works from any directory, but skills are only discovered${R}
      ${D}when Claude Code is opened in the repo directory.${R}
EOF
    ask 1 2

    if [ "$REPLY" -eq 1 ]; then
        install_code_direct_project
    else
        install_code_direct_user
    fi
}

install_code_direct_project() {
    local target="$SCRIPT_DIR/.mcp.json"

    if [ -f "$target" ] && jq -e ".mcpServers.\"$SERVICE_NAME\"" "$target" >/dev/null 2>&1; then
        info ".mcp.json already has $SERVICE_NAME configured"
    else
        require "jq" "jq is required. Install with: brew install jq"
        header "Writing $SERVICE_NAME to .mcp.json..."
        if [ ! -f "$target" ] || ! jq -e . "$target" >/dev/null 2>&1; then
            echo '{"mcpServers":{}}' > "$target"
        fi
        # Use relative path for project-level config (portable)
        local config='{"command":"uv","args":["--directory","mcp/ticker-cache","run","main.py"]}'
        jq --arg n "$SERVICE_NAME" --argjson c "$config" \
            '.mcpServers[$n] = $c' "$target" > "${target}.tmp"
        mv "${target}.tmp" "$target"
        info "Registered $SERVICE_NAME in .mcp.json"
    fi

    if [ -f "$SCRIPT_DIR/.claude-plugin/plugin.json" ]; then
        info "Plugin manifest found — skills auto-discovered"
    else
        warn "No plugin.json found — skills won't be auto-discovered"
    fi

    printf "\n"
    info "Setup complete"
    step "Open Claude Code from this directory to use stockpile"
    step "Skills: /stockpile:ticker, /stockpile:stock-clusters"
}

install_code_direct_user() {
    require "claude" "claude CLI is required"

    printf "  Registering %s in user scope...\n" "$SERVICE_NAME"
    # Absolute path required for user scope (runs from any directory)
    claude mcp add --scope user "$SERVICE_NAME" -- uv --directory "$MCP_DIR" run main.py

    printf "\n"
    info "MCP server registered (user scope)"
    warn "Skills are only discovered when Claude Code is opened in the repo directory"
    step "Skills: /stockpile:ticker, /stockpile:stock-clusters"
}

# =============================================================================
# Claude Code — Top-level flow
# =============================================================================

install_code() {
    header "Claude Code Installation"
    cat <<EOF

  ${B}[1] Plugin install${R}
      ${D}Installs everything as a managed plugin package. Skills and MCP${R}
      ${D}server are auto-discovered. Updates via 'claude plugin update'.${R}
      ${D}Does not require the repo to be cloned locally.${R}

  ${B}[2] Direct from cloned repo${R}
      ${D}Uses the cloned repo as-is. Skills and MCP server are discovered${R}
      ${D}from the project directory. Best for development or customization.${R}
      ${D}Requires this repository cloned locally.${R}
EOF
    ask 1 2

    if [ "$REPLY" -eq 1 ]; then
        install_code_plugin
    else
        install_code_direct
    fi
}

# =============================================================================
# Claude Desktop — MCP via MCPB bundle
# =============================================================================

install_desktop_mcpb() {
    require "npx" "npx (Node.js) is required for MCPB bundling"
    require "uv" "uv is required. Install: https://docs.astral.sh/uv/"
    [ -d "$MCP_DIR" ] || fail "MCP server not found at $MCP_DIR"

    printf "  Building MCPB bundle...\n"
    mkdir -p "$DIST_DIR"
    (cd "$MCP_DIR" && uv lock 2>/dev/null && npx @anthropic-ai/mcpb pack . "$DIST_DIR/")

    info "Bundle created in $DIST_DIR/"
    step "Double-click the .mcpb file to install in Claude Desktop"
}

# =============================================================================
# Claude Desktop — MCP via direct config
# =============================================================================

install_desktop_config() {
    require "jq" "jq is required. Install with: brew install jq"
    [ -d "$MCP_DIR" ] || fail "MCP server not found at $MCP_DIR"

    local target="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
    local target_dir
    target_dir=$(dirname "$target")

    # Absolute path required (Claude Desktop resolves from its own context)
    local config
    config=$(printf '{"command":"uv","args":["--directory","%s","run","main.py"]}' "$MCP_DIR")

    mkdir -p "$target_dir"
    if [ ! -f "$target" ] || ! jq -e . "$target" >/dev/null 2>&1; then
        echo '{"mcpServers":{}}' > "$target"
    fi
    jq --arg n "$SERVICE_NAME" --argjson c "$config" \
        '.mcpServers[$n] = $c' "$target" > "${target}.tmp"
    mv "${target}.tmp" "$target"
    chmod 600 "$target"

    info "Registered $SERVICE_NAME in Claude Desktop config"
}

# =============================================================================
# Claude Desktop — Skill ZIPs
# =============================================================================

build_skill_zips() {
    require "zip" "zip is required for building skill packages"
    mkdir -p "$DIST_DIR"

    for skill in "${SKILL_NAMES[@]}"; do
        local skill_dir="$SKILLS_DIR/$skill"
        if [ -d "$skill_dir" ]; then
            (cd "$skill_dir" && zip -rq "$DIST_DIR/$skill.zip" . -x "tests/*" -x "__pycache__/*" -x "*.pyc")
            info "Built $DIST_DIR/$skill.zip"
        else
            warn "Skill directory not found: $skill_dir (skipping)"
        fi
    done
}

# =============================================================================
# Claude Desktop — Top-level flow
# =============================================================================

install_desktop() {
    header "Claude Desktop Installation"

    # Step 1: MCP server
    header "Step 1 — MCP server"
    cat <<EOF

  ${B}[1] MCPB bundle${R}
      ${D}Builds a self-contained .mcpb package. Install by double-clicking.${R}
      ${D}Dependencies resolved at runtime by uv.${R}
      ${D}Requires: npx (Node.js), uv${R}

  ${B}[2] Direct config${R}
      ${D}Writes the server entry to Claude Desktop's JSON config file.${R}
      ${D}Server runs from this directory via uv at runtime.${R}
      ${D}Requires: jq, uv${R}
EOF
    ask 1 2

    if [ "$REPLY" -eq 1 ]; then
        install_desktop_mcpb
    else
        install_desktop_config
    fi

    # Step 2: Skills
    header "Step 2 — Skills"
    printf "  Building skill packages...\n"
    build_skill_zips

    printf "\n"
    step "To install skills in Claude Desktop:"
    step "  1. Open Claude Desktop → Settings → Capabilities → Skills"
    local i=2
    for skill in "${SKILL_NAMES[@]}"; do
        step "  $i. Click 'Upload skill' and select dist/$skill.zip"
        ((i++))
    done
    step "  $i. Restart Claude Desktop"

    printf "\n"
    info "Installation complete"
}

# =============================================================================
# Uninstall — Claude Code
# =============================================================================

uninstall_code() {
    header "Uninstalling from Claude Code"
    cat <<EOF

  How was Stockpile installed?

  ${B}[1] Plugin${R}
  ${B}[2] Direct MCP registration (user scope)${R}
  ${B}[3] Direct MCP registration (project .mcp.json)${R}
EOF
    ask 1 3

    case "$REPLY" in
        1)
            require "claude" "claude CLI is required"
            printf "  Uninstalling plugin...\n"
            claude plugin uninstall "$PLUGIN_NAME" 2>/dev/null \
                && info "Plugin removed" \
                || warn "Plugin not found (skipping)"
            printf "  Removing marketplace...\n"
            claude plugin marketplace remove "$MARKETPLACE_NAME" 2>/dev/null \
                && info "Marketplace removed" \
                || warn "Marketplace not found (skipping)"
            ;;
        2)
            require "claude" "claude CLI is required"
            claude mcp remove "$SERVICE_NAME" \
                && info "MCP server removed (user scope)" \
                || warn "MCP server not found"
            ;;
        3)
            require "jq" "jq is required"
            local target="$SCRIPT_DIR/.mcp.json"
            if [ -f "$target" ] && jq -e ".mcpServers.\"$SERVICE_NAME\"" "$target" >/dev/null 2>&1; then
                jq --arg n "$SERVICE_NAME" 'del(.mcpServers[$n])' "$target" > "${target}.tmp"
                mv "${target}.tmp" "$target"
                info "Removed $SERVICE_NAME from .mcp.json"
            else
                warn "$SERVICE_NAME not found in .mcp.json"
            fi
            ;;
    esac
}

# =============================================================================
# Uninstall — Claude Desktop
# =============================================================================

uninstall_desktop() {
    header "Uninstalling from Claude Desktop"

    require "jq" "jq is required"
    local target="$HOME/Library/Application Support/Claude/claude_desktop_config.json"

    if [ -f "$target" ] && jq -e ".mcpServers.\"$SERVICE_NAME\"" "$target" >/dev/null 2>&1; then
        jq --arg n "$SERVICE_NAME" 'del(.mcpServers[$n])' "$target" > "${target}.tmp"
        mv "${target}.tmp" "$target"
        chmod 600 "$target"
        info "Removed $SERVICE_NAME from Claude Desktop config"
    else
        warn "$SERVICE_NAME not found in Claude Desktop config"
    fi

    step "To remove skills: Claude Desktop → Settings → Capabilities → Skills"
    step "Restart Claude Desktop for changes to take effect"
}

# =============================================================================
# Usage
# =============================================================================

show_usage() {
    cat <<EOF
Usage: $0 [code|desktop] [--uninstall]

  code       Install for Claude Code (default)
  desktop    Install for Claude Desktop
  --uninstall  Remove installation
EOF
    exit 1
}

# =============================================================================
# Main
# =============================================================================

MODE="code"
UNINSTALL=false

while [ $# -gt 0 ]; do
    case "$1" in
        code|desktop) MODE="$1" ;;
        --uninstall)  UNINSTALL=true ;;
        -h|--help)    show_usage ;;
        *)            fail "Unknown argument: $1. Use --help for usage." ;;
    esac
    shift
done

show_overview

if [ "$UNINSTALL" = true ]; then
    case "$MODE" in
        code)    uninstall_code ;;
        desktop) uninstall_desktop ;;
    esac
else
    case "$MODE" in
        code)    install_code ;;
        desktop) install_desktop ;;
    esac
fi
