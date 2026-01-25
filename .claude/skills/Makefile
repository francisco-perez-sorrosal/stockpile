# Makefile for Claude skills
# Usage: make SKILL=<skill_name> <target>
#
# Examples:
#   make SKILL=foo build
#   make SKILL=foo install
#   make SKILL=foo clean

ifndef SKILL
$(error SKILL is required. Usage: make SKILL=<skill_name> <target>)
endif

SKILL_DIR := $(SKILL)
DIST_DIR := ../dist
ZIP_FILE := $(DIST_DIR)/$(SKILL).zip

.PHONY: all clean build install list help

help:
	@echo "Makefile for Claude skills"
	@echo ""
	@echo "Usage: make SKILL=<skill_name> <target>"
	@echo ""
	@echo "Targets:"
	@echo "  build    - Create zip file for Claude Desktop installation"
	@echo "  clean    - Remove generated files"
	@echo "  list     - List contents of the skill directory"
	@echo "  install  - Copy skill to ~/.claude/skills/"
	@echo "  help     - Show this help message"

all: build

build: $(ZIP_FILE)

$(ZIP_FILE): $(SKILL_DIR)/SKILL.md
	@mkdir -p $(DIST_DIR)
	@cd $(SKILL_DIR) && zip -r ../$(ZIP_FILE) . -x Makefile
	@echo "Created $(ZIP_FILE)"

clean:
	@rm -f $(ZIP_FILE)
	@echo "Cleaned $(SKILL) artifacts"

list:
	@echo "Contents of $(SKILL) skill:"
	@ls -la $(SKILL_DIR)

install: build
	@echo "Installing $(SKILL) to Claude Code..."
	@mkdir -p ~/.claude/skills/$(SKILL)
	@unzip -o $(ZIP_FILE) -d ~/.claude/skills/$(SKILL)/
	@echo "Installed $(SKILL) skill"
