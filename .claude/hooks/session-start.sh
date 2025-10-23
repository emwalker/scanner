#!/bin/bash

# SessionStart hook to auto-load rust-analyzer skill for Rust projects

cat << 'RUST_ANALYZER_REMINDER'
IMPORTANT: This is a Rust project with rust-analyzer MCP tools available.

MANDATORY Tool Usage Rules:
- Looking for symbol usages? → Use find_references (NOT Grep or Bash(grep))
- Renaming something? → Use rename_symbol (NOT find/replace or manual editing)
- Just made changes? → Use get_diagnostics (NOT cargo build)
- Need to organize imports? → Use organize_imports (NOT manual editing)
- Searching for symbols? → Use workspace_symbols (NOT Grep)
- Finding definitions? → Use find_definition (NOT Grep)
- Refactoring code? → Use extract_function, inline_function, move_items (NOT manual editing)

The rust-analyzer MCP provides semantic understanding of Rust code. Use these tools BEFORE falling back to Grep or Bash commands.

Skill available: Use Skill tool with "rust-analyzer-tools" for complete guidance.
RUST_ANALYZER_REMINDER
