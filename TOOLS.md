# MCP Integration with rust-analyzer

This project is configured to use the `rust-mcp` MCP server, which provides rust-analyzer integration for Claude Code.

## What is rust-mcp?

rust-mcp is a Model Context Protocol server that exposes rust-analyzer's capabilities through 19 tools, enabling AI assistants to understand Rust code structure, navigate definitions, analyze errors, and perform refactoring operations.

## Installation

The rust-mcp server is already built and configured for this project. The configuration is in `.mcp.json`.

### Manual Setup (if needed)

1. Build rust-mcp (already done):
   ```bash
   cd /home/walker/code/rust-mcp
   cargo build --release
   ```

2. Verify rust-analyzer is installed:
   ```bash
   rust-analyzer --version
   ```

3. The server binary is at: `/home/walker/code/rust-mcp/target/release/rustmcp`

## Available Tools

### Code Analysis (4 tools)
- `find_definition` - Navigate to symbol definitions
- `find_references` - Find all uses of a symbol
- `get_diagnostics` - Get compiler errors/warnings with suggested fixes
- `workspace_symbols` - Search for symbols across the entire project

### Code Generation (4 tools)
- `generate_struct` - Create structs with derives and constructors
- `generate_enum` - Create enums with variants
- `generate_trait_impl` - Generate trait implementations with method stubs
- `generate_tests` - Create unit or integration test templates

### Refactoring (5 tools)
- `rename_symbol` - Rename symbols with scope awareness
- `extract_function` - Extract code into separate functions
- `inline_function` - Inline function calls
- `organize_imports` - Sort and organize use statements
- `format_code` - Apply rustfmt formatting

### Quality Assurance (2 tools)
- `apply_clippy_suggestions` - Apply clippy automatic fixes
- `validate_lifetimes` - Check lifetime and borrow checker issues

### Project Management (2 tools)
- `analyze_manifest` - Parse and analyze Cargo.toml
- `run_cargo_check` - Execute cargo check with error parsing

### Advanced Features (2 tools)
- `get_type_hierarchy` - Get type relationships for symbols
- `suggest_dependencies` - Recommend crates based on code patterns

## Usage with Claude Code

Claude Code will automatically use these tools when analyzing Rust code. You can explicitly request their use with prompts like:

### Analysis Examples
- "Find all references to the `StationCandidate` struct"
- "Show me the definition of `calculate_audio_quality`"
- "Get diagnostics for src/main.rs"
- "Search for all symbols containing 'audio' in the workspace"

### Code Generation Examples
- "Generate a struct called `SignalMetrics` with fields for power, snr, and frequency"
- "Create unit tests for the audio quality calculation functions"
- "Generate a Display trait implementation for StationCandidate"

### Refactoring Examples
- "Rename `old_name` to `new_name` throughout the codebase"
- "Extract this signal processing code into a separate function"
- "Organize all imports in src/audio_quality/mod.rs"
- "Format the code in src/scanning/window/audio.rs"

### Quality Examples
- "Run clippy and apply fixes to src/ui/tui/model/mod.rs"
- "Check for lifetime issues in src/pipeline/mod.rs"

### Project Management Examples
- "Analyze Cargo.toml and show dependency information"
- "Run cargo check and report compilation errors"

## When to Use rust-analyzer Tools vs. Regular Tools

**Use rust-analyzer tools when:**
- You need to understand Rust-specific semantics (types, lifetimes, traits)
- You want to navigate code (definitions, references)
- You need compiler diagnostics with context
- You want to perform safe refactoring operations
- You need to understand type relationships

**Use regular tools (Grep, Read) when:**
- You're doing simple text searches
- You want to read file contents
- You're looking for patterns across multiple file types
- You need quick, lightweight searches

## Troubleshooting

### Server Not Found
If Claude Code can't find the MCP server, verify:
1. The binary exists: `ls -lh /home/walker/code/rust-mcp/target/release/rustmcp`
2. It's executable: `chmod +x /home/walker/code/rust-mcp/target/release/rustmcp`
3. The `.mcp.json` file has the correct path

### rust-analyzer Issues
If rust-analyzer isn't working:
1. Check it's installed: `rust-analyzer --version`
2. Verify the workspace has a valid Cargo.toml
3. Try running rust-analyzer directly on the project

### Slow First Run
The first time rust-analyzer analyzes this project, it may take 10-30 seconds to load all dependencies and build the project index. Subsequent queries will be much faster.

## Configuration File

The MCP configuration is in `.mcp.json` (gitignored). It points to:
- Server binary: `/home/walker/code/rust-mcp/target/release/rustmcp`
- rust-analyzer: `/home/walker/.cargo/bin/rust-analyzer`

To use a different rust-analyzer, update the `RUST_ANALYZER_PATH` in `.mcp.json`.
