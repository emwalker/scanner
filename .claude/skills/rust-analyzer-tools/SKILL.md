---
name: rust-analyzer-tools
description: Use when working with Rust code to leverage rust-analyzer's semantic understanding for navigation, error analysis, refactoring, and code generation. Apply this skill when investigating compiler errors, navigating Rust code structure, performing safe refactoring, or generating Rust code. Prefer rust-analyzer tools over Grep/Read when Rust-specific semantics matter (types, traits, lifetimes, scope).
---

# rust-analyzer Tools

## Overview

Leverage rust-analyzer's Language Server Protocol capabilities through MCP tools to provide deep semantic understanding of Rust code. These tools enable precise code navigation, compiler error analysis, safe refactoring, and intelligent code generation that understands Rust's type system, borrow checker, and module structure.

## When to Use This Skill

Apply this skill when:

- **Investigating compiler errors** - Understanding E-codes, borrow checker errors, trait bound issues
- **Navigating Rust code** - Finding definitions, references, or implementations across modules
- **Understanding types** - Exploring trait implementations, type hierarchies, or lifetime relationships
- **Refactoring safely** - Renaming symbols, extracting functions, or reorganizing code with semantic awareness
- **Generating Rust code** - Creating structs, enums, trait implementations, or tests following Rust conventions
- **Analyzing project structure** - Searching symbols, analyzing dependencies, or checking compilation status

Do NOT use this skill for:
- Simple text searches across multiple file types (use Grep)
- Reading file contents (use Read)
- Quick string matching (use Grep)
- Non-Rust files

## Core Capabilities

### 1. Error Investigation and Diagnostics

When encountering compiler errors, warnings, or lints, start with diagnostics to get structured error information.

**Primary tool: `get_diagnostics`**

Use `get_diagnostics` to:
- Retrieve compiler errors with error codes (E0425, E0308, etc.)
- Get structured error information with file ranges
- See suggested fixes from the compiler
- Identify clippy lints and warnings

Example workflow:
1. Run `get_diagnostics` on the file with errors
2. Examine error codes and messages for root cause
3. If types are unclear, use `find_definition` to understand involved types
4. Check `find_references` to see how similar code patterns work
5. Use `get_type_hierarchy` if trait bounds are causing issues

**Common error scenarios:**
- "Cannot find value in scope" (E0425) → Use `workspace_symbols` to locate the symbol
- "Mismatched types" (E0308) → Use `get_type_hierarchy` to understand trait relationships
- "Lifetime errors" → Use `validate_lifetimes` for detailed lifetime analysis
- "Trait bound not satisfied" → Use `get_type_hierarchy` to see what traits are implemented

### 2. Code Navigation

Navigate Rust code using semantic understanding, not text matching.

**find_definition** - Locate where a symbol is defined
- Handles complex module structures
- Finds trait implementations, not just trait definitions
- Works across crate boundaries
- Example: "Where is `AudioQualityMetrics` defined?"

**find_references** - Find all uses of a symbol
- Semantic references only (ignores comments, strings)
- Includes method calls, struct instantiation, imports
- Scope-aware (distinguishes local vs module-level)
- Example: "Find all references to `calculate_snr`"

**workspace_symbols** - Search for symbols across the project
- Searches only actual definitions (structs, enums, functions, traits)
- Supports partial matching
- Returns symbol kind (struct/enum/function/etc.)
- Example: "Find all structs containing 'audio'"

### 3. Safe Refactoring

Perform refactoring operations with semantic awareness to avoid breaking changes.

**rename_symbol** - Rename variables, functions, types
- Scope-aware: won't rename unrelated symbols with same name
- Updates all references automatically
- Checks for naming conflicts before applying
- Updates imports and qualified paths
- Example: "Rename `old_var` to `new_var` throughout the codebase"

**extract_function** - Move code into a separate function
- Analyzes data flow to determine parameters
- Infers return type from extracted code
- Handles mutable references correctly
- Updates the original location with function call
- Example: "Extract this validation logic into a function"

**inline_function** - Remove unnecessary function indirection
- Replaces all call sites with function body
- Handles parameter substitution
- Preserves semantics
- Example: "Inline this wrapper function"

**organize_imports** - Clean up use statements
- Groups imports by category (std, external crates, local)
- Removes unused imports
- Sorts alphabetically within groups
- Follows Rust conventions
- Example: "Organize imports in src/audio_quality/mod.rs"

**format_code** - Apply rustfmt formatting
- Formats according to project's rustfmt.toml if present
- Can format specific ranges or entire files
- Example: "Format the code in src/main.rs"

### 4. Code Generation

Generate Rust code that follows conventions and best practices.

**generate_struct** - Create struct definitions
- Adds appropriate derives (Debug, Clone, etc.)
- Creates constructor methods (new, default)
- Handles generic parameters
- Example: "Generate a struct for signal metrics with power, snr, and frequency fields"

**generate_enum** - Create enum types
- Generates variants with appropriate data
- Adds derive macros
- Creates helper methods (is_*, as_*, etc.)
- Example: "Create an enum for scan states"

**generate_trait_impl** - Generate trait implementations
- Creates method stubs with proper signatures
- Includes trait bounds
- Provides TODO comments for implementation
- Example: "Generate Display impl for StationCandidate"

**generate_tests** - Create test templates
- Generates properly structured test modules
- Includes common test patterns
- Sets up unit or integration test structure
- Example: "Create unit tests for audio quality calculation"

### 5. Type Understanding

Explore type relationships and trait implementations.

**get_type_hierarchy** - Understand type relationships
- Shows implemented traits
- Displays supertraits and subtraits
- Useful for understanding trait bounds
- Example: "What traits does `StationCandidate` implement?"

**validate_lifetimes** - Analyze lifetime and borrow issues
- Detailed lifetime analysis
- Explains borrow checker errors
- Shows lifetime constraints
- Example: "Check for lifetime issues in src/pipeline/mod.rs"

### 6. Project Management

Analyze project structure and dependencies.

**analyze_manifest** - Parse Cargo.toml
- Shows dependency tree
- Identifies version conflicts
- Displays features and build configuration
- Example: "Analyze Cargo.toml dependencies"

**run_cargo_check** - Execute compilation check
- Runs full project check
- Returns all compilation errors
- Parses error output for structured information
- Example: "Does the project compile?"

**suggest_dependencies** - Recommend crates
- Suggests crates based on code patterns
- Provides recommendations for common tasks
- Example: "What crate should I use for async HTTP?"

### 7. Quality Assurance

Improve code quality using automated tools.

**apply_clippy_suggestions** - Apply clippy fixes
- Automatically applies clippy suggestions
- Improves code quality and idiomaticity
- Handles multiple lints at once
- Example: "Apply clippy fixes to src/main.rs"

## Decision Tree: Choosing the Right Tool

```
Starting point: What do you need to do?

├─ Understand an error?
│  ├─ Compiler error/warning → get_diagnostics
│  ├─ Lifetime/borrow error → validate_lifetimes
│  └─ Missing symbol → workspace_symbols (to find it)
│
├─ Navigate code?
│  ├─ Find where defined → find_definition
│  ├─ Find where used → find_references
│  └─ Search for symbol → workspace_symbols
│
├─ Understand types?
│  ├─ What traits implemented? → get_type_hierarchy
│  ├─ Type relationships → get_type_hierarchy
│  └─ Borrow checker issues → validate_lifetimes
│
├─ Refactor code?
│  ├─ Rename → rename_symbol (NEVER manual find-replace)
│  ├─ Extract code → extract_function
│  ├─ Remove indirection → inline_function
│  ├─ Clean imports → organize_imports
│  └─ Format → format_code
│
├─ Generate code?
│  ├─ Struct → generate_struct
│  ├─ Enum → generate_enum
│  ├─ Trait impl → generate_trait_impl
│  └─ Tests → generate_tests
│
├─ Check project?
│  ├─ Compilation → run_cargo_check
│  ├─ Dependencies → analyze_manifest
│  └─ Code quality → apply_clippy_suggestions
│
└─ Simple text search? → Use Grep (not rust-analyzer)
```

## Common Workflows

### Workflow 1: Investigating a Compiler Error

1. Run `get_diagnostics` on the file to see structured error information
2. Identify the error code (E0425, E0308, etc.) and read the message
3. If the error involves unknown types, use `find_definition` to locate type definitions
4. If trait bounds are involved, use `get_type_hierarchy` to understand implementations
5. Check `find_references` to see how similar patterns are used elsewhere
6. Apply fixes and verify with `run_cargo_check`

### Workflow 2: Understanding Unfamiliar Code

1. Use `workspace_symbols` to find relevant types and functions
2. Use `find_definition` to locate implementations
3. Use `find_references` to see usage patterns
4. Use `get_type_hierarchy` to understand trait relationships
5. Use `get_diagnostics` if there are compilation issues

### Workflow 3: Safe Refactoring

1. Use `find_references` to understand current usage patterns
2. Apply refactoring: `rename_symbol`, `extract_function`, or `inline_function`
3. Run `run_cargo_check` to verify no breakage
4. Use `organize_imports` to clean up use statements
5. Apply `format_code` for consistent style
6. Run `apply_clippy_suggestions` for final improvements

### Workflow 4: Adding New Code

1. Use `workspace_symbols` to check for existing similar types
2. Generate scaffolding: `generate_struct`, `generate_enum`, or `generate_trait_impl`
3. Use `generate_tests` to create test templates
4. Implement functionality
5. Run `get_diagnostics` to check for errors
6. Run `apply_clippy_suggestions` for quality
7. Use `format_code` for final formatting

## Best Practices

### When to Use rust-analyzer vs Standard Tools

**Prefer rust-analyzer tools when:**
- Working with Rust-specific concepts (traits, lifetimes, types)
- Need semantic understanding (what is this, not just where is this text)
- Performing safe refactoring (rename, extract, inline)
- Investigating compiler errors with context
- Generating Rust code following conventions

**Prefer standard tools (Grep, Read) when:**
- Simple text search across multiple file types
- Quick string matching without semantic meaning
- Reading file contents directly
- Working with non-Rust files
- Performance-critical simple searches

### Refactoring Safety

**ALWAYS use `rename_symbol` for renaming** - Never use manual find-replace or Grep + Edit for renaming Rust symbols. The `rename_symbol` tool is scope-aware and will prevent accidental renaming of unrelated symbols.

**Verify after refactoring** - After any refactoring operation, run `run_cargo_check` to ensure no compilation errors were introduced.

**Start with references** - Before refactoring, use `find_references` to understand the full scope of changes.

### Error Investigation Strategy

**Start with diagnostics** - Always begin error investigation with `get_diagnostics` to get structured information rather than reading compiler output from terminal.

**Use type hierarchy for trait errors** - When encountering "trait bound not satisfied" errors, use `get_type_hierarchy` to understand what traits are actually implemented.

**Leverage suggestions** - The `get_diagnostics` tool often includes suggested fixes from the compiler—apply these when appropriate.

## References

For detailed tool selection guidance including comparison tables and decision flows, see:
- `references/tool_selection.md` - Comprehensive tool selection guide

This reference includes:
- Detailed descriptions of all 19 tools
- When to use each tool with examples
- Decision flow for common scenarios
- Comparison table: rust-analyzer vs standard tools
- Common patterns and workflows
