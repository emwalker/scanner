# rust-analyzer Tool Selection Guide

This reference provides detailed guidance on selecting the appropriate rust-analyzer tool for different Rust development tasks.

## Tool Categories and When to Use Them

### Code Analysis Tools

**find_definition**
- Use when: Need to locate where a symbol is defined
- Examples: "Where is `AudioQualityMetrics` defined?", "Show me the implementation of this trait"
- Advantages over Grep: Handles Rust's module system, finds the actual definition not just text matches
- Returns: Precise file location and line number

**find_references**
- Use when: Need to find all uses of a symbol
- Examples: "Find all references to `calculate_snr`", "Where is this function called?"
- Advantages over Grep: Finds semantic references, not string matches; ignores comments and strings
- Returns: List of all locations where the symbol is used

**get_diagnostics**
- Use when: Investigating compiler errors, warnings, or lints
- Examples: "What errors are in src/main.rs?", "Why won't this compile?", "Show me clippy warnings"
- Advantages over `cargo check`: Provides structured error information with suggested fixes
- Returns: Detailed diagnostics with error codes, ranges, and suggestions

**workspace_symbols**
- Use when: Searching for symbols across the entire project
- Examples: "Find all structs containing 'audio'", "Search for functions matching 'calculate'"
- Advantages over Grep: Searches only actual symbol definitions, understands Rust syntax
- Returns: Symbol names, kinds (struct/enum/function/etc.), and locations

### Code Generation Tools

**generate_struct**
- Use when: Creating new struct definitions
- Examples: "Generate a struct for signal metrics", "Create a config struct"
- Provides: Proper derives, constructors, field validation

**generate_enum**
- Use when: Creating new enum types
- Examples: "Create an enum for scan states", "Generate an error enum"
- Provides: Variant definitions, pattern matching templates

**generate_trait_impl**
- Use when: Implementing traits for types
- Examples: "Generate Display impl for User", "Implement FromStr"
- Provides: Method stubs with proper signatures

**generate_tests**
- Use when: Creating test templates
- Examples: "Generate unit tests for this function", "Create integration tests"
- Provides: Properly structured test modules

### Refactoring Tools

**rename_symbol**
- Use when: Renaming variables, functions, types across codebase
- Examples: "Rename `data` to `user_input`", "Change `calc` to `calculate`"
- Advantages over find-replace: Scope-aware, won't rename in comments/strings
- Safety: Will fail if rename would cause conflicts

**extract_function**
- Use when: Moving code into a separate function
- Examples: "Extract this validation logic", "Move this into a helper function"
- Advantages: Handles parameter passing, return types automatically

**inline_function**
- Use when: Removing unnecessary function indirection
- Examples: "Inline this helper function", "Remove this wrapper"
- Advantages: Updates all call sites automatically

**organize_imports**
- Use when: Cleaning up use statements
- Examples: "Organize imports in this file", "Clean up the use statements"
- Provides: Sorted, grouped imports following Rust conventions

**format_code**
- Use when: Applying rustfmt formatting
- Examples: "Format this file", "Apply rustfmt"
- Advantages over manual rustfmt: Can be done without saving file

### Quality Assurance Tools

**apply_clippy_suggestions**
- Use when: Improving code quality with clippy
- Examples: "Apply clippy fixes", "Fix lint warnings"
- Advantages: Applies automatic fixes, doesn't just report

**validate_lifetimes**
- Use when: Investigating borrow checker or lifetime errors
- Examples: "Why is the borrow checker complaining?", "Check lifetime issues"
- Provides: Detailed lifetime analysis

### Project Management Tools

**analyze_manifest**
- Use when: Understanding project dependencies
- Examples: "What dependencies do we have?", "Analyze Cargo.toml"
- Returns: Dependency tree, version information

**run_cargo_check**
- Use when: Running full project compilation check
- Examples: "Does the project compile?", "Run cargo check"
- Returns: All compilation errors and warnings

### Advanced Tools

**get_type_hierarchy**
- Use when: Understanding trait implementations and type relationships
- Examples: "What traits does this type implement?", "Show type hierarchy"
- Returns: Implemented traits, supertypes, subtypes

**suggest_dependencies**
- Use when: Looking for appropriate crates
- Examples: "What crate should I use for HTTP?", "Suggest async runtime"
- Returns: Recommended crates based on patterns

## Decision Flow

### Starting an Investigation

1. **Known symbol, need location** → `find_definition`
2. **Known symbol, need usage** → `find_references`
3. **Compilation error** → `get_diagnostics` first, then dig deeper
4. **Unknown symbol, searching** → `workspace_symbols`

### Understanding Code

1. **What does this error mean?** → `get_diagnostics`
2. **What type is this?** → hover info (not yet in tool list, use definition)
3. **What traits does this implement?** → `get_type_hierarchy`
4. **Where is this used?** → `find_references`

### Making Changes

1. **Renaming** → Always use `rename_symbol`, never manual find-replace
2. **Extracting code** → Use `extract_function` for safety
3. **Cleaning up** → Use `organize_imports` and `format_code`
4. **Fixing lints** → Use `apply_clippy_suggestions`

## Common Patterns

### Pattern: Investigating an Error
1. Run `get_diagnostics` on the file
2. Examine error code and message
3. If unclear, use `find_definition` to understand types involved
4. Check `find_references` to see how similar code is used
5. Use `get_type_hierarchy` if trait bounds are involved

### Pattern: Refactoring Code
1. Use `find_references` to understand current usage
2. Apply `rename_symbol` or `extract_function` as needed
3. Run `cargo_check` to verify changes
4. Apply `organize_imports` and `format_code` for cleanup
5. Run `apply_clippy_suggestions` for final polish

### Pattern: Understanding Unfamiliar Code
1. Use `workspace_symbols` to find relevant types/functions
2. Use `find_definition` to locate implementations
3. Use `find_references` to see usage examples
4. Use `get_type_hierarchy` to understand trait relationships

## Comparison: rust-analyzer vs Standard Tools

| Task | Standard Tool | rust-analyzer Tool | Advantage |
|------|---------------|-------------------|-----------|
| Find function definition | Grep for "fn name" | find_definition | Handles modules, traits, impls |
| Find all uses | Grep for "name" | find_references | Ignores comments, strings, other contexts |
| Check errors | cargo check | get_diagnostics | Structured data, suggested fixes |
| Rename variable | find + sed | rename_symbol | Scope-aware, safer |
| Search symbols | Grep | workspace_symbols | Syntax-aware, no false positives |
| Format code | rustfmt | format_code | In-process, no file save needed |

## When NOT to Use rust-analyzer

- **Simple text search across multiple file types** → Use Grep
- **Reading file contents** → Use Read tool
- **Quick string matching** → Use Grep
- **Non-Rust files** → Use standard tools
- **Performance-critical searches** → Grep is faster for simple patterns
