# Rust src

- Prioritize testability and shutdown safety
- Avoid re-exports; just import from the nested directory
- Import structs at the top of the file rather than using long qualified names
- Avoid modules with underscores; use short module names and group related modules thematically
- Avoid long struct names; use the module as a prefix if needed to disambiguate two similar structs
- Avoid panics in runtime code
- Use lock-free approaches whenever possible
