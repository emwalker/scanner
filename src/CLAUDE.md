# Rust src

- Prioritize testability and shutdown safety
- Avoid re-exports; just import from the nested directory
- Avoid long, nested modules when referring to imported structs and functions
- Avoid modules with underscores; use short module names and group related modules thematically
- Avoid long struct names; use the module as a prefix if needed to disambiguate two similar structs
- Avoid panics in runtime code if possible
- Use lock-free approaches whenever possible
