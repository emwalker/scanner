//! Shared styling utilities for TUI renderers

use ratatui::{symbols::border, widgets::Borders};

/// Border set with rounded corners on top and bottom, no visible sides
pub fn rounded_horizontal_border() -> border::Set {
    border::Set {
        top_left: "╭",
        top_right: "╮",
        bottom_left: "╰",
        bottom_right: "╯",
        vertical_left: " ",
        vertical_right: " ",
        horizontal_top: "─",
        horizontal_bottom: "─",
    }
}

/// Borders configuration for horizontal-only borders with rounded corners
pub fn horizontal_borders() -> Borders {
    Borders::ALL
}
