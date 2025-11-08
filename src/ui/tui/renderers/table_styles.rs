use std::hash::Hash;

use ratatui::{
    layout::Alignment,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Cell, Row, Table},
};

use super::styles;
use crate::ui::tui::{model::types::FocusedTable, themes::Theme};

pub fn create_table_block(title: &str, has_focus: bool, theme: &dyn Theme) -> Block<'static> {
    let border_style = if has_focus {
        Style::default()
            .fg(theme.accent())
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
            .fg(theme.secondary())
            .add_modifier(Modifier::DIM)
    };

    let title_line = Line::from(vec![
        Span::styled("─", border_style),
        Span::raw(title.to_string()),
    ]);

    Block::default()
        .title(title_line)
        .title_alignment(Alignment::Left)
        .borders(styles::horizontal_borders())
        .border_style(border_style)
        .border_set(styles::rounded_horizontal_border())
}

pub fn header_style() -> Style {
    Style::default().add_modifier(Modifier::BOLD)
}

pub fn row_style(is_selected: bool, theme: &dyn Theme) -> Style {
    if is_selected {
        Style::default().bg(theme.selection_bg())
    } else {
        Style::default()
    }
}

pub fn check_focus(
    focus_state: &crate::ui::tui::model::types::FocusState,
    table: FocusedTable,
) -> bool {
    matches!(focus_state.focused_table(), ft if ft == table)
}

pub fn apply_column_spacing<'a, T>(table: Table<'a>) -> Table<'a>
where
    T: Into<ratatui::widgets::Row<'a>>,
{
    table.column_spacing(2)
}

#[derive(Debug, Clone)]
pub struct ScrollState {
    pub offset: usize,
    pub viewport_height: usize,
}

impl ScrollState {
    pub fn new(viewport_height: usize) -> Self {
        Self {
            offset: 0,
            viewport_height,
        }
    }

    pub fn adjust_for_selection(&mut self, selected_index: usize, _total_visible: usize) {
        if selected_index < self.offset {
            self.offset = selected_index;
        } else if selected_index >= self.offset + self.viewport_height {
            self.offset = selected_index.saturating_sub(self.viewport_height.saturating_sub(1));
        }
    }
}

impl Default for ScrollState {
    fn default() -> Self {
        Self::new(10)
    }
}

#[derive(Debug, Clone)]
pub struct ScrollbarState {
    pub position: usize,
    pub total_items: usize,
    pub viewport_height: usize,
    pub should_render: bool,
}

impl ScrollbarState {
    pub fn new(position: usize, total_items: usize, viewport_height: usize) -> Self {
        Self {
            position,
            total_items,
            viewport_height,
            should_render: total_items > viewport_height,
        }
    }
}

pub trait TableRow {
    fn build_cells(&self, theme: &dyn Theme) -> Vec<Cell<'static>>;

    fn special_style(&self, _theme: &dyn Theme) -> Option<Style> {
        None
    }
}

pub trait RowGroup {
    type GroupId: Eq + Hash + Clone;
    fn group_id(&self) -> Self::GroupId;
}

#[derive(Debug, Clone)]
pub struct VisibilityContext<G> {
    pub selected_group: Option<G>,
    pub active_group: Option<G>,
}

impl<G> VisibilityContext<G> {
    pub fn new(selected_group: Option<G>, active_group: Option<G>) -> Self {
        Self {
            selected_group,
            active_group,
        }
    }
}

pub trait VisibilityFilter<T: RowGroup> {
    fn is_visible(&self, item: &T, context: &VisibilityContext<T::GroupId>) -> bool;
}

#[derive(Debug, Clone, Copy)]
pub struct AlwaysVisibleFilter;

impl<T: RowGroup> VisibilityFilter<T> for AlwaysVisibleFilter {
    fn is_visible(&self, _item: &T, _context: &VisibilityContext<T::GroupId>) -> bool {
        true
    }
}

pub struct TableRenderer2<'a, T, F>
where
    T: TableRow + RowGroup,
    F: VisibilityFilter<T>,
{
    items: &'a [T],
    focused_table: FocusedTable,
    filter: F,
    scroll_state: &'a mut ScrollState,
}

impl<'a, T, F> TableRenderer2<'a, T, F>
where
    T: TableRow + RowGroup,
    F: VisibilityFilter<T>,
{
    pub fn new(
        items: &'a [T],
        focused_table: FocusedTable,
        filter: F,
        scroll_state: &'a mut ScrollState,
    ) -> Self {
        Self {
            items,
            focused_table,
            filter,
            scroll_state,
        }
    }

    fn get_selected_index(
        &self,
        focus_state: &crate::ui::tui::model::types::FocusState,
    ) -> Option<usize> {
        match (focus_state, self.focused_table) {
            (
                crate::ui::tui::model::types::FocusState::Activities(selected),
                FocusedTable::Activities,
            ) => Some(*selected),
            (
                crate::ui::tui::model::types::FocusState::TunersTable(selected),
                FocusedTable::Tuners,
            ) => Some(*selected),
            (
                crate::ui::tui::model::types::FocusState::ScanProgress(selected),
                FocusedTable::ScanProgress,
            ) => Some(*selected),
            (
                crate::ui::tui::model::types::FocusState::SignalsTable(selected),
                FocusedTable::SignalsTable,
            ) => Some(*selected),
            _ => None,
        }
    }

    fn map_flat_to_visible(&self, flat_index: usize, visible_items: &[(usize, &T)]) -> usize {
        visible_items
            .iter()
            .position(|(idx, _)| *idx == flat_index)
            .unwrap_or(0)
    }

    fn is_row_selected(
        &self,
        focus_state: &crate::ui::tui::model::types::FocusState,
        flat_index: usize,
    ) -> bool {
        self.get_selected_index(focus_state)
            .map(|sel| sel == flat_index)
            .unwrap_or(false)
    }

    fn selected_row_style(&self, theme: &dyn Theme) -> Style {
        Style::default()
            .bg(theme.selected_row_bg())
            .fg(theme.selected_row_fg())
    }

    pub fn render(
        &mut self,
        focus_state: &crate::ui::tui::model::types::FocusState,
        theme: &dyn Theme,
        visibility_context: VisibilityContext<T::GroupId>,
    ) -> (Vec<Row<'static>>, ScrollbarState) {
        let visible_items: Vec<_> = self
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| self.filter.is_visible(item, &visibility_context))
            .collect();

        if let Some(selected_flat_index) = self.get_selected_index(focus_state) {
            let selected_visible_index =
                self.map_flat_to_visible(selected_flat_index, &visible_items);
            self.scroll_state
                .adjust_for_selection(selected_visible_index, visible_items.len());
        }

        let scrollbar_state = ScrollbarState::new(
            self.scroll_state.offset,
            visible_items.len(),
            self.scroll_state.viewport_height,
        );

        let rows = visible_items
            .iter()
            .skip(self.scroll_state.offset)
            .take(self.scroll_state.viewport_height)
            .map(|(flat_idx, item)| {
                let is_selected = self.is_row_selected(focus_state, *flat_idx);

                let style = if is_selected {
                    self.selected_row_style(theme)
                } else if let Some(special) = item.special_style(theme) {
                    special
                } else {
                    row_style(false, theme)
                };

                let cells = item.build_cells(theme);
                Row::new(cells).style(style)
            })
            .collect();

        (rows, scrollbar_state)
    }
}

pub fn render_scrollbar(
    f: &mut ratatui::Frame,
    area: ratatui::layout::Rect,
    scrollbar_state: &ScrollbarState,
    theme: &dyn crate::ui::tui::themes::Theme,
) {
    use ratatui::{
        layout::Margin,
        style::Style,
        widgets::{Scrollbar, ScrollbarOrientation, ScrollbarState as RatatuiScrollbarState},
    };

    if !scrollbar_state.should_render {
        return;
    }

    let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
        .style(Style::default().fg(theme.secondary()));

    let scrollbar_area = area.inner(Margin {
        horizontal: 0,
        vertical: 1,
    });

    let scrollable_range = scrollbar_state
        .total_items
        .saturating_sub(scrollbar_state.viewport_height);
    let mut ratatui_state =
        RatatuiScrollbarState::new(scrollable_range).position(scrollbar_state.position);

    f.render_stateful_widget(scrollbar, scrollbar_area, &mut ratatui_state);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scroll_state_no_adjustment_when_selection_in_viewport() {
        let mut scroll = ScrollState::new(10);
        scroll.offset = 5;

        scroll.adjust_for_selection(7, 20);

        assert_eq!(scroll.offset, 5);
    }

    #[test]
    fn test_scroll_state_adjusts_down_when_selection_below_viewport() {
        let mut scroll = ScrollState::new(10);
        scroll.offset = 0;

        scroll.adjust_for_selection(15, 20);

        assert_eq!(scroll.offset, 6);
    }

    #[test]
    fn test_scroll_state_adjusts_up_when_selection_above_viewport() {
        let mut scroll = ScrollState::new(10);
        scroll.offset = 10;

        scroll.adjust_for_selection(5, 20);

        assert_eq!(scroll.offset, 5);
    }

    #[test]
    fn test_scroll_state_zero_viewport_height_with_selection() {
        let mut scroll = ScrollState::new(0);
        scroll.offset = 0;

        // This used to panic with "attempt to subtract with overflow"
        // when viewport_height was 0 and selection was beyond offset
        scroll.adjust_for_selection(5, 100);

        assert_eq!(scroll.offset, 5);
    }

    #[test]
    fn test_scroll_state_zero_viewport_height_selection_in_range() {
        let mut scroll = ScrollState::new(0);
        scroll.offset = 3;

        // Selection within offset (even though viewport is 0)
        scroll.adjust_for_selection(2, 100);

        assert_eq!(scroll.offset, 2);
    }

    #[test]
    fn test_scroll_state_one_viewport_height() {
        let mut scroll = ScrollState::new(1);
        scroll.offset = 0;

        // With viewport height of 1, selecting index 5 should set offset to 5
        scroll.adjust_for_selection(5, 100);

        assert_eq!(scroll.offset, 5);
    }

    #[test]
    fn test_scroll_state_very_small_viewport() {
        let mut scroll = ScrollState::new(2);
        scroll.offset = 0;

        // With viewport height of 2, selecting at index 5 should position
        // so index 5 is visible at the bottom
        scroll.adjust_for_selection(5, 100);

        // offset should be 5 - (2 - 1) = 4
        assert_eq!(scroll.offset, 4);
    }

    #[test]
    fn test_always_visible_filter() {
        struct TestRow;
        impl RowGroup for TestRow {
            type GroupId = ();

            fn group_id(&self) -> Self::GroupId {}
        }

        let filter = AlwaysVisibleFilter;
        let item = TestRow;
        let context = VisibilityContext::new(None, None);

        assert!(filter.is_visible(&item, &context));
    }
}
