//! Tuner table display

use ratatui::{
    Frame,
    layout::{Constraint, Rect},
    widgets::{Cell, Row, Table},
};

use super::table_styles::{self, AlwaysVisibleFilter, TableRenderer2, VisibilityContext};
use crate::ui::tui::{
    model::{Model, types::FocusedTable},
    themes::Theme,
};

pub fn render_tuners(f: &mut Frame, area: Rect, model: &mut Model, theme: &dyn Theme) {
    let has_focus = table_styles::check_focus(&model.focus_state, FocusedTable::Tuners);
    let block = table_styles::create_table_block("Tuners", has_focus, theme);

    let header = Row::new(vec![Cell::from("Name"), Cell::from("Activity")])
        .style(table_styles::header_style());

    let tuners = model.tuner_list();

    let viewport_height = area.height.saturating_sub(3) as usize;
    model.tuners_scroll.viewport_height = viewport_height;

    let visibility_context = VisibilityContext::new(None, None);

    let (rows, scrollbar_state) = {
        let mut renderer = TableRenderer2::new(
            &tuners,
            FocusedTable::Tuners,
            AlwaysVisibleFilter,
            &mut model.tuners_scroll,
        );

        renderer.render(&model.focus_state, theme, visibility_context)
    };

    let table = Table::new(rows, [Constraint::Min(20), Constraint::Length(10)])
        .header(header)
        .block(block)
        .column_spacing(2);

    f.render_widget(table, area);
    table_styles::render_scrollbar(f, area, &scrollbar_state, theme);
}
