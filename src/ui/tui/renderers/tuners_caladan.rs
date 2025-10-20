//! Tuner enumeration display for Caladan theme

use crate::ui::tui::{
    colors::ACTIVE_STATE_GREEN,
    model::{FocusState, Model},
    themes::Theme,
};
use ratatui::{
    Frame,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph},
};

pub fn render_tuners(f: &mut Frame, area: Rect, model: &Model, theme: &dyn Theme) {
    if area.height < 4 || area.width < 20 {
        return;
    }

    let bracket_color = ratatui::style::Color::Rgb(160, 200, 220);

    let tuners = model.tuner_list();
    if tuners.is_empty() {
        render_no_devices(f, area, theme, bracket_color);
        return;
    }

    let mut y_offset = 0;

    for (tuner_idx, tuner_info) in tuners.iter().enumerate() {
        if y_offset + 5 > area.height {
            return;
        }

        let tuner_area = Rect {
            x: area.x,
            y: area.y + y_offset,
            width: area.width,
            height: 5,
        };

        let has_focus = matches!(model.focus_state, FocusState::Tuner(i) if i == tuner_idx);
        render_tuner_block(f, tuner_area, tuner_info, theme, bracket_color, has_focus);
        y_offset += 5;
    }
}

fn render_no_devices(
    f: &mut Frame,
    area: Rect,
    theme: &dyn Theme,
    bracket_color: ratatui::style::Color,
) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(
            Style::default()
                .fg(bracket_color)
                .add_modifier(Modifier::DIM),
        )
        .border_set(ratatui::symbols::border::Set {
            top_left: "╭",
            top_right: "╮",
            bottom_left: "╰",
            bottom_right: "╯",
            vertical_left: " ",
            vertical_right: " ",
            horizontal_top: "─",
            horizontal_bottom: "─",
        })
        .padding(ratatui::widgets::Padding::horizontal(1));

    let inner = block.inner(area);

    let lines = vec![
        Line::from(vec![Span::styled(
            "No SDR devices detected",
            Style::default()
                .fg(theme.secondary())
                .add_modifier(Modifier::DIM),
        )]),
        Line::from(vec![Span::styled(
            "Waiting for device discovery...",
            Style::default()
                .fg(theme.secondary())
                .add_modifier(Modifier::DIM),
        )]),
    ];

    let paragraph = Paragraph::new(lines);

    f.render_widget(block, area);
    f.render_widget(paragraph, inner);
}

fn render_tuner_block(
    f: &mut Frame,
    area: Rect,
    tuner: &crate::ui::tui::model::TunerDisplayInfo,
    theme: &dyn Theme,
    bracket_color: ratatui::style::Color,
    has_focus: bool,
) {
    let border_style = if has_focus {
        Style::default()
            .fg(bracket_color)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
            .fg(bracket_color)
            .add_modifier(Modifier::DIM)
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(border_style)
        .border_set(ratatui::symbols::border::Set {
            top_left: "╭",
            top_right: "╮",
            bottom_left: "╰",
            bottom_right: "╯",
            vertical_left: " ",
            vertical_right: " ",
            horizontal_top: "─",
            horizontal_bottom: "─",
        })
        .padding(ratatui::widgets::Padding::horizontal(1));

    let inner = block.inner(area);

    let tuner_id_str = match &tuner.id.device_id {
        crate::hardware::DeviceId::Driver { driver, serial, .. } => {
            format!("{}:{}", driver, serial)
        }
        crate::hardware::DeviceId::Usb {
            vid, pid, serial, ..
        } => {
            format!("USB {:04x}:{:04x} ({})", vid, pid, serial)
        }
    };

    let status_style = match tuner.state {
        crate::ui::tui::model::TunerState::Listening
        | crate::ui::tui::model::TunerState::Scanning => Style::default()
            .fg(ACTIVE_STATE_GREEN)
            .add_modifier(Modifier::BOLD),
        crate::ui::tui::model::TunerState::Available => Style::default()
            .fg(theme.foreground())
            .add_modifier(Modifier::DIM),
    };

    let lines = vec![
        Line::from(vec![Span::styled(
            &tuner.label,
            Style::default()
                .fg(theme.primary())
                .add_modifier(Modifier::BOLD),
        )]),
        Line::from(vec![Span::styled(
            &tuner_id_str,
            Style::default().fg(theme.secondary()),
        )]),
        Line::from(vec![Span::styled(tuner.state.display(), status_style)]),
    ];

    let paragraph = Paragraph::new(lines);

    f.render_widget(block, area);
    f.render_widget(paragraph, inner);
}

#[cfg(test)]
mod tests {
    use crate::ui::tui::colors::ACTIVE_STATE_GREEN;
    use ratatui::style::Color;

    #[test]
    fn test_active_state_green_matches_scanning_listening_color() {
        assert_eq!(
            ACTIVE_STATE_GREEN,
            Color::Rgb(150, 255, 150),
            "ACTIVE_STATE_GREEN must match the color used for Scanning/Listening tuner states"
        );
    }
}
