//! Tuner enumeration display for Caladan theme

use crate::terminal::tui::{
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

    // Use discovered devices if available, otherwise show "No devices" message
    if model.devices.is_empty() {
        render_no_devices(f, area, theme, bracket_color);
        return;
    }

    let mut y_offset = 0;
    for (idx, device) in model.devices.iter().enumerate() {
        if y_offset + 5 > area.height {
            break;
        }

        let tuner_area = Rect {
            x: area.x,
            y: area.y + y_offset,
            width: area.width,
            height: 5,
        };

        let has_focus = matches!(model.focus_state, FocusState::Tuner(i) if i == idx);
        render_tuner_block(f, tuner_area, device, theme, bracket_color, has_focus);
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
    device: &crate::sdr::DeviceInfo,
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

    // Extract device info from label and ID
    let device_id_str = match &device.id {
        crate::sdr::DeviceId::Backend { backend, serial } => {
            format!("{}:{}", backend, serial)
        }
        crate::sdr::DeviceId::Usb {
            vid, pid, serial, ..
        } => {
            format!("USB {:04x}:{:04x} ({})", vid, pid, serial)
        }
    };

    let lines = vec![
        Line::from(vec![Span::styled(
            &device.label,
            Style::default()
                .fg(theme.primary())
                .add_modifier(Modifier::BOLD),
        )]),
        Line::from(vec![Span::styled(
            &device_id_str,
            Style::default().fg(theme.secondary()),
        )]),
        Line::from(vec![Span::styled(
            "Ready",
            Style::default()
                .fg(theme.foreground())
                .add_modifier(Modifier::DIM),
        )]),
    ];

    let paragraph = Paragraph::new(lines);

    f.render_widget(block, area);
    f.render_widget(paragraph, inner);
}
