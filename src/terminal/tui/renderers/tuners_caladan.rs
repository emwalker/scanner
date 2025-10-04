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

    let tuners = [
        TunerInfo {
            device_name: "SDRplay RSPduo",
            tuner_id: "Tuner 1",
            antenna_port: "50Ω Port A",
            freq_range: "1 kHz - 2 GHz",
            bandwidth: "10 MHz",
        },
        TunerInfo {
            device_name: "SDRplay RSPduo",
            tuner_id: "Tuner 2",
            antenna_port: "50Ω Port B",
            freq_range: "1 kHz - 2 GHz",
            bandwidth: "2 MHz",
        },
    ];

    let mut y_offset = 0;
    for (idx, tuner) in tuners.iter().enumerate() {
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
        render_tuner_block(f, tuner_area, tuner, theme, bracket_color, has_focus);
        y_offset += 5;
    }
}

struct TunerInfo {
    device_name: &'static str,
    tuner_id: &'static str,
    antenna_port: &'static str,
    freq_range: &'static str,
    bandwidth: &'static str,
}

fn render_tuner_block(
    f: &mut Frame,
    area: Rect,
    tuner: &TunerInfo,
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

    let lines = vec![
        Line::from(vec![
            Span::styled(
                tuner.tuner_id,
                Style::default()
                    .fg(theme.primary())
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" • {}", tuner.device_name),
                Style::default().fg(theme.foreground()),
            ),
        ]),
        Line::from(vec![Span::styled(
            format!("{} • {}", tuner.antenna_port, tuner.freq_range),
            Style::default().fg(theme.secondary()),
        )]),
        Line::from(vec![Span::styled(
            format!("BW: {}", tuner.bandwidth),
            Style::default().fg(theme.secondary()),
        )]),
    ];

    let paragraph = Paragraph::new(lines);

    f.render_widget(block, area);
    f.render_widget(paragraph, inner);
}
