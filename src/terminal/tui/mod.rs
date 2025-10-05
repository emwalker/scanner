//! TUI module using The Elm Architecture pattern

use crate::terminal::TuiEvent;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Frame, Terminal, backend::CrosstermBackend};
use std::{
    io,
    sync::mpsc,
    time::{Duration, Instant},
};
use tokio_util::sync::CancellationToken;

pub mod layout;
pub mod model;
pub mod renderers;
pub mod themes;

use layout::{CaladanLayout, TuiLayout};
use model::Model;
use renderers::{
    console::ConsoleRenderer, header, instructions, scan, scan_caladan, spectrum, spectrum_caladan,
    tuners_caladan,
};
use themes::{Theme, ThemeName, UiVariant, create_theme};

/// TUI-based progress display for multiple candidates using The Elm Architecture
pub struct TuiProgressDisplay {
    receiver: mpsc::Receiver<TuiEvent>,
    command_sender: Option<mpsc::Sender<crate::terminal::ScannerCommand>>,
    model: Model,
    _last_update: Instant,
    shutdown_token: CancellationToken,
    theme: Box<dyn Theme>,
    current_theme: ThemeName,
}

impl TuiProgressDisplay {
    /// Create new TUI progress display with default theme
    pub fn new(receiver: mpsc::Receiver<TuiEvent>, shutdown_token: CancellationToken) -> Self {
        let current_theme = ThemeName::CaladanDark;
        let theme = create_theme(&current_theme);
        Self {
            receiver,
            command_sender: None,
            model: Model::new(),
            _last_update: Instant::now(),
            shutdown_token,
            theme,
            current_theme,
        }
    }

    /// Create new TUI progress display with specified theme and command channel
    pub fn new_with_theme(
        receiver: mpsc::Receiver<TuiEvent>,
        shutdown_token: CancellationToken,
        theme: Box<dyn Theme>,
        current_theme: ThemeName,
    ) -> Self {
        Self {
            receiver,
            command_sender: None,
            model: Model::new(),
            _last_update: Instant::now(),
            shutdown_token,
            theme,
            current_theme,
        }
    }

    /// Set the command sender for interactive control
    pub fn with_command_sender(
        mut self,
        sender: mpsc::Sender<crate::terminal::ScannerCommand>,
    ) -> Self {
        self.command_sender = Some(sender);
        self
    }

    /// Run the TUI display loop
    pub fn run(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Check if we're in an interactive terminal
        if !self.is_terminal_interactive() {
            return self.run_text_fallback();
        }

        self.run_full_tui()
    }

    /// Run the full TUI with ratatui
    fn run_full_tui(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Setup terminal with error handling
        if let Err(_e) = enable_raw_mode() {
            return self.run_simple_tui();
        }

        let mut stdout = io::stdout();

        // Use alternate screen mode for full TUI
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;

        let backend = CrosstermBackend::new(stdout);
        let mut terminal = match Terminal::new(backend) {
            Ok(t) => t,
            Err(_) => {
                let _ = disable_raw_mode(); // Try to restore
                return self.run_simple_tui();
            }
        };

        let result = self.run_app(&mut terminal);

        // Restore terminal
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        Ok(result?)
    }

    #[allow(clippy::cognitive_complexity)]
    fn run_app<B: ratatui::backend::Backend>(
        &mut self,
        terminal: &mut Terminal<B>,
    ) -> io::Result<()> {
        let mut iterations = 0;
        let animation_interval = Duration::from_millis(100); // 10 FPS for slower, smoother animation

        loop {
            // Always redraw for animation
            terminal.draw(|f| self.ui(f))?;
            iterations += 1;

            if self.model.should_quit || self.shutdown_token.is_cancelled() {
                break;
            }

            // Auto-exit after some time if no events to prevent hanging
            if iterations > 2000 && self.model.is_empty() {
                // ~100 seconds with no candidates
                break;
            }

            // Handle events with timeout matching animation interval
            if event::poll(animation_interval)?
                && let Ok(Event::Key(key)) = event::read()
            {
                match key.code {
                    KeyCode::Char('c') if key.modifiers.contains(event::KeyModifiers::CONTROL) => {
                        self.model.quit();
                    }
                    KeyCode::Char('d') if key.modifiers.contains(event::KeyModifiers::CONTROL) => {
                        self.model.quit();
                    }
                    KeyCode::Char('q') if !self.model.theme_selector_open => {
                        self.model.quit();
                    }
                    _ if self.model.theme_selector_open => match key.code {
                        KeyCode::Up => {
                            let all_themes = themes::ThemeName::all();
                            self.model.theme_selector_prev(all_themes.len());
                            self.current_theme =
                                all_themes[self.model.theme_selector_index].clone();
                            self.theme = create_theme(&self.current_theme);
                        }
                        KeyCode::Down => {
                            let all_themes = themes::ThemeName::all();
                            self.model.theme_selector_next(all_themes.len());
                            self.current_theme =
                                all_themes[self.model.theme_selector_index].clone();
                            self.theme = create_theme(&self.current_theme);
                        }
                        KeyCode::Enter => {
                            self.model.close_theme_selector();
                        }
                        KeyCode::Esc | KeyCode::Char('q') => {
                            self.model.close_theme_selector();
                        }
                        _ => {}
                    },
                    KeyCode::Char('T') if !self.model.theme_selector_open => {
                        let all_themes = themes::ThemeName::all();
                        let current_idx = all_themes
                            .iter()
                            .position(|t| t == &self.current_theme)
                            .unwrap_or(0);
                        self.model.theme_selector_index = current_idx;
                        self.model.toggle_theme_selector();
                    }
                    KeyCode::Up if !self.model.theme_selector_open => {
                        // Just navigate without pausing scan or stopping playback
                        self.model.navigate_up();
                    }
                    KeyCode::Down if !self.model.theme_selector_open => {
                        // Just navigate without stopping playback
                        self.model.navigate_down();
                    }
                    KeyCode::Left if !self.model.theme_selector_open => {
                        // Just navigate without stopping playback
                        self.model.navigate_left();
                    }
                    KeyCode::Right if !self.model.theme_selector_open => {
                        // Just navigate without stopping playback
                        // Hardcode tuner count to 2 for now (will be dynamic later)
                        self.model.navigate_right(2);
                    }
                    KeyCode::Enter
                        if !self.model.theme_selector_open
                            && matches!(
                                self.model.focus_state,
                                crate::terminal::tui::model::FocusState::Scan
                            )
                            && self.model.selection_mode
                            && !self.model.browsing_mode
                            && self.model.selected_candidate_index.is_some() =>
                    {
                        // Enter browsing mode: pause scan, then tune when Paused event arrives
                        self.model.browsing_mode = true;
                        self.model.pending_tune = true;

                        // Send Pause - we'll tune when we receive Paused event
                        if let Some(sender) = &self.command_sender {
                            let _ = sender.send(crate::terminal::ScannerCommand::Pause);
                        }
                    }
                    KeyCode::Enter
                        if !self.model.theme_selector_open
                            && self.model.browsing_mode
                            && self.model.is_continue_scan_selected() =>
                    {
                        // Exit browsing mode and resume scan
                        self.model.exit_browsing_mode();
                        if let Some(sender) = &self.command_sender {
                            let _ = sender.send(crate::terminal::ScannerCommand::ResumeScan);
                        }

                        // Stop listening when exiting browsing mode
                        if self.model.playback_active {
                            if let Some(sender) = &self.command_sender {
                                let _ = sender.send(crate::terminal::ScannerCommand::StopListening);
                            }
                            self.model.playback_active = false;
                        }
                    }
                    _ => {}
                }
            }

            // Process TUI events (progress and discovery)
            while let Ok(event) = self.receiver.try_recv() {
                let is_paused_event = matches!(event, crate::terminal::TuiEvent::Paused);
                self.model.update_tui_event(event);
                iterations = 0; // Reset iteration counter when we get events

                // If we received Paused event and have a pending tune, send it now
                if is_paused_event && self.model.pending_tune {
                    self.model.pending_tune = false;
                    if let Some((
                        window_id,
                        center_freq,
                        candidate_freq,
                        signal_strength,
                        audio_quality,
                    )) = self.model.selected_candidate_info()
                        && let Some(sender) = &self.command_sender
                    {
                        let _ = sender.send(crate::terminal::ScannerCommand::TuneToCandidate {
                            window_id,
                            center_frequency: center_freq,
                            candidate_frequency: candidate_freq,
                            signal_strength,
                            audio_quality,
                        });
                        self.model.playback_active = true;
                    }
                }
            }
        }

        Ok(())
    }

    fn ui(&self, f: &mut Frame) {
        let theme = self.theme.as_ref();
        let theme_name = self.current_theme.to_string();
        let ui_variant = theme.ui_variant();

        match ui_variant {
            UiVariant::Caladan => {
                let layout = CaladanLayout::new(f.area());

                header::render_header(f, layout.header, &self.model, theme);
                spectrum_caladan::render_spectrum(f, layout.spectrum, &self.model, theme);
                scan_caladan::render_scan(f, layout.progress, &self.model, theme);
                tuners_caladan::render_tuners(f, layout.tuners, &self.model, theme);

                let all_themes: Vec<String> = themes::ThemeName::all()
                    .iter()
                    .map(|t| t.display_name().to_string())
                    .collect();
                instructions::render_instructions(
                    f,
                    layout.instructions,
                    theme,
                    &theme_name,
                    &self.model,
                    &all_themes,
                );
            }
            UiVariant::Standard => {
                let layout = TuiLayout::new(f.area());

                header::render_header(f, layout.header, &self.model, theme);
                spectrum::render_spectrum(f, layout.spectrum, &self.model, theme);
                scan::render_scan(f, layout.progress, &self.model, theme);

                let all_themes: Vec<String> = themes::ThemeName::all()
                    .iter()
                    .map(|t| t.display_name().to_string())
                    .collect();
                instructions::render_instructions(
                    f,
                    layout.instructions,
                    theme,
                    &theme_name,
                    &self.model,
                    &all_themes,
                );
            }
        }
    }

    /// Check if we're running in an interactive terminal
    fn is_terminal_interactive(&self) -> bool {
        use std::io::IsTerminal;

        // Check if stdout is a TTY
        if !io::stdout().is_terminal() {
            return false;
        }

        // Check for CI environment variables
        if std::env::var("CI").is_ok()
            || std::env::var("GITHUB_ACTIONS").is_ok()
            || std::env::var("GITLAB_CI").is_ok()
            || std::env::var("JENKINS_URL").is_ok()
        {
            return false;
        }

        // Check if TERM is set to something reasonable
        match std::env::var("TERM") {
            Ok(term) => !term.is_empty() && term != "dumb",
            Err(_) => false,
        }
    }

    /// Fallback text-based progress display for non-interactive environments
    fn run_text_fallback(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        ConsoleRenderer::tty_println("┌─ Scanning FM stations ... ───────────────┐");
        ConsoleRenderer::tty_println("│ Running in text mode                     │");
        ConsoleRenderer::tty_println("│ Press CTRL-C to exit                     │");
        ConsoleRenderer::tty_println("└──────────────────────────────────────────┘");

        let mut last_update = Instant::now();
        let update_interval = Duration::from_millis(1000); // Update every second

        loop {
            // Check for shutdown signal
            if self.shutdown_token.is_cancelled() {
                break;
            }

            // Process progress events
            while let Ok(event) = self.receiver.try_recv() {
                self.model.update_tui_event(event);
            }

            // Update display periodically
            if last_update.elapsed() >= update_interval {
                ConsoleRenderer::print_text_progress(&self.model);
                last_update = Instant::now();
            }

            // Small sleep to prevent busy waiting
            std::thread::sleep(Duration::from_millis(50));

            // Auto-exit after reasonable time if no activity, or if all candidates are done
            if (self.model.is_empty() && last_update.elapsed() > Duration::from_secs(30))
                || (!self.model.is_empty()
                    && self.model.all_complete()
                    && last_update.elapsed() > Duration::from_secs(3))
            {
                break;
            }
        }

        ConsoleRenderer::tty_println("Scanning complete.");
        Ok(())
    }

    /// Simple TUI-like display that works without raw mode or alternate screen
    fn run_simple_tui(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Clear screen and move to top
        ConsoleRenderer::tty_print("\x1B[2J\x1B[H");

        ConsoleRenderer::tty_println("┌─ Scanning FM stations ... ───────────────┐");
        ConsoleRenderer::tty_println("│ TUI Mode (Simplified)                    │");
        ConsoleRenderer::tty_println("│ Press CTRL-C to exit                     │");
        ConsoleRenderer::tty_println("└──────────────────────────────────────────┘");
        ConsoleRenderer::tty_println("");

        let mut last_update = Instant::now();
        let update_interval = Duration::from_millis(500); // Update every 500ms
        let mut last_candidate_count = 0;

        loop {
            // Check for shutdown signal
            if self.shutdown_token.is_cancelled() {
                break;
            }

            // Process progress events
            while let Ok(event) = self.receiver.try_recv() {
                self.model.update_tui_event(event);
            }

            // Update display periodically or when candidates change
            let current_candidate_count = self.model.candidate_count();
            if last_update.elapsed() >= update_interval
                || current_candidate_count != last_candidate_count
            {
                // Move cursor up to overwrite previous output
                if last_candidate_count > 0 {
                    let lines_to_clear = ConsoleRenderer::calculate_display_lines(&self.model);
                    ConsoleRenderer::tty_print(&format!("\x1B[{}A", lines_to_clear)); // Move cursor up
                }

                ConsoleRenderer::print_tui_style_progress(&self.model);
                last_update = Instant::now();
                last_candidate_count = current_candidate_count;
            }

            // Small sleep to prevent busy waiting
            std::thread::sleep(Duration::from_millis(50));

            // Auto-exit conditions
            if (self.model.is_empty() && last_update.elapsed() > Duration::from_secs(30))
                || (!self.model.is_empty()
                    && self.model.all_complete()
                    && last_update.elapsed() > Duration::from_secs(3))
            {
                break;
            }
        }

        ConsoleRenderer::tty_println("\nScanning complete.");
        Ok(())
    }
}
