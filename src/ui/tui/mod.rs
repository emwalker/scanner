//! TUI module using The Elm Architecture pattern

use crate::ui::TuiEvent;
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
use tracing::debug;

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
    command_sender: Option<mpsc::Sender<crate::ui::ScannerCommand>>,
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
    pub fn with_command_sender(mut self, sender: mpsc::Sender<crate::ui::ScannerCommand>) -> Self {
        self.command_sender = Some(sender);
        self
    }

    /// Pre-populate the model with cached devices
    pub fn with_cached_devices(mut self, devices: Vec<crate::hardware::DeviceInfo>) -> Self {
        self.model = self.model.with_cached_devices(devices);
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

    /// Check if the application should quit
    fn should_quit(&self) -> bool {
        self.model.should_quit || self.shutdown_token.is_cancelled()
    }

    /// Check if the application should auto-exit due to inactivity
    fn should_auto_exit(&self, iterations: u32) -> bool {
        iterations > 2000 && self.model.is_empty()
    }

    /// Handle quit keys (Ctrl-C, Ctrl-D, 'q')
    fn handle_quit_keys(&mut self, key: &event::KeyEvent) -> bool {
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(event::KeyModifiers::CONTROL) => {
                self.model.quit();
                true
            }
            KeyCode::Char('d') if key.modifiers.contains(event::KeyModifiers::CONTROL) => {
                self.model.quit();
                true
            }
            KeyCode::Char('q') if !self.model.theme_selector_open => {
                self.model.quit();
                true
            }
            _ => false,
        }
    }

    /// Handle theme selector navigation
    fn handle_theme_selector(&mut self, key: KeyCode) {
        match key {
            KeyCode::Up => {
                let all_themes = themes::ThemeName::all();
                self.model.theme_selector_prev(all_themes.len());
                self.current_theme = all_themes[self.model.theme_selector_index].clone();
                self.theme = create_theme(&self.current_theme);
            }
            KeyCode::Down => {
                let all_themes = themes::ThemeName::all();
                self.model.theme_selector_next(all_themes.len());
                self.current_theme = all_themes[self.model.theme_selector_index].clone();
                self.theme = create_theme(&self.current_theme);
            }
            KeyCode::Enter | KeyCode::Esc | KeyCode::Char('q') => {
                self.model.close_theme_selector();
            }
            _ => {}
        }
    }

    /// Handle navigation keys (arrows)
    fn handle_navigation_keys(&mut self, key: KeyCode) {
        match key {
            KeyCode::Up => self.model.navigate_up(),
            KeyCode::Down => self.model.navigate_down(),
            KeyCode::Left => self.model.navigate_left(),
            KeyCode::Right => {
                // Hardcode tuner count to 2 for now (will be dynamic later)
                self.model.navigate_right(2);
            }
            _ => {}
        }
    }

    fn handle_enter_browsing_mode(&mut self, selected_index: usize) -> bool {
        self.model.ui_mode = model::UiMode::AwaitingTune {
            navigation_index: selected_index,
            tuning_index: selected_index,
        };

        if let Some(sender) = &self.command_sender {
            let _ = sender.send(crate::ui::ScannerCommand::Pause);
        }

        true
    }

    fn handle_switch_station(
        &mut self,
        selected_index: usize,
        info: model::SelectedCandidateInfo,
    ) -> bool {
        debug!(
            candidate_id = ?info.candidate_id,
            window_id = info.metadata.window_id,
            candidate_frequency_mhz = info.candidate_frequency / 1e6,
            "TUI: Sending TuneToCandidate command"
        );

        self.model.ui_mode = model::UiMode::AwaitingTune {
            navigation_index: selected_index,
            tuning_index: selected_index,
        };

        if let Some(sender) = &self.command_sender {
            let _ = sender.send(crate::ui::ScannerCommand::TuneToCandidate {
                candidate_id: info.candidate_id,
                window_id: info.metadata.window_id,
                center_frequency: info.metadata.center_frequency_hz,
                candidate_frequency: info.candidate_frequency,
                signal_strength: info.signal_strength,
                audio_quality: info.audio_quality,
            });
        }

        self.model.playback_active = true;
        true
    }

    fn handle_resume_scan(&mut self) -> bool {
        self.model.exit_browsing_mode();
        self.model.ui_mode = model::UiMode::Idle;

        if let Some(sender) = &self.command_sender {
            let _ = sender.send(crate::ui::ScannerCommand::ResumeScan);
        }

        if self.model.playback_active {
            if let Some(sender) = &self.command_sender {
                let _ = sender.send(crate::ui::ScannerCommand::StopListening);
            }
            self.model.playback_active = false;
        }

        true
    }

    /// Handle tuning/playback actions (Enter key in various contexts)
    fn handle_tuning_actions(&mut self, key: &event::KeyEvent) -> bool {
        if key.code != KeyCode::Enter || self.model.theme_selector_open {
            return false;
        }

        // Case 1: Enter browsing mode from scan mode
        if matches!(self.model.focus_state, model::FocusState::Scan)
            && self.model.selection_mode()
            && !self.model.browsing_mode()
            && let Some(selected_index) = self.model.selected_candidate_index()
        {
            return self.handle_enter_browsing_mode(selected_index);
        }

        // Case 2: Switch station while listening
        if matches!(self.model.ui_mode, model::UiMode::Listening { .. })
            && !self.model.is_continue_scan_selected()
            && let Some(selected_index) = self.model.selected_candidate_index()
            && let Some(info) = self.model.selected_candidate_info()
            && self.command_sender.is_some()
        {
            return self.handle_switch_station(selected_index, info);
        }

        // Case 3: Resume scan from browsing mode
        if matches!(
            self.model.ui_mode,
            model::UiMode::Listening { .. } | model::UiMode::AwaitingTune { .. }
        ) && self.model.is_continue_scan_selected()
        {
            return self.handle_resume_scan();
        }

        false
    }

    /// Process TUI events (progress updates and discovery)
    fn process_tui_events(&mut self, iterations: &mut u32) -> io::Result<()> {
        while let Ok(event) = self.receiver.try_recv() {
            let is_paused_event = matches!(event, TuiEvent::Paused { .. });
            self.model.update_tui_event(event);
            *iterations = 0; // Reset iteration counter when we get events

            // If we received Paused event and we're awaiting tune, send it now
            if is_paused_event
                && matches!(self.model.ui_mode, model::UiMode::AwaitingTune { .. })
                && let Some(info) = self.model.selected_candidate_info()
                && let Some(sender) = &self.command_sender
            {
                let _ = sender.send(crate::ui::ScannerCommand::TuneToCandidate {
                    candidate_id: info.candidate_id,
                    window_id: info.metadata.window_id,
                    center_frequency: info.metadata.center_frequency_hz,
                    candidate_frequency: info.candidate_frequency,
                    signal_strength: info.signal_strength,
                    audio_quality: info.audio_quality,
                });
                self.model.playback_active = true;
            }
        }
        Ok(())
    }

    /// Handle keyboard input with timeout, returns true if redraw needed
    fn handle_keyboard_input(&mut self, animation_interval: Duration) -> io::Result<bool> {
        if event::poll(animation_interval)?
            && let Ok(Event::Key(key)) = event::read()
        {
            // Check for quit keys first
            if self.handle_quit_keys(&key) {
                return Ok(false);
            }

            // Theme selector takes priority when open
            if self.model.theme_selector_open {
                self.handle_theme_selector(key.code);
                return Ok(false);
            }

            // Toggle theme selector
            if matches!(key.code, KeyCode::Char('T')) {
                let all_themes = themes::ThemeName::all();
                let current_idx = all_themes
                    .iter()
                    .position(|t| t == &self.current_theme)
                    .unwrap_or(0);
                self.model.theme_selector_index = current_idx;
                self.model.toggle_theme_selector();
                return Ok(false);
            }

            // Handle navigation
            if matches!(
                key.code,
                KeyCode::Up | KeyCode::Down | KeyCode::Left | KeyCode::Right
            ) {
                self.handle_navigation_keys(key.code);
                return Ok(false);
            }

            // Handle tuning actions (Enter key)
            let needs_redraw = self.handle_tuning_actions(&key);
            return Ok(needs_redraw);
        }

        Ok(false)
    }

    /// Main TUI event loop
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

            if self.should_quit() {
                break;
            }

            if self.should_auto_exit(iterations) {
                break;
            }

            // Handle keyboard input
            let needs_redraw = self.handle_keyboard_input(animation_interval)?;

            // Redraw immediately if state changed from keyboard input
            if needs_redraw {
                terminal.draw(|f| self.ui(f))?;
            }

            // Process TUI events
            self.process_tui_events(&mut iterations)?;
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
