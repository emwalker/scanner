//! TUI module using The Elm Architecture pattern

use crate::ecs::{AudioEntity, CandidateEntity, Entities, Entity, ScanEntity, StationEntity};
use crate::ui::TuiEvent;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use indexmap::IndexMap;
use ratatui::{Frame, Terminal, backend::CrosstermBackend};
use std::{
    collections::HashMap,
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
    model: Model,
    _last_update: Instant,
    shutdown_token: CancellationToken,
    theme: Box<dyn Theme>,
    current_theme: ThemeName,
    ui_update_system: Option<crate::ecs::systems::UIUpdateSystem>,
    scan_entities: Option<Entities<ScanEntity>>,
    station_entities: Option<Entities<StationEntity>>,
    audio_entities: Option<Entities<AudioEntity>>,
    candidate_entities: Option<Entities<CandidateEntity>>,
    last_station_gen: u64,
    last_audio_gen: u64,
    last_candidate_gen: u64,
}

impl TuiProgressDisplay {
    /// Create new TUI progress display with default theme
    pub fn new(receiver: mpsc::Receiver<TuiEvent>, shutdown_token: CancellationToken) -> Self {
        let current_theme = ThemeName::CaladanDark;
        let theme = create_theme(&current_theme);
        Self {
            receiver,
            model: Model::new(),
            _last_update: Instant::now(),
            shutdown_token,
            theme,
            current_theme,
            ui_update_system: None,
            scan_entities: None,
            station_entities: None,
            audio_entities: None,
            candidate_entities: None,
            last_station_gen: 0,
            last_audio_gen: 0,
            last_candidate_gen: 0,
        }
    }

    /// Create new TUI progress display with specified theme
    pub fn new_with_theme(
        receiver: mpsc::Receiver<TuiEvent>,
        shutdown_token: CancellationToken,
        theme: Box<dyn Theme>,
        current_theme: ThemeName,
    ) -> Self {
        Self {
            receiver,
            model: Model::new(),
            _last_update: Instant::now(),
            shutdown_token,
            theme,
            current_theme,
            ui_update_system: None,
            scan_entities: None,
            station_entities: None,
            audio_entities: None,
            candidate_entities: None,
            last_station_gen: 0,
            last_audio_gen: 0,
            last_candidate_gen: 0,
        }
    }

    /// Set entity worlds for spectrum display integration
    pub fn with_entities(
        mut self,
        scan_entities: Entities<ScanEntity>,
        station_entities: Entities<StationEntity>,
        audio_entities: Entities<AudioEntity>,
        candidate_entities: Entities<CandidateEntity>,
    ) -> Self {
        self.scan_entities = Some(scan_entities);
        self.station_entities = Some(station_entities);
        self.audio_entities = Some(audio_entities);
        self.candidate_entities = Some(candidate_entities);
        self.ui_update_system = Some(crate::ecs::systems::UIUpdateSystem::new());
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

    /// Check if awaiting tune should transition to listening
    fn check_awaiting_tune_transition(&mut self) {
        if !matches!(self.model.ui_mode, model::UiMode::AwaitingTune { .. }) {
            return;
        }

        if let Some(ref scan_entities) = self.scan_entities
            && let Ok(entities) = scan_entities.read()
            && let Some(scan) = entities.iter().next()
            && scan.is_paused()
            && let model::UiMode::AwaitingTune {
                navigation_index,
                tuning_index,
            } = self.model.ui_mode
            && let Some(info) = self.model.selected_candidate_info()
        {
            self.model.ui_mode = model::UiMode::Listening {
                navigation_index,
                playing_index: tuning_index,
                playing_candidate_id: info.candidate_id.clone(),
            };
            debug!(
                candidate_id = ?info.candidate_id,
                "TUI: Transitioned from AwaitingTune to Listening"
            );
        }
    }

    /// Update spectrum display data from entity worlds
    fn update_spectrum_from_entities(&mut self) {
        if let Some(system) = &mut self.ui_update_system
            && let (Some(station_entities), Some(audio_entities), Some(candidate_entities)) = (
                &self.station_entities,
                &self.audio_entities,
                &self.candidate_entities,
            )
        {
            let station_gen = station_entities.read().unwrap().generation();
            let audio_gen = audio_entities.read().unwrap().generation();
            let candidate_gen = candidate_entities.read().unwrap().generation();

            if station_gen != self.last_station_gen
                || audio_gen != self.last_audio_gen
                || candidate_gen != self.last_candidate_gen
            {
                use crate::ecs::{System, SystemContext};

                debug!(
                    station_gen = station_gen,
                    audio_gen = audio_gen,
                    candidate_gen = candidate_gen,
                    last_station_gen = self.last_station_gen,
                    last_audio_gen = self.last_audio_gen,
                    last_candidate_gen = self.last_candidate_gen,
                    "TUI: Running UIUpdateSystem due to generation change"
                );

                let mut context = SystemContext::new()
                    .with_station_entities(std::sync::Arc::clone(station_entities))
                    .with_audio_entities(std::sync::Arc::clone(audio_entities))
                    .with_candidate_entities(std::sync::Arc::clone(candidate_entities));

                if system.run(&mut context).is_ok() {
                    let spectrum_count = system.stations().len();
                    let active_count = system.stations().iter().filter(|s| s.is_active).count();

                    self.model.spectrum_stations = system.stations().to_vec();
                    self.model.active_audio_frequency = system.active_frequency();

                    let candidates_by_window = system.candidates_by_window().clone();
                    self.sync_candidates_to_model(&candidates_by_window);

                    debug!(
                        spectrum_count = spectrum_count,
                        active_count = active_count,
                        candidate_count = candidates_by_window
                            .values()
                            .map(|v| v.len())
                            .sum::<usize>(),
                        "TUI: Updated model from entities"
                    );
                }

                self.last_station_gen = station_gen;
                self.last_audio_gen = audio_gen;
                self.last_candidate_gen = candidate_gen;
            }
        }
    }

    /// Sync candidate data from UIUpdateSystem to Model
    fn sync_candidates_to_model(
        &mut self,
        candidates_by_window: &IndexMap<usize, Vec<crate::ecs::systems::ui::CandidateData>>,
    ) {
        use crate::ui::tui::model::types::{CandidateProgress, CandidateStatus, WindowProgress};
        use std::time::Instant;

        for (window_id, candidate_data_list) in candidates_by_window {
            let window = self
                .model
                .windows
                .entry(*window_id)
                .or_insert_with(|| WindowProgress {
                    window_id: *window_id,
                    candidates: Vec::new(),
                    is_complete: false,
                    candidate_lookup: HashMap::new(),
                });

            for candidate_data in candidate_data_list {
                let status = match candidate_data.state {
                    crate::ecs::CandidateState::Detected => CandidateStatus::Detected,
                    crate::ecs::CandidateState::Analyzing => CandidateStatus::Analyzing,
                    crate::ecs::CandidateState::Signal => CandidateStatus::Signal,
                    crate::ecs::CandidateState::Playing => CandidateStatus::Playing,
                    crate::ecs::CandidateState::Rejected => CandidateStatus::Rejected,
                    crate::ecs::CandidateState::Completed => CandidateStatus::Completed,
                };

                let candidate_progress = CandidateProgress {
                    candidate_id: candidate_data.candidate_id.clone(),
                    frequency_hz: candidate_data.frequency_hz,
                    metadata: crate::scanning::window::WindowMetadata {
                        window_id: *window_id,
                        center_frequency_hz: candidate_data.frequency_hz,
                    },
                    completion: candidate_data.completion,
                    status,
                    audio_quality: candidate_data.audio_quality,
                    signal_strength: candidate_data.signal_strength,
                    last_update: Instant::now(),
                };

                if let Some(&index) = window.candidate_lookup.get(&candidate_data.candidate_id) {
                    window.candidates[index] = candidate_progress;
                } else {
                    let index = window.candidates.len();
                    window
                        .candidate_lookup
                        .insert(candidate_data.candidate_id.clone(), index);
                    window.candidates.push(candidate_progress);
                }
            }
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

        // ECS Phase 5: Pure ECS - set pause_request with station info
        if let Some(ref scan_entities) = self.scan_entities
            && let Ok(mut entities) = scan_entities.write()
            && let Some(scan) = entities.iter_mut().next()
        {
            let window_num = self.model.current_window;

            if let Some(info) = self.model.selected_candidate_info() {
                scan.request_pause_with_station(
                    window_num,
                    info.candidate_frequency,
                    info.metadata.center_frequency_hz,
                );
                debug!(
                    scan_id = ?scan.id(),
                    window_num = window_num,
                    station_frequency_mhz = info.candidate_frequency / 1e6,
                    "TUI: Set pause_request with station on ScanEntity"
                );
            } else {
                scan.request_pause(window_num);
                debug!(
                    scan_id = ?scan.id(),
                    window_num = window_num,
                    "TUI: Set pause_request on ScanEntity (no station selected)"
                );
            }
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
            "TUI: Switching to different station"
        );

        self.model.ui_mode = model::UiMode::AwaitingTune {
            navigation_index: selected_index,
            tuning_index: selected_index,
        };

        // ECS Phase 5: RequestProcessorSystem will handle the tune_request
        // Just set pause_request with the new station info
        if let Some(ref scan_entities) = self.scan_entities
            && let Ok(mut entities) = scan_entities.write()
            && let Some(scan) = entities.iter_mut().next()
        {
            let window_num = self.model.current_window;
            scan.request_pause_with_station(
                window_num,
                info.candidate_frequency,
                info.metadata.center_frequency_hz,
            );
            debug!(
                scan_id = ?scan.id(),
                window_num = window_num,
                station_frequency_mhz = info.candidate_frequency / 1e6,
                "TUI: Set pause_request with new station on ScanEntity"
            );
        }

        self.model.playback_active = true;
        true
    }

    fn handle_resume_scan(&mut self) -> bool {
        self.model.exit_browsing_mode();
        self.model.ui_mode = model::UiMode::Idle;

        // ECS Phase 5: Pure ECS - only set component, no commands
        if let Some(ref scan_entities) = self.scan_entities
            && let Ok(mut entities) = scan_entities.write()
            && let Some(scan) = entities.iter_mut().next()
        {
            let window_num = self.model.current_window;
            scan.request_resume(window_num);
            debug!(
                scan_id = ?scan.id(),
                window_num = window_num,
                "TUI: Set resume_request on ScanEntity"
            );
        }

        // ECS Phase 5: Pure ECS - only set component, no commands
        if self.model.playback_active {
            if let Some(ref audio_entities) = self.audio_entities
                && let Ok(mut entities) = audio_entities.write()
                && let Some(audio) = entities.iter_mut().next()
            {
                audio.request_stop_listening();
                debug!(
                    audio_id = ?audio.id(),
                    "TUI: Set stop_listening_request on AudioEntity"
                );
            }

            self.model.playback_active = false;
        }

        true
    }

    /// Handle tuning/playback actions (Enter key in various contexts)
    fn handle_tuning_actions(&mut self, key: &event::KeyEvent) -> bool {
        tracing::error!(key_code = ?key.code, theme_selector_open = self.model.theme_selector_open, "TUI: handle_tuning_actions called");

        if key.code != KeyCode::Enter || self.model.theme_selector_open {
            tracing::error!("TUI: Not ENTER or theme selector open, returning false");
            return false;
        }

        tracing::error!(
            focus_state = ?self.model.focus_state,
            browsing_mode = self.model.browsing_mode(),
            selected_index = ?self.model.selected_candidate_index(),
            ui_mode = ?self.model.ui_mode,
            "TUI: ENTER key pressed - CONFIRMED"
        );

        // Case 1: Enter browsing mode from scan mode
        // Allow entering browsing mode when:
        // - Focus is on scan results
        // - Not already in browsing mode
        // - Have a candidate selected
        // - Scan is paused/idle (including after scan completes)
        if matches!(self.model.focus_state, model::FocusState::Scan)
            && !self.model.browsing_mode()
            && let Some(selected_index) = self.model.selected_candidate_index()
        {
            tracing::error!(
                selected_index = selected_index,
                "TUI: Calling handle_enter_browsing_mode"
            );
            return self.handle_enter_browsing_mode(selected_index);
        } else {
            tracing::error!(
                focus_state = ?self.model.focus_state,
                browsing_mode = self.model.browsing_mode(),
                selected_index = ?self.model.selected_candidate_index(),
                "TUI: Case 1 condition FAILED - not entering browsing mode"
            );
        }

        // Case 2: Switch station while listening
        if matches!(self.model.ui_mode, model::UiMode::Listening { .. })
            && !self.model.is_continue_scan_selected()
            && let Some(selected_index) = self.model.selected_candidate_index()
            && let Some(info) = self.model.selected_candidate_info()
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
            self.model.update_tui_event(event);
            *iterations = 0; // Reset iteration counter when we get events
        }
        Ok(())
    }

    /// Handle keyboard input with timeout, returns true if redraw needed
    fn handle_keyboard_input(&mut self, animation_interval: Duration) -> io::Result<bool> {
        if event::poll(animation_interval)? {
            match event::read() {
                Ok(Event::Key(key)) => {
                    // Log ALL keys including char codes with ERROR level so they show up
                    match key.code {
                        KeyCode::Char(c) => {
                            tracing::error!(key_code = ?key.code, char_value = ?c, char_as_u8 = c as u8, modifiers = ?key.modifiers, kind = ?key.kind, "TUI: Char key event");
                        }
                        _ => {
                            tracing::error!(key_code = ?key.code, modifiers = ?key.modifiers, kind = ?key.kind, "TUI: Key event received");
                        }
                    }

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
                    tracing::error!("TUI: About to call handle_tuning_actions");
                    let needs_redraw = self.handle_tuning_actions(&key);
                    tracing::error!(
                        needs_redraw = needs_redraw,
                        "TUI: handle_tuning_actions returned"
                    );
                    return Ok(needs_redraw);
                }
                Ok(other_event) => {
                    debug!(event = ?other_event, "TUI: Non-key event received");
                }
                Err(e) => {
                    debug!(error = ?e, "TUI: Error reading event");
                }
            }
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
            // Update spectrum data from entities
            self.update_spectrum_from_entities();

            // Always mark dirty for smooth spectrum animation at 10 FPS
            self.model.mark_dirty();

            // Redraw for animation
            terminal.draw(|f| self.ui(f))?;
            self.model.clear_dirty();
            iterations += 1;

            if self.should_quit() {
                break;
            }

            if self.should_auto_exit(iterations) {
                break;
            }

            // Handle keyboard input
            self.handle_keyboard_input(animation_interval)?;

            // Process TUI events
            self.process_tui_events(&mut iterations)?;

            // Check if we need to transition from AwaitingTune to Listening
            self.check_awaiting_tune_transition();
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
