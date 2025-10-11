use super::MainThread;
use super::commands::{CommandHandler, CommandHandlerConfig};
use crate::audio::session::AudioSession;
use crate::core::types::Result;
use crate::hardware::pool::TunerProvider;
use crate::scanning::window::{Window, WindowConfig};
use std::sync::Arc;
use tracing::debug;

impl MainThread {
    pub(super) fn process_commands(
        &mut self,
        window_num: usize,
        _total_windows: usize,
        audio_session: &mut Option<AudioSession>,
    ) -> Result<()> {
        let mut commands = Vec::new();
        if let Some(receiver) = &self.command_receiver {
            while let Ok(command) = receiver.try_recv() {
                commands.push(command);
            }
        }

        for command in commands {
            let mut handler = CommandHandler::new(CommandHandlerConfig {
                scanner_state: &mut self.scanner_state,
                pause_signal: &self.pause_signal,
                pool: &self.pool,
                config: &self.config,
                shutdown_coordinator: &self.shutdown_coordinator,
                progress_reporter: &self.progress_reporter,
                tui_event_sender: &self.tui_event_sender,
                current_playing: &mut self.current_playing,
            });
            handler.handle_command(command, window_num, audio_session)?;
        }
        Ok(())
    }

    pub(super) fn check_and_handle_command(
        &mut self,
        window_num: usize,
        audio_session: &mut Option<AudioSession>,
    ) -> Result<()> {
        if let Some(receiver) = &self.command_receiver
            && let Ok(command) = receiver.try_recv()
        {
            let mut handler = CommandHandler::new(CommandHandlerConfig {
                scanner_state: &mut self.scanner_state,
                pause_signal: &self.pause_signal,
                pool: &self.pool,
                config: &self.config,
                shutdown_coordinator: &self.shutdown_coordinator,
                progress_reporter: &self.progress_reporter,
                tui_event_sender: &self.tui_event_sender,
                current_playing: &mut self.current_playing,
            });
            handler.handle_command(command, window_num, audio_session)?;
        }
        Ok(())
    }

    pub(super) fn process_commands_with_pause_check(
        &mut self,
        window_num: usize,
        total_windows: usize,
        audio_session: &mut Option<AudioSession>,
    ) -> Result<bool> {
        self.process_commands(window_num, total_windows, audio_session)?;

        if self.scanner_state.is_paused() {
            return Ok(true);
        }

        self.check_and_handle_command(window_num, audio_session)?;
        Ok(self.scanner_state.is_paused())
    }

    pub(super) fn process_window(
        &self,
        window_num: usize,
        center_freq: f64,
        total_windows: usize,
    ) -> Result<()> {
        debug!(
            window = window_num,
            total = total_windows,
            "Processing window"
        );

        let window = Window::new(WindowConfig {
            center_freq,
            window_num,
            total_windows,
            tuner_provider: Arc::clone(&self.pool) as Arc<dyn TunerProvider>,
            config: self.config.clone(),
            progress_reporter: self.progress_reporter.clone(),
            shutdown_coordinator: self.shutdown_coordinator.clone(),
            pause_signal: Some(self.pause_signal.clone()),
        });

        window.process_with_pool()?;

        debug!(
            completed_window = window_num,
            next_window = window_num + 1,
            "Window complete"
        );
        Ok(())
    }
}
