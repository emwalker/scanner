//! Shared state machine context for scanning tasks

use crate::audio::session::AudioSession;
use crate::core::types::{Result, ScanningConfig};
use crate::hardware::DeviceId;
use crate::hardware::pool::{Pool, TunerId};
use crate::main_thread::audio_coordinator::{AudioCoordinator, TuneParams};
use crate::scanner_state::{PauseSignal, ScannerState};
use crate::scanning::window::{Window, WindowConfig};
use crate::shutdown::ShutdownCoordinator;
use crate::ui::{ProgressReporter, ScannerCommand, TuiEvent};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;
use tracing::debug;

pub enum LoopControl {
    Continue,
    Break,
    Advance,
    ResubmitAfter(Duration),
}

pub struct ScanContext<'a> {
    pub config: &'a ScanningConfig,
    pub pool: &'a Arc<Pool>,
    pub shutdown_coordinator: &'a Arc<ShutdownCoordinator>,
    pub progress_reporter: &'a Arc<dyn ProgressReporter>,
    pub pause_signal: &'a PauseSignal,
    pub command_receiver: &'a mut Option<Receiver<ScannerCommand>>,
    pub tui_event_sender: &'a Option<Sender<TuiEvent>>,

    pub scanner_state: ScannerState,
    pub current_playing: Option<TuneParams>,
    pub audio_session: Option<AudioSession>,
    pub window_centers: Vec<f64>,
    pub windows_to_process: usize,
    pub window_index: usize,
}

impl<'a> ScanContext<'a> {
    pub fn handle_shutting_down_mode(&self) -> Result<LoopControl> {
        debug!("Shutdown requested, stopping scanning");
        Ok(LoopControl::Break)
    }

    pub fn handle_scan_complete_mode(&mut self) -> Result<LoopControl> {
        self.process_commands(self.windows_to_process, self.window_centers.len())?;
        std::thread::sleep(std::time::Duration::from_millis(100));
        Ok(LoopControl::Continue)
    }

    pub fn handle_scan_complete_paused_mode(&mut self) -> Result<LoopControl> {
        self.process_commands(self.windows_to_process, self.window_centers.len())?;
        std::thread::sleep(std::time::Duration::from_millis(100));
        Ok(LoopControl::Continue)
    }

    pub fn handle_paused_mode(&mut self) -> Result<LoopControl> {
        if !self.window_index.is_multiple_of(50) {
            debug!(
                iteration = self.window_index,
                total = self.window_centers.len(),
                "Paused - waiting for commands"
            );
        }
        self.process_commands(self.window_index + 1, self.window_centers.len())?;
        std::thread::sleep(std::time::Duration::from_millis(100));
        Ok(LoopControl::Continue)
    }

    pub fn handle_listening_mode(&mut self) -> Result<LoopControl> {
        self.process_commands(self.window_index + 1, self.window_centers.len())?;
        std::thread::sleep(std::time::Duration::from_millis(100));
        Ok(LoopControl::Continue)
    }

    pub fn handle_scanning_mode(&mut self) -> Result<LoopControl> {
        if self.window_index >= self.windows_to_process {
            debug!("Scan complete - all windows processed");
            self.scanner_state
                .mark_scan_complete(self.windows_to_process);
            return Ok(LoopControl::Continue);
        }

        debug!(
            iteration = self.window_index,
            total = self.windows_to_process,
            "Start of scan loop iteration"
        );

        if self
            .process_commands_with_pause_check(self.window_index + 1, self.window_centers.len())?
        {
            debug!("Pause detected during scanning, transitioning to Paused mode");
            return Ok(LoopControl::Continue);
        }

        let center_freq = self.window_centers[self.window_index];
        match self.process_window(
            self.window_index + 1,
            center_freq,
            self.window_centers.len(),
        ) {
            Ok(()) => {
                self.process_commands(self.window_index + 1, self.window_centers.len())?;

                debug!(
                    completed_window = self.window_index + 1,
                    next_window = self.window_index + 2,
                    remaining = self.windows_to_process - self.window_index - 1,
                    "Window complete, advancing to next"
                );

                Ok(LoopControl::Advance)
            }
            Err(crate::core::types::ScannerError::NoAvailableTuner(_)) => {
                debug!("No tuners available, yielding semaphore to allow device discovery");
                Ok(LoopControl::ResubmitAfter(Duration::from_millis(100)))
            }
            Err(e) => Err(e),
        }
    }

    fn process_window(
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
            tuner_provider: self.pool.clone(),
            config: Arc::new(self.config.clone()),
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

    fn process_commands(&mut self, window_num: usize, _total_windows: usize) -> Result<()> {
        let mut commands = Vec::new();
        if let Some(receiver) = self.command_receiver {
            while let Ok(command) = receiver.try_recv() {
                commands.push(command);
            }
        }

        for command in commands {
            self.handle_command(command, window_num)?;
        }
        Ok(())
    }

    fn check_and_handle_command(&mut self, window_num: usize) -> Result<()> {
        if let Some(receiver) = self.command_receiver
            && let Ok(command) = receiver.try_recv()
        {
            self.handle_command(command, window_num)?;
        }
        Ok(())
    }

    fn process_commands_with_pause_check(
        &mut self,
        window_num: usize,
        total_windows: usize,
    ) -> Result<bool> {
        self.process_commands(window_num, total_windows)?;

        if self.scanner_state.is_paused() {
            return Ok(true);
        }

        self.check_and_handle_command(window_num)?;
        Ok(self.scanner_state.is_paused())
    }

    fn handle_command(&mut self, command: ScannerCommand, window_num: usize) -> Result<()> {
        match command {
            ScannerCommand::Pause => {
                self.handle_pause(window_num)?;
            }
            ScannerCommand::ResumeScan => {
                self.handle_resume(window_num);
            }
            ScannerCommand::TuneToCandidate {
                candidate_id,
                window_id,
                center_frequency,
                candidate_frequency,
                signal_strength,
                audio_quality,
            } => {
                let params = TuneParams {
                    candidate_id,
                    window_id,
                    center_frequency,
                    candidate_frequency,
                    signal_strength,
                    audio_quality,
                };
                self.handle_tune_to_candidate(window_num, params)?;
            }
            ScannerCommand::StopListening => {
                let coordinator = AudioCoordinator::new(
                    self.pool,
                    self.config,
                    self.shutdown_coordinator,
                    self.progress_reporter,
                );
                let playing_params = self.current_playing.take();
                coordinator.stop_listening(&mut self.audio_session, playing_params);
                self.scanner_state.handle_stop_listening();
            }
        }
        Ok(())
    }

    fn handle_pause(&mut self, window_num: usize) -> Result<()> {
        debug!(window = window_num, "Scanner paused, creating AudioSession");
        self.pause_signal.pause();
        self.scanner_state.handle_pause(window_num);

        self.audio_session = Some(AudioSession::new(
            self.config,
            self.shutdown_coordinator.clone(),
        )?);
        debug!("AudioSession created for browse mode");

        if let Some(sender) = self.tui_event_sender {
            let status = self.pool.status();
            let tuner_id = status
                .tuners
                .first()
                .map(|t| t.id.clone())
                .unwrap_or_else(|| TunerId::new(DeviceId::from_serial("unknown", "0"), 0));
            debug!(tuner_id = ?tuner_id, "Sending Paused event to TUI");
            let _ = sender.send(TuiEvent::Paused { tuner_id });
        } else {
            debug!("No TUI event sender available, cannot send Paused event");
        }

        Ok(())
    }

    fn handle_resume(&mut self, window_num: usize) {
        debug!(
            window = window_num,
            "Scanner resuming - exiting selection mode and continuing scan"
        );
        self.pause_signal.unpause();
        let _next_window = self.scanner_state.handle_resume();

        self.audio_session = None;
        debug!("AudioSession dropped, returning to scan mode");
    }

    fn handle_tune_to_candidate(&mut self, window_num: usize, params: TuneParams) -> Result<()> {
        debug!(
            candidate_id = ?params.candidate_id,
            window_id = params.window_id,
            candidate_frequency_mhz = params.candidate_frequency / 1e6,
            "ScanContext: Received TuneToCandidate command"
        );
        self.scanner_state.handle_tune(window_num);

        if let Some(session) = &mut self.audio_session {
            let coordinator = AudioCoordinator::new(
                self.pool,
                self.config,
                self.shutdown_coordinator,
                self.progress_reporter,
            );
            coordinator.tune_to_station(session, params.clone())?;
            self.current_playing = Some(params);
        } else {
            debug!("TuneToCandidate received but no AudioSession exists");
        }

        Ok(())
    }
}
