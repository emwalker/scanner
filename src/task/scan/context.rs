//! Shared state machine context for scanning tasks

use crate::audio::session::AudioSession;
use crate::core::types::{Result, ScanningConfig};
use crate::ecs::{AudioEntity, Entity, ScanEntity, StationEntity};
use crate::hardware::DeviceId;
use crate::hardware::pool::{Pool, TunerId};
use crate::main_thread::audio_coordinator::{AudioCoordinator, TuneParams};
use crate::pause_signal::PauseSignal;
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

    pub scan_entity: ScanEntity,
    pub audio_session: Option<AudioSession>,

    // ECS Phase 4: Station and Audio entities
    pub current_station: Option<StationEntity>,
    pub current_audio: Option<AudioEntity>,

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
            self.scan_entity.progress.mark_complete();
            self.scan_entity.lifecycle.complete();
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

        // Synchronize: Start window in scan entity
        self.scan_entity.progress.start_window(self.window_index);

        match self.process_window(
            self.window_index + 1,
            center_freq,
            self.window_centers.len(),
        ) {
            Ok(()) => {
                self.process_commands(self.window_index + 1, self.window_centers.len())?;

                // Synchronize: Complete window in scan entity
                self.scan_entity.progress.complete_window();

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

        if self.scan_entity.is_paused() {
            return Ok(true);
        }

        self.check_and_handle_command(window_num)?;
        Ok(self.scan_entity.is_paused())
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
                coordinator.stop_listening(
                    &mut self.audio_session,
                    self.current_station.as_ref(),
                    self.current_audio.as_ref(),
                );

                if let Some(mut audio) = self.current_audio.take() {
                    audio.stop();
                }
                self.current_station = None;

                if let crate::ecs::components::scan::ScanPauseState::Listening {
                    paused_at_window,
                } = self.scan_entity.progress.state
                {
                    self.scan_entity.progress.stop_listening(paused_at_window);
                }
            }
        }
        Ok(())
    }

    fn handle_pause(&mut self, window_num: usize) -> Result<()> {
        debug!(window = window_num, "Scanner paused, creating AudioSession");
        self.pause_signal.pause();
        self.scan_entity.progress.pause(window_num);
        self.scan_entity.lifecycle.pause();

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
        self.scan_entity.progress.resume();

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
        self.scan_entity.progress.start_listening(window_num);

        if let Some(session) = &mut self.audio_session {
            let coordinator = AudioCoordinator::new(
                self.pool,
                self.config,
                self.shutdown_coordinator,
                self.progress_reporter,
            );
            coordinator.tune_to_station(session, params.clone())?;

            // Dual-write: Create entities from TuneParams
            let signal = crate::core::types::Signal {
                frequency_hz: params.candidate_frequency,
                signal_strength: params.signal_strength.unwrap_or(0.1) as f32,
                bandwidth_hz: 200_000.0,
                modulation: crate::core::types::ModulationType::WFM,
                audio_sample_rate: self.config.audio.sample_rate,
                detected_at: std::time::SystemTime::now(),
                analysis_duration_ms: 0,
                detection_center_freq: params.center_frequency,
                audio_quality: params
                    .audio_quality
                    .unwrap_or(crate::audio::quality::AudioQuality::Unknown),
            };

            let window_metadata = crate::scanning::window::WindowMetadata {
                center_frequency_hz: params.center_frequency,
                window_id: params.window_id,
            };

            // Get tuner ID from pool status
            let status = self.pool.status();
            let tuner_id = status
                .tuners
                .iter()
                .find(|t| {
                    t.state == crate::hardware::pool::TunerState::Allocated
                        && t.activity == Some(crate::hardware::pool::TunerActivity::Listening)
                })
                .map(|t| t.id.device_id.clone());

            self.current_station = Some(StationEntity::from_signal(
                &signal,
                *self.scan_entity.id(),
                window_metadata,
            ));
            self.current_audio = Some(AudioEntity::new(signal, params.center_frequency, tuner_id));
        } else {
            debug!("TuneToCandidate received but no AudioSession exists");
        }

        Ok(())
    }
}
