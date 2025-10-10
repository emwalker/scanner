use crate::audio::session::AudioSession;
use crate::core::types::{Result, ScanningConfig};
use crate::hardware::DeviceId;
use crate::hardware::pool::Pool;
use crate::main_thread::audio_coordinator::{AudioCoordinator, TuneParams};
use crate::scanner_state::PauseSignal;
use crate::scanner_state::ScannerState;
use crate::shutdown::ShutdownCoordinator;
use crate::ui::{ProgressReporter, ScannerCommand, TuiEvent};
use std::sync::Arc;
use std::sync::mpsc::Sender;
use tracing::debug;

pub struct CommandHandlerConfig<'a> {
    pub scanner_state: &'a mut ScannerState,
    pub pause_signal: &'a PauseSignal,
    pub pool: &'a Arc<Pool>,
    pub config: &'a ScanningConfig,
    pub shutdown_coordinator: &'a Arc<ShutdownCoordinator>,
    pub progress_reporter: &'a Arc<dyn ProgressReporter>,
    pub tui_event_sender: &'a Option<Sender<TuiEvent>>,
    pub current_playing: &'a mut Option<TuneParams>,
}

pub struct CommandHandler<'a> {
    scanner_state: &'a mut ScannerState,
    pause_signal: &'a PauseSignal,
    pool: &'a Arc<Pool>,
    config: &'a ScanningConfig,
    shutdown_coordinator: &'a Arc<ShutdownCoordinator>,
    progress_reporter: &'a Arc<dyn ProgressReporter>,
    tui_event_sender: &'a Option<Sender<TuiEvent>>,
    current_playing: &'a mut Option<TuneParams>,
}

impl<'a> CommandHandler<'a> {
    pub fn new(handler_config: CommandHandlerConfig<'a>) -> Self {
        Self {
            scanner_state: handler_config.scanner_state,
            pause_signal: handler_config.pause_signal,
            pool: handler_config.pool,
            config: handler_config.config,
            shutdown_coordinator: handler_config.shutdown_coordinator,
            progress_reporter: handler_config.progress_reporter,
            tui_event_sender: handler_config.tui_event_sender,
            current_playing: handler_config.current_playing,
        }
    }

    pub fn handle_pause(
        &mut self,
        window_num: usize,
        audio_session: &mut Option<AudioSession>,
    ) -> Result<()> {
        debug!(window = window_num, "Scanner paused, creating AudioSession");
        self.pause_signal.pause();
        self.scanner_state.handle_pause(window_num);

        *audio_session = Some(AudioSession::new(
            self.config,
            self.shutdown_coordinator.clone(),
        )?);
        debug!("AudioSession created for browse mode");

        // Send Paused event to TUI
        if let Some(sender) = self.tui_event_sender {
            let status = self.pool.status();
            let tuner_id = status
                .tuners
                .first()
                .map(|t| t.id.device_id.clone())
                .unwrap_or_else(|| DeviceId::from_serial("unknown", "0"));
            let _ = sender.send(TuiEvent::Paused { tuner_id });
        }

        Ok(())
    }

    pub fn handle_resume(&mut self, window_num: usize, audio_session: &mut Option<AudioSession>) {
        debug!(
            window = window_num,
            "Scanner resuming - exiting selection mode and continuing scan"
        );
        self.pause_signal.unpause();
        let _next_window = self.scanner_state.handle_resume();

        // Pool will automatically handle tuner state when AudioSession drops
        *audio_session = None;
        debug!("AudioSession dropped, returning to scan mode");
    }

    pub fn handle_tune_to_candidate(
        &mut self,
        window_num: usize,
        audio_session: &mut Option<AudioSession>,
        params: TuneParams,
    ) -> Result<()> {
        debug!(
            candidate_id = ?params.candidate_id,
            window_id = params.window_id,
            candidate_frequency_mhz = params.candidate_frequency / 1e6,
            "CommandHandler: Received TuneToCandidate command"
        );
        self.scanner_state.handle_tune(window_num);

        if let Some(session) = audio_session {
            let coordinator = AudioCoordinator::new(
                self.pool,
                self.config,
                self.shutdown_coordinator,
                self.progress_reporter,
            );
            coordinator.tune_to_station(session, params.clone())?;
            *self.current_playing = Some(params);
        } else {
            debug!("TuneToCandidate received but no AudioSession exists");
        }

        Ok(())
    }

    pub fn handle_stop_listening(&mut self, audio_session: &mut Option<AudioSession>) {
        let coordinator = AudioCoordinator::new(
            self.pool,
            self.config,
            self.shutdown_coordinator,
            self.progress_reporter,
        );

        let playing_params = self.current_playing.take();
        coordinator.stop_listening(audio_session, playing_params);
        self.scanner_state.handle_stop_listening();
    }

    pub fn handle_command(
        &mut self,
        command: ScannerCommand,
        window_num: usize,
        audio_session: &mut Option<AudioSession>,
    ) -> Result<()> {
        match command {
            ScannerCommand::Pause => {
                self.handle_pause(window_num, audio_session)?;
            }
            ScannerCommand::ResumeScan => {
                self.handle_resume(window_num, audio_session);
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
                self.handle_tune_to_candidate(window_num, audio_session, params)?;
            }
            ScannerCommand::StopListening => {
                self.handle_stop_listening(audio_session);
            }
        }
        Ok(())
    }
}
