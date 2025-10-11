use super::MainThread;
use crate::audio::session::AudioSession;
use crate::core::types::Result;
use crate::scanner_state::ScanMode;
use tracing::debug;

// Loop control enum for scan_band state machine
pub enum LoopControl {
    Continue,
    Break,
    Advance,
}

pub struct ScanContext<'a> {
    main_thread: &'a mut MainThread,
    window_centers: Vec<f64>,
    windows_to_process: usize,
    window_index: usize,
    audio_session: Option<AudioSession>,
}

impl<'a> ScanContext<'a> {
    pub fn new(
        main_thread: &'a mut MainThread,
        window_centers: Vec<f64>,
        windows_to_process: usize,
    ) -> Self {
        Self {
            main_thread,
            window_centers,
            windows_to_process,
            window_index: 0,
            audio_session: None,
        }
    }

    pub fn advance(&mut self) {
        self.window_index += 1;
    }

    pub fn determine_next_action(&mut self) -> Result<LoopControl> {
        if self.main_thread.shutdown_coordinator.is_shutdown() {
            self.main_thread.scanner_state.shutdown();
        }

        match &self.main_thread.scanner_state.mode {
            ScanMode::ShuttingDown(_) => self.handle_shutting_down_mode(),
            ScanMode::ScanComplete(_) => self.handle_scan_complete_mode(),
            ScanMode::ScanCompletePaused(_) => self.handle_scan_complete_paused_mode(),
            ScanMode::Paused(_) => self.handle_paused_mode(),
            ScanMode::Listening(_) => self.handle_listening_mode(),
            ScanMode::Scanning(_) => self.handle_scanning_mode(),
        }
    }

    fn handle_shutting_down_mode(&self) -> Result<LoopControl> {
        debug!("Shutdown requested, stopping band scanning");
        Ok(LoopControl::Break)
    }

    fn handle_scan_complete_mode(&mut self) -> Result<LoopControl> {
        self.main_thread
            .check_and_handle_command(self.windows_to_process, &mut self.audio_session)?;
        std::thread::sleep(std::time::Duration::from_millis(100));
        Ok(LoopControl::Continue)
    }

    fn handle_scan_complete_paused_mode(&mut self) -> Result<LoopControl> {
        self.main_thread.process_commands(
            self.windows_to_process,
            self.window_centers.len(),
            &mut self.audio_session,
        )?;
        std::thread::sleep(std::time::Duration::from_millis(100));
        Ok(LoopControl::Continue)
    }

    fn handle_paused_mode(&mut self) -> Result<LoopControl> {
        if !self.window_index.is_multiple_of(50) {
            debug!(
                iteration = self.window_index,
                total = self.windows_to_process,
                "Paused - waiting for commands"
            );
        }
        self.main_thread.process_commands(
            self.window_index + 1,
            self.window_centers.len(),
            &mut self.audio_session,
        )?;
        std::thread::sleep(std::time::Duration::from_millis(100));
        Ok(LoopControl::Continue)
    }

    fn handle_listening_mode(&mut self) -> Result<LoopControl> {
        self.main_thread.process_commands(
            self.window_index + 1,
            self.window_centers.len(),
            &mut self.audio_session,
        )?;
        std::thread::sleep(std::time::Duration::from_millis(100));
        Ok(LoopControl::Continue)
    }

    fn handle_scanning_mode(&mut self) -> Result<LoopControl> {
        if self.window_index >= self.windows_to_process {
            debug!("Scan band complete - all windows processed");
            self.main_thread
                .scanner_state
                .mark_scan_complete(self.windows_to_process);
            return Ok(LoopControl::Continue);
        }

        debug!(
            iteration = self.window_index,
            total = self.windows_to_process,
            "Start of scan loop iteration"
        );

        if self.main_thread.process_commands_with_pause_check(
            self.window_index + 1,
            self.window_centers.len(),
            &mut self.audio_session,
        )? {
            return Ok(LoopControl::Continue);
        }

        let center_freq = self.window_centers[self.window_index];
        self.main_thread.process_window(
            self.window_index + 1,
            center_freq,
            self.window_centers.len(),
        )?;
        self.main_thread.process_commands(
            self.window_index + 1,
            self.window_centers.len(),
            &mut self.audio_session,
        )?;

        debug!(
            completed_window = self.window_index + 1,
            next_window = self.window_index + 2,
            remaining = self.windows_to_process - self.window_index - 1,
            "Window complete, advancing to next"
        );

        Ok(LoopControl::Advance)
    }
}
