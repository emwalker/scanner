mod audio;
mod config;
mod processing;

pub use config::{WindowConfig, WindowMetadata};

// Re-export public audio functions that are used by audio_session.rs
pub use audio::{
    create_audio_fm_graph, create_audio_stream, process_signal_for_audio, setup_audio_device,
};

use crate::core::types::{Result, ScanningConfig};
use crate::hardware::pool::{SegmentTrait, TunerProvider};
use crate::scanner_state::PauseSignal;
use crate::shutdown::ShutdownCoordinator;
use crate::ui::{ProgressEvent, ProgressEventType, ProgressReporter};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::debug;

pub struct Window {
    center_freq: f64,
    window_num: usize,
    total_windows: usize,
    station_mode: bool,
    tuner_provider: Arc<dyn TunerProvider>,
    config: Arc<ScanningConfig>,
    progress_reporter: Arc<dyn ProgressReporter>,
    shutdown_token: CancellationToken,
    metadata: WindowMetadata,
    pause_signal: Option<PauseSignal>,
}

impl Window {
    pub fn new(window_config: WindowConfig) -> Self {
        Self {
            center_freq: window_config.center_freq,
            window_num: window_config.window_num,
            total_windows: window_config.total_windows,
            station_mode: false,
            tuner_provider: window_config.tuner_provider,
            config: window_config.config,
            progress_reporter: window_config.progress_reporter,
            shutdown_token: window_config.shutdown_coordinator.token(),
            metadata: WindowMetadata {
                center_frequency_hz: window_config.center_freq,
                window_id: window_config.window_num,
            },
            pause_signal: window_config.pause_signal,
        }
    }

    pub fn for_station(
        center_freq: f64,
        window_num: usize,
        total_windows: usize,
        tuner_provider: Arc<dyn TunerProvider>,
        config: Arc<ScanningConfig>,
        progress_reporter: Arc<dyn ProgressReporter>,
        shutdown_coordinator: Arc<ShutdownCoordinator>,
    ) -> Self {
        Self {
            center_freq,
            window_num,
            total_windows,
            station_mode: true,
            tuner_provider,
            config,
            progress_reporter,
            shutdown_token: shutdown_coordinator.token(),
            metadata: WindowMetadata {
                center_frequency_hz: center_freq,
                window_id: window_num,
            },
            pause_signal: None,
        }
    }

    pub fn process_with_pool(&self) -> Result<()> {
        debug!(
            "Scanning window {} of {} at {:.1} MHz (pool-based)",
            self.window_num,
            self.total_windows,
            self.center_freq / 1e6
        );

        if self.shutdown_token.is_cancelled() {
            debug!("Shutdown requested before pool acquisition, aborting");
            return Ok(());
        }

        let requirements = crate::hardware::pool::TaskRequirements {
            frequency_hz: self.center_freq,
            bandwidth_hz: self.config.samp_rate,
            required_sample_rate: self.config.samp_rate,
            priority: crate::hardware::pool::TaskPriority::Normal,
        };

        let tuner = self.tuner_provider.acquire(
            &requirements,
            crate::hardware::pool::TunerActivity::Scanning,
        )?;
        debug!(tuner_id = ?tuner.id(), "Acquired tuner from tuner provider");

        let mut segment = crate::hardware::pool::Segment::from_tuner(
            tuner,
            self.center_freq,
            &self.config,
            self.shutdown_token.clone(),
        )?;

        let result = self.process(&segment);

        debug!("Processing complete, stopping stream before dropping segment");
        segment.stop_stream()?;

        debug!("Stream stopped, pool::Segment will drop and return tuner to pool");

        result
    }

    pub fn process(&self, segment: &dyn SegmentTrait) -> Result<()> {
        debug!(
            "Scanning window {} of {} at {:.1} MHz",
            self.window_num,
            self.total_windows,
            self.center_freq / 1e6
        );

        let peaks = processing::peaks(self.station_mode, self.center_freq, &self.config, segment)?;

        for peak in peaks.iter() {
            let candidate_id = format!("{:.1}-{}", peak.frequency_hz / 1e6, self.window_num);
            self.progress_reporter.report(ProgressEvent {
                event_type: ProgressEventType::PeakDetected,
                frequency_hz: peak.frequency_hz,
                metadata: self.metadata,
                candidate_id: Some(candidate_id),
                audio_quality: None,
                signal_strength: None,
                timestamp: std::time::Instant::now(),
                tuner_id: None,
            });
        }

        if !peaks.is_empty() {
            debug!("Found {} peaks in this window", peaks.len());
            processing::debug_peaks(self.window_num, self.center_freq, &self.config, &peaks);

            let candidates = processing::candidates_from_peaks(
                self.station_mode,
                self.window_num,
                self.center_freq,
                &self.config,
                &peaks,
            );

            for candidate in candidates.iter() {
                let freq = match candidate {
                    crate::core::types::Candidate::Fm(fm_candidate) => fm_candidate.frequency_hz,
                };
                let candidate_id = format!("{:.1}-{}", freq / 1e6, self.window_num);
                self.progress_reporter.report(ProgressEvent {
                    event_type: ProgressEventType::CandidateCreated,
                    frequency_hz: freq,
                    metadata: self.metadata,
                    candidate_id: Some(candidate_id),
                    audio_quality: None,
                    signal_strength: None,
                    timestamp: std::time::Instant::now(),
                    tuner_id: None,
                });
            }

            let ctx = processing::CandidateProcessingContext {
                window_num: self.window_num,
                center_freq: self.center_freq,
                config: &self.config,
                metadata: self.metadata,
                progress_reporter: &self.progress_reporter,
                pause_signal: &self.pause_signal,
            };

            let signals =
                processing::process_candidates(&ctx, candidates, segment, |threads, timeout| {
                    self.wait_for_threads_with_timeout(threads, timeout)
                })?;

            if self.shutdown_token.is_cancelled() {
                debug!("Shutdown requested after candidate processing, skipping audio playback");
                return Ok(());
            }

            audio::play_signals(
                self.window_num,
                &self.config,
                &self.progress_reporter,
                &self.shutdown_token,
                &self.pause_signal,
                signals,
                segment,
            )
        } else {
            debug!("No peaks detected in this window");
            processing::debug_peaks(self.window_num, self.center_freq, &self.config, &peaks);
            Ok(())
        }
    }

    fn wait_for_threads_with_timeout(
        &self,
        threads: Vec<thread::JoinHandle<Result<()>>>,
        timeout: Duration,
    ) -> usize {
        use std::time::Instant;

        let start_time = Instant::now();
        let mut completed = 0;
        let mut remaining_threads = threads;

        let check_interval = Duration::from_millis(100);

        while !remaining_threads.is_empty() && start_time.elapsed() < timeout {
            if self.should_stop_waiting() {
                break;
            }

            self.log_pause_signal_if_present(&remaining_threads);

            let (_newly_completed, still_running) =
                self.join_finished_threads(remaining_threads, &mut completed);
            remaining_threads = still_running;

            if !remaining_threads.is_empty() && start_time.elapsed() < timeout {
                std::thread::sleep(check_interval);
            }
        }

        completed += self.join_remaining_threads(remaining_threads, timeout);
        completed
    }

    fn should_stop_waiting(&self) -> bool {
        if self.shutdown_token.is_cancelled() {
            debug!("Shutdown signal detected, stopping thread wait");
            true
        } else {
            false
        }
    }

    fn log_pause_signal_if_present(&self, remaining_threads: &[thread::JoinHandle<Result<()>>]) {
        if let Some(ref signal) = self.pause_signal
            && signal.is_paused()
        {
            debug!(
                "Pause signal detected, continuing to wait for {} remaining threads to finish",
                remaining_threads.len()
            );
        }
    }

    fn join_finished_threads(
        &self,
        threads: Vec<thread::JoinHandle<Result<()>>>,
        completed: &mut usize,
    ) -> (usize, Vec<thread::JoinHandle<Result<()>>>) {
        let mut newly_completed = 0;
        let mut still_running = Vec::new();

        for handle in threads.into_iter() {
            if !handle.is_finished() {
                still_running.push(handle);
                continue;
            }

            match handle.join() {
                Ok(Ok(())) => {
                    *completed += 1;
                    newly_completed += 1;
                    debug!("Thread {} completed successfully", *completed);
                }
                Ok(Err(e)) => {
                    *completed += 1;
                    newly_completed += 1;
                    debug!("Thread {} completed with error: {}", *completed, e);
                }
                Err(_) => {
                    debug!("Thread {} panicked", *completed + 1);
                }
            }
        }

        (newly_completed, still_running)
    }

    fn join_remaining_threads(
        &self,
        remaining_threads: Vec<thread::JoinHandle<Result<()>>>,
        timeout: Duration,
    ) -> usize {
        if remaining_threads.is_empty() {
            return 0;
        }

        debug!(
            "{} threads timed out after {:?}, joining them now",
            remaining_threads.len(),
            timeout
        );

        let mut completed = 0;
        for handle in remaining_threads.into_iter() {
            let _ = handle.join();
            completed += 1;
        }

        completed
    }
}
