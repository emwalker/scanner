use crate::core::types::{self, Peak, Result, ScanningConfig};
use std::sync::mpsc::SyncSender;

pub mod candidates;
pub mod deemph;
pub mod detection;
pub mod filter_config;
pub mod freq_xlating_fir;
pub mod frequency_tracking;
pub mod iq_capture;
pub mod peaks;
pub mod pipeline_builder;
pub mod squelch;
pub mod state;
pub mod throttle;

#[cfg(test)]
mod tests;

pub use detection::{DetectionGraphConfig, create_detection_graph};
pub use state::{PROCESSED_FREQUENCIES, clear_processed_frequencies};

#[derive(Debug, Clone)]
pub struct Candidate {
    pub frequency_hz: f64,
    pub peak_count: usize,
    pub max_magnitude: f32,
    pub avg_magnitude: f32,
    pub signal_strength: String,
}

impl Candidate {
    pub fn analyze(
        &self,
        sdr_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
        signal_tx: SyncSender<crate::core::types::Signal>,
        context: &crate::pipeline::AnalysisContext,
    ) -> Result<()> {
        // Delegate to the new testable pipeline function
        crate::pipeline::process_peak_to_signal(self.frequency_hz, sdr_rx, signal_tx, context)
    }
}

pub fn collect_peaks(
    config: &ScanningConfig,
    sdr_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
    center_freq: f64,
) -> Result<Vec<Peak>> {
    use tracing::debug;

    debug!(
        peak_scan_seconds = config.peak_detection.scan_duration,
        center_freq_mhz = center_freq / 1e6,
        "Starting unified peak detection scan"
    );

    let mut sdr_source = crate::testing::SdrStreamSource::new(
        sdr_rx,
        config.samp_rate,
        center_freq,
        config.peak_detection.scan_duration,
    );

    crate::signal::peaks::collect_peaks_from_source(config, &mut sdr_source)
}

pub fn find_candidates(
    peaks: &[Peak],
    config: &ScanningConfig,
    center_freq: f64,
) -> Vec<types::Candidate> {
    candidates::creation::find_candidates(peaks, config, center_freq)
}
