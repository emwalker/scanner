//! Core pipeline for processing peaks into signals
//!
//! This module provides the testable pipeline function that processes
//! individual frequency peaks through the complete analysis pipeline.

use crate::{
    core::types::{Result, ScanningConfig, Signal},
    ui::ProgressReporter,
};

mod detection;
mod frequency_refining;
mod frequency_tracking;
mod squelch_monitoring;
mod thread_coordination;

#[cfg(test)]
mod tests;

pub struct AnalysisContext<'a> {
    pub config: &'a ScanningConfig,
    pub center_freq: f64,
    pub progress_reporter: std::sync::Arc<dyn ProgressReporter + Send + Sync>,
    pub metadata: crate::scanning::window::WindowMetadata,
}

/// Process a single peak through the complete pipeline to generate a signal
pub fn process_peak_to_signal(
    frequency_hz: f64,
    sdr_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
    signal_tx: std::sync::mpsc::SyncSender<Signal>,
    context: &AnalysisContext,
) -> Result<()> {
    use crate::ui::{ProgressEvent, ProgressEventType};

    tracing::debug!(
        frequency_hz = frequency_hz / 1e6,
        "Processing peak through pipeline"
    );

    // Generate candidate ID for tracking
    let candidate_id = format!("{:.1}-{}", frequency_hz / 1e6, context.metadata.window_id);

    // Report peak detection
    context.progress_reporter.report(ProgressEvent {
        event_type: ProgressEventType::PeakDetected,
        frequency_hz,
        metadata: context.metadata,
        candidate_id: Some(candidate_id.clone()),
        audio_quality: None,
        signal_strength: None,
        timestamp: std::time::Instant::now(),
        tuner_id: None, // Tuner ID tracking removed with device field
    });

    let refined_frequency =
        frequency_refining::refine_frequency(frequency_hz, context.config, sdr_rx.resubscribe())?;

    if frequency_refining::is_frequency_already_processed(refined_frequency)? {
        tracing::debug!(
            freq_mhz = refined_frequency / 1e6,
            "Frequency already processed, skipping candidate creation"
        );
        return Ok(());
    }

    context.progress_reporter.report(ProgressEvent {
        event_type: ProgressEventType::CandidateCreated,
        frequency_hz: refined_frequency,
        metadata: context.metadata,
        candidate_id: Some(candidate_id.clone()),
        audio_quality: None,
        signal_strength: None,
        timestamp: std::time::Instant::now(),
        tuner_id: None,
    });

    detection::run_detection_analysis(
        frequency_hz,
        refined_frequency,
        sdr_rx,
        signal_tx,
        &candidate_id,
        context,
    )
}
