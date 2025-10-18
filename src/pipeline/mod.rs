//! Core pipeline for processing peaks into signals
//!
//! This module provides the testable pipeline function that processes
//! individual frequency peaks through the complete analysis pipeline.

use crate::core::types::{Result, ScanningConfig, Signal};

mod detection;
mod frequency_refining;
mod frequency_tracking;
mod squelch_monitoring;
mod thread_coordination;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod entity_lifecycle_tests;

use crate::ecs::{CandidateEntity, Entities};

pub struct AnalysisContext<'a> {
    pub config: &'a ScanningConfig,
    pub center_freq: f64,
    pub metadata: crate::scanning::window::WindowMetadata,
    pub candidate_entities: &'a Option<Entities<CandidateEntity>>,
}

/// Process a single peak through the complete pipeline to generate a signal
pub fn process_peak_to_signal(
    frequency_hz: f64,
    sdr_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
    signal_tx: std::sync::mpsc::SyncSender<Signal>,
    context: &AnalysisContext,
) -> Result<()> {
    tracing::debug!(
        frequency_hz = frequency_hz / 1e6,
        "Processing peak through pipeline"
    );

    let candidate_id = format!("{:.1}-{}", frequency_hz / 1e6, context.metadata.window_id);

    let refined_frequency =
        frequency_refining::refine_frequency(frequency_hz, context.config, sdr_rx.resubscribe())?;

    if frequency_refining::is_frequency_already_processed(refined_frequency)? {
        tracing::debug!(
            freq_mhz = refined_frequency / 1e6,
            "Frequency already processed, skipping candidate creation"
        );
        return Ok(());
    }

    detection::run_detection_analysis(
        frequency_hz,
        refined_frequency,
        sdr_rx,
        signal_tx,
        &candidate_id,
        context,
    )
}
