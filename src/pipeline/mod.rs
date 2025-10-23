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

use crate::ecs::WindowId;

pub struct AnalysisContext<'a> {
    pub config: &'a ScanningConfig,
    pub center_freq: f64,
    pub window_id: WindowId,
}

/// Process a single peak through the complete pipeline to generate a signal
///
/// Takes two receivers created at the same buffer position to avoid race conditions
/// where resubscribe() would be called after thread scheduling delay.
pub fn process_peak_to_signal(
    frequency_hz: f64,
    sdr_rx_refining: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
    sdr_rx_detection: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
    signal_tx: std::sync::mpsc::Sender<Signal>,
    context: &AnalysisContext,
) -> Result<()> {
    tracing::debug!(
        frequency_hz = frequency_hz / 1e6,
        "Processing peak through pipeline"
    );

    let signal_id = format!(
        "{:.1}-{}-{}",
        frequency_hz / 1e6,
        context.window_id.task_id,
        context.window_id.window_index
    );

    let refined_frequency =
        frequency_refining::refine_frequency(frequency_hz, context.config, sdr_rx_refining)?;

    if frequency_refining::is_frequency_already_processed(refined_frequency)? {
        tracing::debug!(
            freq_mhz = refined_frequency / 1e6,
            "Frequency already processed, skipping signal creation"
        );
        return Ok(());
    }

    detection::run_detection_analysis(
        frequency_hz,
        refined_frequency,
        sdr_rx_detection,
        signal_tx,
        &signal_id,
        context,
    )
}
