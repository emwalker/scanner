use crate::core::types::{Result, Signal};
use rustradio::graph::GraphRunner;

use super::{
    AnalysisContext,
    squelch_monitoring::{SquelchMonitoringParams, spawn_squelch_monitoring_thread},
    thread_coordination::wait_for_threads_completion,
};

fn build_detection_graph_config<'a>(
    refined_frequency: f64,
    sdr_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
    signal_tx: std::sync::mpsc::SyncSender<Signal>,
    context: &'a AnalysisContext,
) -> crate::signal::DetectionGraphConfig<'a> {
    let audio_analyzer = context.config.audio.analyzer.clone();

    crate::signal::DetectionGraphConfig {
        source_receiver: sdr_rx,
        samp_rate: context.config.samp_rate,
        config: context.config,
        center_freq: context.center_freq,
        tune_freq: refined_frequency,
        signal_tx: Some(signal_tx),
        audio_analyzer,
        progress_reporter: Some(context.progress_reporter.clone()),
        window_id: context.metadata.window_id,
    }
}

pub(crate) fn run_detection_analysis(
    original_frequency_hz: f64,
    refined_frequency: f64,
    sdr_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
    signal_tx: std::sync::mpsc::SyncSender<Signal>,
    candidate_id: &str,
    context: &AnalysisContext,
) -> Result<()> {
    let graph_config = build_detection_graph_config(refined_frequency, sdr_rx, signal_tx, context);
    let (detection_graph, decision_state) = crate::signal::create_detection_graph(graph_config)?;

    let detection_cancel_token = detection_graph.cancel_token();

    tracing::debug!(
        "Processing candidate at {:.1} MHz with center freq {:.1} MHz",
        original_frequency_hz / 1e6,
        context.center_freq / 1e6
    );

    let detection_handle = spawn_detection_graph_thread(detection_graph, original_frequency_hz);

    let (rejection_tx, rejection_rx) = std::sync::mpsc::channel();

    let timer_handle = spawn_squelch_monitoring_thread(
        SquelchMonitoringParams {
            squelch_learning_duration: context.config.audio.squelch.learning_duration,
            refined_frequency,
            original_frequency_hz,
            candidate_id: candidate_id.to_string(),
            metadata: context.metadata,
            tuner_id: None,
        },
        decision_state,
        detection_cancel_token,
        rejection_tx,
    );

    wait_for_threads_completion(
        detection_handle,
        timer_handle,
        original_frequency_hz,
        &*context.progress_reporter,
        rejection_rx,
        candidate_id,
        None,
    )
}

fn spawn_detection_graph_thread(
    mut detection_graph: rustradio::graph::Graph,
    frequency_hz: f64,
) -> std::thread::JoinHandle<()> {
    use rustradio::graph::GraphRunner;

    std::thread::spawn(move || {
        tracing::debug!("Detection graph started for {:.1} MHz", frequency_hz / 1e6);
        if let Err(e) = detection_graph.run() {
            tracing::debug!("Detection graph error for {}: {}", frequency_hz / 1e6, e);
        }
        tracing::debug!(
            "Detection graph terminated for {:.1} MHz",
            frequency_hz / 1e6
        );
    })
}
