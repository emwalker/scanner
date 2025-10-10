use crate::{core::types::Result, ui::ProgressReporter};

pub(crate) fn wait_for_threads_completion(
    detection_handle: std::thread::JoinHandle<()>,
    timer_handle: std::thread::JoinHandle<()>,
    frequency_hz: f64,
    progress_reporter: &dyn ProgressReporter,
    rejection_rx: std::sync::mpsc::Receiver<crate::ui::ProgressEvent>,
    candidate_id: &str,
    tuner_id: Option<crate::hardware::DeviceId>,
) -> Result<()> {
    tracing::debug!(
        "Waiting for detection graph and timer threads to complete for {:.1} MHz",
        frequency_hz / 1e6
    );

    if let Err(e) = timer_handle.join() {
        tracing::debug!(
            "Timer thread panicked for {:.1} MHz: {:?}",
            frequency_hz / 1e6,
            e
        );
    }

    if let Err(e) = detection_handle.join() {
        tracing::debug!(
            "Detection graph thread panicked for {:.1} MHz: {:?}",
            frequency_hz / 1e6,
            e
        );
    }

    let mut received_metadata = None;
    for _ in 0..10 {
        if let Ok(rejection_event) = rejection_rx.try_recv() {
            received_metadata = Some(rejection_event.metadata);
            progress_reporter.report(rejection_event);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    tracing::debug!(
        "All detection threads completed for {:.1} MHz",
        frequency_hz / 1e6
    );

    if let Some(metadata) = received_metadata {
        progress_reporter.report(crate::ui::ProgressEvent {
            event_type: crate::ui::ProgressEventType::AudioAnalysisCompleted,
            frequency_hz,
            metadata,
            candidate_id: Some(candidate_id.to_string()),
            audio_quality: None,
            signal_strength: None,
            timestamp: std::time::Instant::now(),
            tuner_id,
        });
    }

    Ok(())
}
