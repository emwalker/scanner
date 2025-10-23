use super::squelch_monitoring::AnalysisResult;
use crate::core::types::Result;

pub(crate) fn wait_for_threads_completion(
    detection_handle: std::thread::JoinHandle<()>,
    timer_handle: std::thread::JoinHandle<()>,
    frequency_hz: f64,
    result_rx: std::sync::mpsc::Receiver<AnalysisResult>,
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

    for _ in 0..10 {
        if result_rx.try_recv().is_ok() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    tracing::debug!(
        "All detection threads completed for {:.1} MHz",
        frequency_hz / 1e6
    );

    Ok(())
}
