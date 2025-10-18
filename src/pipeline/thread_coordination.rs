use super::squelch_monitoring::AnalysisResult;
use crate::core::types::Result;
use crate::ecs::{CandidateEntity, CandidateId, Entities};

pub(crate) fn wait_for_threads_completion(
    detection_handle: std::thread::JoinHandle<()>,
    timer_handle: std::thread::JoinHandle<()>,
    frequency_hz: f64,
    result_rx: std::sync::mpsc::Receiver<AnalysisResult>,
    candidate_entities: &Option<Entities<CandidateEntity>>,
    window_id: usize,
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
        if let Ok(analysis_result) = result_rx.try_recv() {
            // Update candidate entity if this was a rejection
            if matches!(analysis_result, AnalysisResult::Noise)
                && let Some(entities_arc) = candidate_entities
            {
                let id = CandidateId::new(frequency_hz, window_id);
                if let Ok(mut entities) = entities_arc.write()
                    && let Some(entity) = entities.get_mut(&id)
                {
                    entity.reject();
                }
            }
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
