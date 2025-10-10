use super::frequency_refining::mark_frequency_as_processed;

pub(crate) struct SquelchMonitoringParams {
    pub squelch_learning_duration: f32,
    pub refined_frequency: f64,
    pub original_frequency_hz: f64,
    pub candidate_id: String,
    pub metadata: crate::scanning::window::WindowMetadata,
    pub tuner_id: Option<crate::hardware::DeviceId>,
}

fn create_progress_event(
    event_type: crate::ui::ProgressEventType,
    params: &SquelchMonitoringParams,
) -> crate::ui::ProgressEvent {
    crate::ui::ProgressEvent {
        event_type,
        frequency_hz: params.original_frequency_hz,
        metadata: params.metadata,
        candidate_id: Some(params.candidate_id.clone()),
        audio_quality: None,
        signal_strength: None,
        timestamp: std::time::Instant::now(),
        tuner_id: params.tuner_id.clone(),
    }
}

fn handle_noise_decision(
    params: &SquelchMonitoringParams,
    rejection_sender: &std::sync::mpsc::Sender<crate::ui::ProgressEvent>,
    detection_cancel_token: &rustradio::graph::CancellationToken,
) {
    tracing::debug!("Squelch detected noise, exiting early");

    let event = create_progress_event(crate::ui::ProgressEventType::CandidateRejected, params);
    let _ = rejection_sender.send(event);
    detection_cancel_token.cancel();
}

fn handle_audio_decision(
    params: &SquelchMonitoringParams,
    rejection_sender: &std::sync::mpsc::Sender<crate::ui::ProgressEvent>,
    detection_cancel_token: &rustradio::graph::CancellationToken,
) {
    tracing::debug!(
        "squelch detected audio at {:.1} MHz",
        params.original_frequency_hz / 1e6
    );
    let frequency_khz = (params.refined_frequency / 1000.0) as u64;
    mark_frequency_as_processed(frequency_khz);

    let event = create_progress_event(crate::ui::ProgressEventType::AudioAnalysisCompleted, params);
    let _ = rejection_sender.send(event);

    tracing::debug!("Audio detected, terminating detection graph");
    detection_cancel_token.cancel();
}

fn handle_timeout(
    params: &SquelchMonitoringParams,
    rejection_sender: &std::sync::mpsc::Sender<crate::ui::ProgressEvent>,
    detection_cancel_token: &rustradio::graph::CancellationToken,
    max_wait_time: f64,
) {
    tracing::debug!(
        "Squelch did not complete analysis after {:.1} seconds, moving to next candidate",
        max_wait_time
    );

    let event = create_progress_event(crate::ui::ProgressEventType::AudioAnalysisCompleted, params);
    let _ = rejection_sender.send(event);
    detection_cancel_token.cancel();
}

pub(crate) fn spawn_squelch_monitoring_thread(
    params: SquelchMonitoringParams,
    decision_state: std::sync::Arc<std::sync::atomic::AtomicU8>,
    detection_cancel_token: rustradio::graph::CancellationToken,
    rejection_sender: std::sync::mpsc::Sender<crate::ui::ProgressEvent>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let check_interval = std::time::Duration::from_millis(100);
        let max_wait_time = params.squelch_learning_duration + 1.0;
        let total_checks = (max_wait_time * 1000.0) as u32 / 100;

        for _check_num in 0..total_checks {
            std::thread::sleep(check_interval);

            let current_decision = crate::signal::squelch::Decision::from_u8(
                decision_state.load(std::sync::atomic::Ordering::Relaxed),
            );

            match current_decision {
                crate::signal::squelch::Decision::Noise => {
                    handle_noise_decision(&params, &rejection_sender, &detection_cancel_token);
                    return;
                }
                crate::signal::squelch::Decision::Audio => {
                    handle_audio_decision(&params, &rejection_sender, &detection_cancel_token);
                    return;
                }
                crate::signal::squelch::Decision::Learning => {
                    // Still learning, continue waiting
                }
            }
        }

        handle_timeout(
            &params,
            &rejection_sender,
            &detection_cancel_token,
            max_wait_time as f64,
        );
    })
}
