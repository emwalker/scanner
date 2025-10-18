use super::frequency_refining::mark_frequency_as_processed;

pub(crate) struct SquelchMonitoringParams {
    pub squelch_learning_duration: f32,
    pub refined_frequency: f64,
    pub original_frequency_hz: f64,
    #[allow(dead_code)]
    pub candidate_id: String,
    #[allow(dead_code)]
    pub metadata: crate::scanning::window::WindowMetadata,
    #[allow(dead_code)]
    pub tuner_id: Option<crate::hardware::DeviceId>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum AnalysisResult {
    Noise,
    AudioDetected,
    Timeout,
}

fn handle_noise_decision(
    _params: &SquelchMonitoringParams,
    result_sender: &std::sync::mpsc::Sender<AnalysisResult>,
    detection_cancel_token: &rustradio::graph::CancellationToken,
) {
    tracing::debug!("Squelch detected noise, exiting early");
    let _ = result_sender.send(AnalysisResult::Noise);
    detection_cancel_token.cancel();
}

fn handle_audio_decision(
    params: &SquelchMonitoringParams,
    result_sender: &std::sync::mpsc::Sender<AnalysisResult>,
    detection_cancel_token: &rustradio::graph::CancellationToken,
) {
    tracing::debug!(
        "squelch detected audio at {:.1} MHz",
        params.original_frequency_hz / 1e6
    );
    let frequency_khz = (params.refined_frequency / 1000.0) as u64;
    mark_frequency_as_processed(frequency_khz);

    let _ = result_sender.send(AnalysisResult::AudioDetected);

    tracing::debug!("Audio detected, terminating detection graph");
    detection_cancel_token.cancel();
}

fn handle_timeout(
    _params: &SquelchMonitoringParams,
    result_sender: &std::sync::mpsc::Sender<AnalysisResult>,
    detection_cancel_token: &rustradio::graph::CancellationToken,
    max_wait_time: f64,
) {
    tracing::debug!(
        "Squelch did not complete analysis after {:.1} seconds, moving to next candidate",
        max_wait_time
    );

    let _ = result_sender.send(AnalysisResult::Timeout);
    detection_cancel_token.cancel();
}

pub(crate) fn spawn_squelch_monitoring_thread(
    params: SquelchMonitoringParams,
    decision_state: std::sync::Arc<std::sync::atomic::AtomicU8>,
    detection_cancel_token: rustradio::graph::CancellationToken,
    result_sender: std::sync::mpsc::Sender<AnalysisResult>,
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
                    handle_noise_decision(&params, &result_sender, &detection_cancel_token);
                    return;
                }
                crate::signal::squelch::Decision::Audio => {
                    handle_audio_decision(&params, &result_sender, &detection_cancel_token);
                    return;
                }
                crate::signal::squelch::Decision::Learning => {
                    // Still learning, continue waiting
                }
            }
        }

        handle_timeout(
            &params,
            &result_sender,
            &detection_cancel_token,
            max_wait_time as f64,
        );
    })
}
