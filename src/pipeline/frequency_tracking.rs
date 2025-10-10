use crate::core::types::ScanningConfig;

fn parse_tracking_method(
    config: &ScanningConfig,
) -> crate::signal::frequency_tracking::TrackingMethod {
    use crate::signal::frequency_tracking::TrackingMethod;

    match config
        .signal_processing
        .frequency_tracking
        .method
        .parse::<TrackingMethod>()
    {
        Ok(method) => method,
        Err(e) => {
            tracing::debug!(error = %e, "Invalid frequency tracking method, falling back to PLL");
            TrackingMethod::Pll
        }
    }
}

fn create_tracking_config(
    config: &ScanningConfig,
    tracking_method: crate::signal::frequency_tracking::TrackingMethod,
) -> crate::signal::frequency_tracking::TrackingConfig {
    use crate::signal::frequency_tracking::TrackingConfig;

    TrackingConfig {
        method: tracking_method,
        convergence_threshold: config.signal_processing.frequency_tracking.accuracy,
        timeout_samples: (config.samp_rate * config.audio.squelch.learning_duration as f64 * 0.5)
            as usize,
        search_window: 200_000.0,
        min_samples_for_convergence: (config.samp_rate * 0.01) as usize,
    }
}

fn handle_tracking_state(
    state: &crate::signal::frequency_tracking::TrackingState,
    tracker: &dyn crate::signal::frequency_tracking::FrequencyTracker,
) -> Option<f64> {
    use crate::signal::frequency_tracking::TrackingState;

    match state {
        TrackingState::Converged(freq) => {
            tracing::debug!(
                refined_freq_mhz = freq / 1e6,
                confidence = tracker.confidence(),
                "Frequency tracking converged"
            );
            Some(*freq)
        }
        TrackingState::Failed => {
            tracing::debug!("Frequency tracking failed to converge");
            None
        }
        TrackingState::Timeout => {
            tracing::debug!("Frequency tracking timed out");
            None
        }
        TrackingState::Converging => None,
    }
}

pub(crate) fn run_frequency_tracking(
    frequency_hz: f64,
    config: &ScanningConfig,
    mut sdr_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
) -> Option<f64> {
    use crate::signal::frequency_tracking::create_tracker;

    let tracking_method = parse_tracking_method(config);
    let tracking_config = create_tracking_config(config, tracking_method);

    let mut tracker = create_tracker(
        tracking_method,
        frequency_hz,
        config.samp_rate,
        &tracking_config,
    );

    tracing::debug!(
        initial_freq_mhz = frequency_hz / 1e6,
        method = format!("{:?}", tracking_method),
        accuracy_hz = config.signal_processing.frequency_tracking.accuracy,
        timeout_ms = tracking_config.timeout_samples as f64 / config.samp_rate * 1000.0,
        "Starting frequency tracking"
    );

    loop {
        match sdr_rx.try_recv() {
            Ok(packet) => {
                for &sample in packet.as_slice() {
                    let state = tracker.process_sample(sample);
                    if let Some(freq) = handle_tracking_state(&state, tracker.as_ref()) {
                        return Some(freq);
                    }
                    if matches!(
                        state,
                        crate::signal::frequency_tracking::TrackingState::Failed
                            | crate::signal::frequency_tracking::TrackingState::Timeout
                    ) {
                        return None;
                    }
                }
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => {
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => {
                tracing::debug!("Frequency tracking lagged behind SDR stream");
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Closed) => {
                tracing::debug!("SDR stream closed during frequency tracking");
                return None;
            }
        }
    }
}
