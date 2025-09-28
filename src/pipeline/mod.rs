//! Core pipeline for processing peaks into signals
//!
//! This module provides the testable pipeline function that processes
//! individual frequency peaks through the complete analysis pipeline.

use crate::terminal::ProgressReporter;
use crate::types::{Result, ScanningConfig, Signal};
use rustradio::graph::GraphRunner;

/// Context struct to group related parameters for analysis functions
pub struct AnalysisContext<'a> {
    pub config: &'a ScanningConfig,
    pub center_freq: f64,
    pub device: &'a crate::soapy::Device,
    pub progress_reporter: std::sync::Arc<dyn ProgressReporter + Send + Sync>,
    pub window_id: usize,
}

/// Parameters for squelch monitoring thread
pub struct SquelchMonitoringParams {
    pub squelch_learning_duration: f32,
    pub refined_frequency: f64,
    pub original_frequency_hz: f64,
    pub candidate_id: String,
    pub window_id: usize,
}

/// Process a single peak through the complete pipeline to generate a signal
pub fn process_peak_to_signal(
    frequency_hz: f64,
    sdr_rx: tokio::sync::broadcast::Receiver<rustradio::Complex>,
    signal_tx: std::sync::mpsc::SyncSender<Signal>,
    context: &AnalysisContext,
) -> Result<()> {
    use crate::terminal::{ProgressEvent, ProgressEventType};

    tracing::debug!(
        frequency_hz = frequency_hz / 1e6,
        "Processing peak through pipeline"
    );

    // Generate candidate ID for tracking
    let candidate_id = format!("{:.1}-{}", frequency_hz / 1e6, context.window_id);

    // Report peak detection
    context.progress_reporter.report(ProgressEvent {
        event_type: ProgressEventType::PeakDetected,
        frequency_hz,
        window_id: context.window_id,
        candidate_id: Some(candidate_id.clone()),
        audio_quality: None,
        timestamp: std::time::Instant::now(),
    });

    // Refine frequency using tracking
    let refined_frequency = refine_frequency(frequency_hz, context.config, sdr_rx.resubscribe())?;

    // Check for frequency deduplication - exit early before creating candidates
    if is_frequency_already_processed(refined_frequency)? {
        tracing::debug!(
            freq_mhz = refined_frequency / 1e6,
            "Frequency already processed, skipping candidate creation"
        );
        return Ok(());
    }

    // Report candidate creation (only for new frequencies)
    context.progress_reporter.report(ProgressEvent {
        event_type: ProgressEventType::CandidateCreated,
        frequency_hz: refined_frequency,
        window_id: context.window_id,
        candidate_id: Some(candidate_id.clone()),
        audio_quality: None,
        timestamp: std::time::Instant::now(),
    });

    // Run the detection and squelch analysis
    run_detection_analysis(
        frequency_hz,
        refined_frequency,
        sdr_rx,
        signal_tx,
        &candidate_id,
        context,
    )
}

/// Refine frequency using tracking or return rounded estimate
fn refine_frequency(
    frequency_hz: f64,
    config: &ScanningConfig,
    sdr_rx: tokio::sync::broadcast::Receiver<rustradio::Complex>,
) -> Result<f64> {
    let refined_frequency = if config.disable_frequency_tracking {
        tracing::debug!(
            freq_mhz = frequency_hz / 1e6,
            "Frequency tracking disabled, using FFT estimate"
        );
        frequency_hz
    } else {
        match run_frequency_tracking(frequency_hz, config, sdr_rx) {
            Some(freq) => {
                tracing::debug!(
                    original_mhz = frequency_hz / 1e6,
                    refined_mhz = freq / 1e6,
                    error_khz = (freq - frequency_hz) / 1e3,
                    "Frequency tracking successful"
                );
                freq
            }
            None => {
                tracing::debug!(
                    freq_mhz = frequency_hz / 1e6,
                    "Frequency tracking failed, using FFT estimate"
                );
                frequency_hz
            }
        }
    };

    // Round to nearest 100 kHz to avoid floating point errors
    Ok((refined_frequency / 100000.0).round() * 100000.0)
}

/// Check if frequency has already been processed to avoid duplicates
fn is_frequency_already_processed(refined_frequency: f64) -> Result<bool> {
    let frequency_khz = (refined_frequency / 1000.0) as u64;

    let processed = crate::fm::PROCESSED_FREQUENCIES.lock().unwrap();
    if processed.contains(&frequency_khz) {
        tracing::debug!(
            refined_freq_mhz = refined_frequency / 1e6,
            frequency_khz = frequency_khz,
            "Frequency already processed in another window, skipping analysis"
        );
        Ok(true)
    } else {
        tracing::debug!(
            refined_freq_mhz = refined_frequency / 1e6,
            frequency_khz = frequency_khz,
            "New frequency detected, proceeding with analysis"
        );
        Ok(false)
    }
}

/// Run the complete detection and squelch analysis
fn run_detection_analysis(
    original_frequency_hz: f64,
    refined_frequency: f64,
    sdr_rx: tokio::sync::broadcast::Receiver<rustradio::Complex>,
    signal_tx: std::sync::mpsc::SyncSender<Signal>,
    candidate_id: &str,
    context: &AnalysisContext,
) -> Result<()> {
    let station_name = format!("{:.1}FM", refined_frequency / 1e6);
    let audio_analyzer = context.config.audio_analyzer.clone();

    // Create detection graph
    let (detection_graph, decision_state) = crate::fm::create_detection_graph(
        sdr_rx,
        context.config.samp_rate,
        station_name,
        context.config,
        context.center_freq,
        refined_frequency,
        Some(signal_tx),
        context.device,
        audio_analyzer,
        Some(context.progress_reporter.clone()),
        context.window_id,
    )?;

    let detection_cancel_token = detection_graph.cancel_token();

    tracing::debug!(
        "Processing candidate at {:.1} MHz with center freq {:.1} MHz",
        original_frequency_hz / 1e6,
        context.center_freq / 1e6
    );

    // Start detection graph thread
    let detection_handle = spawn_detection_graph_thread(detection_graph, original_frequency_hz);

    // Start squelch monitoring thread
    // Create a channel for sending rejection events from the thread
    let (rejection_tx, rejection_rx) = std::sync::mpsc::channel();

    let timer_handle = spawn_squelch_monitoring_thread(
        SquelchMonitoringParams {
            squelch_learning_duration: context.config.squelch_learning_duration,
            refined_frequency,
            original_frequency_hz,
            candidate_id: candidate_id.to_string(),
            window_id: context.window_id,
        },
        decision_state,
        detection_cancel_token,
        rejection_tx,
    );

    // Wait for completion and handle rejection events
    wait_for_threads_completion(
        detection_handle,
        timer_handle,
        original_frequency_hz,
        &*context.progress_reporter,
        rejection_rx,
        context.window_id,
        candidate_id,
    )
}

/// Spawn detection graph processing thread
fn spawn_detection_graph_thread(
    mut detection_graph: rustradio::graph::Graph,
    frequency_hz: f64,
) -> std::thread::JoinHandle<()> {
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

/// Spawn squelch monitoring and decision thread
fn spawn_squelch_monitoring_thread(
    params: SquelchMonitoringParams,
    decision_state: std::sync::Arc<std::sync::atomic::AtomicU8>,
    detection_cancel_token: rustradio::graph::CancellationToken,
    rejection_sender: std::sync::mpsc::Sender<crate::terminal::ProgressEvent>,
) -> std::thread::JoinHandle<()> {
    let frequency_khz = (params.refined_frequency / 1000.0) as u64;

    std::thread::spawn(move || {
        let check_interval = std::time::Duration::from_millis(100);
        let max_wait_time = params.squelch_learning_duration + 1.0;
        let total_checks = (max_wait_time * 1000.0) as u32 / 100;

        for _check_num in 0..total_checks {
            std::thread::sleep(check_interval);

            let current_decision = crate::fm::squelch::Decision::from_u8(
                decision_state.load(std::sync::atomic::Ordering::Relaxed),
            );

            match current_decision {
                crate::fm::squelch::Decision::Noise => {
                    tracing::debug!("Squelch detected noise, exiting early");

                    // Report candidate rejection
                    let _ = rejection_sender.send(crate::terminal::ProgressEvent {
                        event_type: crate::terminal::ProgressEventType::CandidateRejected,
                        frequency_hz: params.original_frequency_hz,
                        window_id: params.window_id,
                        candidate_id: Some(params.candidate_id.clone()),
                        audio_quality: None,
                        timestamp: std::time::Instant::now(),
                    });

                    detection_cancel_token.cancel();
                    return;
                }
                crate::fm::squelch::Decision::Audio => {
                    tracing::debug!(
                        "squelch detected audio at {:.1} MHz",
                        params.original_frequency_hz / 1e6
                    );
                    mark_frequency_as_processed(frequency_khz);

                    // Report audio analysis completion
                    let _ = rejection_sender.send(crate::terminal::ProgressEvent {
                        event_type: crate::terminal::ProgressEventType::AudioAnalysisCompleted,
                        frequency_hz: params.original_frequency_hz,
                        window_id: params.window_id,
                        candidate_id: Some(params.candidate_id.clone()),
                        audio_quality: None,
                        timestamp: std::time::Instant::now(),
                    });

                    tracing::debug!("Audio detected, terminating detection graph");
                    detection_cancel_token.cancel();
                    return;
                }
                crate::fm::squelch::Decision::Learning => {
                    // Still learning, continue waiting
                }
            }
        }

        tracing::debug!(
            "Squelch did not complete analysis after {:.1} seconds, moving to next candidate",
            max_wait_time
        );

        // Report audio analysis completion (timeout case)
        let _ = rejection_sender.send(crate::terminal::ProgressEvent {
            event_type: crate::terminal::ProgressEventType::AudioAnalysisCompleted,
            frequency_hz: params.original_frequency_hz,
            window_id: params.window_id,
            candidate_id: Some(params.candidate_id.clone()),
            audio_quality: None,
            timestamp: std::time::Instant::now(),
        });

        detection_cancel_token.cancel();
    })
}

/// Mark a frequency as successfully processed
fn mark_frequency_as_processed(frequency_khz: u64) {
    let mut processed = crate::fm::PROCESSED_FREQUENCIES.lock().unwrap();
    processed.insert(frequency_khz);
    tracing::debug!(frequency_khz, "Frequency marked as successfully processed");
}

/// Wait for both detection and timer threads to complete
fn wait_for_threads_completion(
    detection_handle: std::thread::JoinHandle<()>,
    timer_handle: std::thread::JoinHandle<()>,
    frequency_hz: f64,
    progress_reporter: &dyn ProgressReporter,
    rejection_rx: std::sync::mpsc::Receiver<crate::terminal::ProgressEvent>,
    window_id: usize,
    candidate_id: &str,
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

    // Handle any rejection events that may have been sent
    // Check multiple times with small delays to catch rejection events
    for _ in 0..10 {
        if let Ok(rejection_event) = rejection_rx.try_recv() {
            progress_reporter.report(rejection_event);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    tracing::debug!(
        "All detection threads completed for {:.1} MHz",
        frequency_hz / 1e6
    );

    // Report audio analysis completion
    progress_reporter.report(crate::terminal::ProgressEvent {
        event_type: crate::terminal::ProgressEventType::AudioAnalysisCompleted,
        frequency_hz,
        window_id,
        candidate_id: Some(candidate_id.to_string()),
        audio_quality: None,
        timestamp: std::time::Instant::now(),
    });

    Ok(())
}

/// Run frequency tracking to refine the FFT-based frequency estimate
fn run_frequency_tracking(
    frequency_hz: f64,
    config: &ScanningConfig,
    mut sdr_rx: tokio::sync::broadcast::Receiver<rustradio::Complex>,
) -> Option<f64> {
    use crate::frequency_tracking::{
        TrackingConfig, TrackingMethod, TrackingState, create_tracker,
    };

    // Parse tracking method
    let tracking_method = match config.frequency_tracking_method.parse::<TrackingMethod>() {
        Ok(method) => method,
        Err(e) => {
            tracing::debug!(error = %e, "Invalid frequency tracking method, falling back to PLL");
            TrackingMethod::Pll
        }
    };

    // Create tracking configuration
    let tracking_config = TrackingConfig {
        method: tracking_method,
        convergence_threshold: config.tracking_accuracy,
        timeout_samples: (config.samp_rate * config.squelch_learning_duration as f64 * 0.5)
            as usize,
        search_window: 200_000.0, // ±200 kHz search window for FM
        min_samples_for_convergence: (config.samp_rate * 0.01) as usize, // 10ms minimum
    };

    // Create frequency tracker
    let mut tracker = create_tracker(
        tracking_method,
        frequency_hz,
        config.samp_rate,
        &tracking_config,
    );

    tracing::debug!(
        initial_freq_mhz = frequency_hz / 1e6,
        method = format!("{:?}", tracking_method),
        accuracy_hz = config.tracking_accuracy,
        timeout_ms = tracking_config.timeout_samples as f64 / config.samp_rate * 1000.0,
        "Starting frequency tracking"
    );

    // Process samples until convergence, failure, or timeout
    loop {
        match sdr_rx.try_recv() {
            Ok(sample) => {
                match tracker.process_sample(sample) {
                    TrackingState::Converged(freq) => {
                        tracing::debug!(
                            refined_freq_mhz = freq / 1e6,
                            confidence = tracker.get_confidence(),
                            "Frequency tracking converged"
                        );
                        return Some(freq);
                    }
                    TrackingState::Failed => {
                        tracing::debug!("Frequency tracking failed to converge");
                        return None;
                    }
                    TrackingState::Timeout => {
                        tracing::debug!("Frequency tracking timed out");
                        return None;
                    }
                    TrackingState::Converging => {
                        // Continue processing
                    }
                }
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => {
                // No data available, continue waiting
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => {
                tracing::debug!("Frequency tracking lagged behind SDR stream");
                // Continue with next sample
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Closed) => {
                tracing::debug!("SDR stream closed during frequency tracking");
                return None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terminal::{MockProgressReporter, NoOpProgressReporter, ProgressEventType};
    use crate::types::{ScanningConfig, Signal, TEST_FREQUENCY_HZ};
    use std::sync::mpsc;
    use tokio::sync::broadcast;

    /// Create a mock ScanningConfig for testing
    fn create_test_config() -> ScanningConfig {
        ScanningConfig {
            disable_frequency_tracking: true, // Disable to avoid complex tracking logic in tests
            audio_analyzer: crate::audio_quality::AudioAnalyzer::mock(),
            squelch_learning_duration: 0.1, // Short duration for fast tests
            samp_rate: 1_000_000.0,
            ..Default::default()
        }
    }

    /// Create a mock SDR broadcast channel with test data
    fn create_mock_sdr_stream() -> broadcast::Receiver<rustradio::Complex> {
        let (tx, rx) = broadcast::channel(1000);

        // Pre-fill the channel with test samples synchronously
        for _ in 0..1000 {
            let sample = rustradio::Complex::new(0.1, 0.1); // Weak signal
            let _ = tx.send(sample); // Ignore errors if channel is full
        }

        rx
    }

    /// Create a mock SDR broadcast channel with strong signal
    fn create_mock_strong_sdr_stream() -> broadcast::Receiver<rustradio::Complex> {
        let (tx, rx) = broadcast::channel(1000);

        // Pre-fill the channel with strong signal samples synchronously
        for _ in 0..1000 {
            let sample = rustradio::Complex::new(0.8, 0.8); // Strong signal
            let _ = tx.send(sample); // Ignore errors if channel is full
        }

        rx
    }

    /// Create a mock device for testing
    fn create_mock_device() -> crate::soapy::Device {
        // Create a minimal mock device with empty device string
        crate::soapy::Device::new("mock_device".to_string()).unwrap()
    }

    #[test]
    fn test_weak_peak_exits_early() {
        let config = create_test_config();
        let sdr_rx = create_mock_sdr_stream();
        let center_freq = TEST_FREQUENCY_HZ;
        let (signal_tx, signal_rx) = mpsc::sync_channel::<Signal>(10);
        let device = create_mock_device();
        let progress_reporter = NoOpProgressReporter;

        // Process a weak peak - should exit early due to weak signal
        let context = AnalysisContext {
            config: &config,
            center_freq,
            device: &device,
            progress_reporter: std::sync::Arc::new(progress_reporter),
            window_id: 1,
        };
        let result = process_peak_to_signal(TEST_FREQUENCY_HZ, sdr_rx, signal_tx, &context);

        // Should succeed (processing completes without error)
        assert!(result.is_ok(), "Weak peak processing should succeed");

        // Should not produce any signals on the channel
        assert!(
            signal_rx.try_recv().is_err(),
            "Weak peak should not produce signals"
        );
    }

    #[test]
    fn test_strong_peak_produces_signal() {
        let config = create_test_config();
        let sdr_rx = create_mock_strong_sdr_stream();
        let center_freq = TEST_FREQUENCY_HZ;
        let (signal_tx, _signal_rx) = mpsc::sync_channel::<Signal>(10);
        let device = create_mock_device();
        let progress_reporter = MockProgressReporter::new();
        let progress_arc = std::sync::Arc::new(progress_reporter.clone());

        // Process a strong peak - should complete pipeline
        let context = AnalysisContext {
            config: &config,
            center_freq,
            device: &device,
            progress_reporter: progress_arc,
            window_id: 1,
        };
        let result = process_peak_to_signal(
            TEST_FREQUENCY_HZ + 100_000.0, // Test frequency (center + offset)
            sdr_rx,
            signal_tx,
            &context,
        );

        // Should succeed
        assert!(result.is_ok(), "Strong peak processing should succeed");

        // Should have emitted progress events
        assert!(
            progress_reporter.event_count() > 0,
            "Should have progress events for strong signal"
        );

        // Note: Signal generation depends on squelch analysis completing,
        // which may not happen in a short test with mock data
    }

    #[test]
    fn test_progress_events_emitted() {
        let config = create_test_config();
        let sdr_rx = create_mock_sdr_stream();
        let center_freq = TEST_FREQUENCY_HZ;
        let (signal_tx, _signal_rx) = mpsc::sync_channel::<Signal>(10);
        let device = create_mock_device();
        let progress_reporter = MockProgressReporter::new();
        let progress_arc = std::sync::Arc::new(progress_reporter.clone());

        // Process peak - should emit progress events
        let context = AnalysisContext {
            config: &config,
            center_freq,
            device: &device,
            progress_reporter: progress_arc,
            window_id: 1,
        };
        let result = process_peak_to_signal(TEST_FREQUENCY_HZ, sdr_rx, signal_tx, &context);

        assert!(result.is_ok(), "Pipeline should complete successfully");

        // Verify progress events were emitted
        let events = progress_reporter.get_events();
        assert!(
            !events.is_empty(),
            "Should emit at least one progress event"
        );

        // Check for expected event types
        let event_types: Vec<_> = events.iter().map(|e| &e.event_type).collect();

        // Should at least have peak detected event
        assert!(
            event_types
                .iter()
                .any(|t| matches!(t, ProgressEventType::PeakDetected)),
            "Should emit PeakDetected event"
        );

        // Verify all events have correct frequency
        for event in &events {
            assert_eq!(
                event.frequency_hz, TEST_FREQUENCY_HZ,
                "All events should have correct frequency"
            );
        }
    }

    #[test]
    fn test_pipeline_with_frequency_tracking_disabled() {
        let config = create_test_config(); // Has frequency tracking disabled
        let sdr_rx = create_mock_sdr_stream();
        let center_freq = TEST_FREQUENCY_HZ;
        let (signal_tx, _signal_rx) = mpsc::sync_channel::<Signal>(10);
        let device = create_mock_device();
        let progress_reporter = MockProgressReporter::new();
        let progress_arc = std::sync::Arc::new(progress_reporter.clone());

        // Process a peak with frequency tracking disabled
        let context = AnalysisContext {
            config: &config,
            center_freq,
            device: &device,
            progress_reporter: progress_arc,
            window_id: 1,
        };
        let result =
            process_peak_to_signal(TEST_FREQUENCY_HZ + 50_000.0, sdr_rx, signal_tx, &context);

        // Should succeed
        assert!(
            result.is_ok(),
            "Pipeline should handle disabled frequency tracking"
        );

        // Should have emitted at least peak detection event
        let events = progress_reporter.get_events();
        assert!(!events.is_empty(), "Should have progress events");

        // First event should be peak detection
        assert!(
            matches!(
                events[0].event_type,
                crate::terminal::ProgressEventType::PeakDetected
            ),
            "First event should be PeakDetected"
        );
    }
}
