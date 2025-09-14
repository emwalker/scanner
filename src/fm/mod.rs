use crate::{
    file::AudioCaptureBlock,
    fm::squelch::SquelchBlock,
    testing,
    types::{self, Peak, Result, ScanningConfig},
};
use rustradio::blocks::QuadratureDemod;
use rustradio::graph::{Graph, GraphRunner};
use rustradio::{Complex, blockchain};
use std::{
    collections::HashSet,
    sync::{LazyLock, Mutex},
    thread,
    time::Duration,
};
use tracing::debug;

/// Global set of processed frequencies (rounded to nearest kHz) to avoid duplicate analysis
static PROCESSED_FREQUENCIES: LazyLock<Mutex<HashSet<u64>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// Clear the processed frequencies set for a new scanning session
pub fn clear_processed_frequencies() {
    let mut processed = PROCESSED_FREQUENCIES.lock().unwrap();
    let count = processed.len();
    processed.clear();
    debug!(
        cleared_count = count,
        "Cleared processed frequencies for new scanning session"
    );
}

pub mod deemph;
pub mod filter_config;
pub mod pipeline_builder;
pub mod squelch;

use filter_config::FilterPurpose;

#[derive(Debug, Clone)]
pub struct Candidate {
    pub frequency_hz: f64,
    pub peak_count: usize,
    pub max_magnitude: f32,
    pub avg_magnitude: f32,
    pub signal_strength: String,
}

impl Candidate {
    pub fn analyze(
        &self,
        config: &ScanningConfig,
        sdr_rx: tokio::sync::broadcast::Receiver<rustradio::Complex>,
        center_freq: f64,
        signal_tx: std::sync::mpsc::SyncSender<crate::types::Signal>,
        device: &crate::soapy::Device,
    ) -> Result<()> {
        debug!(
            message = "Tuning into station",
            fequency_hz = self.frequency_hz / 1e6,
            signal_strength = self.signal_strength,
            peak_count = self.peak_count,
            self.max_magnitude = self.max_magnitude,
            self.avg_magnitude = self.avg_magnitude
        );

        // Frequency tracking: refine the FFT-detected frequency
        let refined_frequency = if config.disable_frequency_tracking {
            debug!(
                freq_mhz = self.frequency_hz / 1e6,
                "Frequency tracking disabled, using FFT estimate"
            );
            // Round to nearest 100 kHz to avoid floating point errors
            (self.frequency_hz / 100000.0).round() * 100000.0
        } else {
            match self.run_frequency_tracking(config, sdr_rx.resubscribe()) {
                Some(freq) => {
                    // Round to nearest 100 kHz to avoid floating point errors
                    let rounded_freq = (freq / 100000.0).round() * 100000.0;
                    debug!(
                        original_mhz = self.frequency_hz / 1e6,
                        refined_mhz = freq / 1e6,
                        rounded_mhz = rounded_freq / 1e6,
                        error_khz = (freq - self.frequency_hz) / 1e3,
                        "Frequency tracking successful, using rounded frequency"
                    );
                    rounded_freq
                }
                None => {
                    debug!(
                        freq_mhz = self.frequency_hz / 1e6,
                        "Frequency tracking failed, using FFT estimate"
                    );
                    // Round to nearest 100 kHz to avoid floating point errors
                    (self.frequency_hz / 100000.0).round() * 100000.0
                }
            }
        };

        // Convert to kHz for deduplication (frequency is already rounded to nearest 100 kHz)
        let frequency_khz = (refined_frequency / 1000.0) as u64;

        // Check if we've already processed this frequency
        {
            let processed = PROCESSED_FREQUENCIES.lock().unwrap();
            if processed.contains(&frequency_khz) {
                debug!(
                    refined_freq_mhz = refined_frequency / 1e6,
                    frequency_khz = frequency_khz,
                    "Frequency already processed in another window, skipping analysis"
                );
                return Ok(());
            }
            // Don't mark as processed yet - only mark when squelch analysis succeeds
        }

        debug!(
            refined_freq_mhz = refined_frequency / 1e6,
            frequency_khz = frequency_khz,
            "New frequency detected, proceeding with analysis"
        );

        let station_name = format!("{:.1}FM", refined_frequency / 1e6);

        // Create DSP graph to process this candidate's refined frequency
        // Use the provided center_freq and the refined tune frequency for the FreqXlatingFir
        let (mut detection_graph, decision_state) = create_detection_graph(
            sdr_rx,
            config.samp_rate,
            station_name,
            config,
            center_freq,
            refined_frequency,
            Some(signal_tx.clone()), // Pass signal channel to squelch
            device,
        )?;
        let detection_cancel_token = detection_graph.cancel_token();

        debug!(
            "Processing candidate at {:.1} MHz with center freq {:.1} MHz",
            self.frequency_hz / 1e6,
            center_freq / 1e6
        );

        // Start DSP graph in a separate thread
        let frequency_hz = self.frequency_hz;
        let detection_graph_handle = thread::spawn(move || {
            debug!("Detection graph started for {:.1} MHz", frequency_hz / 1e6);
            if let Err(e) = detection_graph.run() {
                debug!("Detection graph error for {}: {}", frequency_hz / 1e6, e);
            }
            debug!(
                "Detection graph terminated for {:.1} MHz",
                frequency_hz / 1e6
            );
        });

        // Set up timer thread with squelch monitoring
        // Detection runs until squelch makes a decision, not limited by --duration
        let squelch_learning_duration = config.squelch_learning_duration;
        let decision_state_clone = decision_state.clone();
        let frequency_khz_for_cleanup = frequency_khz; // Capture for potential cleanup
        let timer_handle = thread::spawn(move || {
            let check_interval = Duration::from_millis(100);
            // Add extra time beyond learning duration to account for processing delays
            let max_wait_time = squelch_learning_duration + 1.0; // Extra second for processing
            let total_checks = (max_wait_time * 1000.0) as u32 / 100;

            let message_printed = false;

            for check_num in 0..total_checks {
                thread::sleep(check_interval);

                // Check squelch decision
                let current_decision = crate::fm::squelch::Decision::from_u8(
                    decision_state_clone.load(std::sync::atomic::Ordering::Relaxed),
                );

                match current_decision {
                    crate::fm::squelch::Decision::Noise => {
                        debug!(
                            "Squelch detected noise, exiting early (frequency available for retry in other windows)"
                        );
                        detection_cancel_token.cancel();
                        return;
                    }
                    crate::fm::squelch::Decision::Audio => {
                        if !message_printed {
                            debug!("squelch detected audio at {:.1} MHz", frequency_hz / 1e6);

                            // Mark frequency as successfully processed to prevent retry in other windows
                            {
                                let mut processed = PROCESSED_FREQUENCIES.lock().unwrap();
                                processed.insert(frequency_khz_for_cleanup);
                                debug!(
                                    frequency_khz = frequency_khz_for_cleanup,
                                    "Frequency marked as successfully processed"
                                );
                            }

                            // OPTIMIZATION: Stop detection graph immediately when audio starts
                            // This frees up significant CPU resources for audio processing
                            debug!(
                                "Audio detected, terminating detection graph to optimize audio quality"
                            );
                            detection_cancel_token.cancel();
                            return; // Exit monitoring thread - audio processing takes over
                        }
                    }
                    crate::fm::squelch::Decision::Learning => {
                        // Still learning, continue waiting
                    }
                }

                // Log progress for debugging
                if check_num == (squelch_learning_duration * 10.0) as u32 {
                    debug!("Squelch learning period complete, waiting for decision");
                }
            }

            // If we get here, squelch didn't make a decision in time
            debug!(
                "Squelch did not complete analysis after {:.1} seconds, moving to next candidate",
                max_wait_time
            );
            detection_cancel_token.cancel();
        });

        // Wait for both detection graph and timer threads to complete
        // This ensures the candidate analysis doesn't return until the squelch decision is made
        debug!(
            "Waiting for detection graph and timer threads to complete for {:.1} MHz",
            self.frequency_hz / 1e6
        );

        if let Err(e) = timer_handle.join() {
            debug!(
                "Timer thread panicked for {:.1} MHz: {:?}",
                self.frequency_hz / 1e6,
                e
            );
        }

        if let Err(e) = detection_graph_handle.join() {
            debug!(
                "Detection graph thread panicked for {:.1} MHz: {:?}",
                self.frequency_hz / 1e6,
                e
            );
        }

        debug!(
            "All detection threads completed for {:.1} MHz",
            self.frequency_hz / 1e6
        );
        Ok(())
    }

    /// Run frequency tracking to refine the FFT-based frequency estimate
    fn run_frequency_tracking(
        &self,
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
                debug!(error = %e, "Invalid frequency tracking method, falling back to PLL");
                TrackingMethod::Pll
            }
        };

        // Create tracking configuration
        let tracking_config = TrackingConfig {
            method: tracking_method,
            convergence_threshold: config.tracking_accuracy,
            // Set reasonable timeout based on squelch learning duration
            timeout_samples: (config.samp_rate * config.squelch_learning_duration as f64 * 0.5)
                as usize,
            search_window: 200_000.0, // ±200 kHz search window for FM
            min_samples_for_convergence: (config.samp_rate * 0.01) as usize, // 10ms minimum
        };

        // Create frequency tracker
        let mut tracker = create_tracker(
            tracking_method,
            self.frequency_hz,
            config.samp_rate,
            &tracking_config,
        );

        debug!(
            initial_freq_mhz = self.frequency_hz / 1e6,
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
                            debug!(
                                refined_freq_mhz = freq / 1e6,
                                confidence = tracker.get_confidence(),
                                "Frequency tracking converged"
                            );
                            return Some(freq);
                        }
                        TrackingState::Failed => {
                            debug!("Frequency tracking failed to converge");
                            return None;
                        }
                        TrackingState::Timeout => {
                            debug!("Frequency tracking timed out");
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
                    debug!("Frequency tracking lagged behind SDR stream");
                    // Continue with next sample
                }
                Err(tokio::sync::broadcast::error::TryRecvError::Closed) => {
                    debug!("SDR stream closed during frequency tracking");
                    return None;
                }
            }
        }
    }
}

/// Collect RF peaks by consuming from a broadcast channel.
/// This performs FFT analysis to detect spectral peaks above the threshold.
#[allow(clippy::type_complexity)]
fn setup_peak_collection(
    config: &ScanningConfig,
) -> (
    Vec<rustfft::num_complex::Complex32>,
    rustfft::FftPlanner<f32>,
    std::sync::Arc<dyn rustfft::Fft<f32>>,
    Vec<Complex>,
) {
    let fft_buffer = vec![rustfft::num_complex::Complex32::default(); config.fft_size];
    let mut planner = rustfft::FftPlanner::new();
    let fft = planner.plan_fft_forward(config.fft_size);
    let read_buffer = vec![Complex::default(); config.fft_size];
    (fft_buffer, planner, fft, read_buffer)
}

fn run_agc_settling_phase(
    sdr_rx: &mut tokio::sync::broadcast::Receiver<Complex>,
    agc_samples_needed: usize,
    start_time: std::time::Instant,
    total_duration: f64,
) -> usize {
    debug!("Phase 1: AGC settling phase - consuming samples without analysis");
    let mut agc_samples_consumed = 0;

    while agc_samples_consumed < agc_samples_needed {
        if start_time.elapsed().as_secs_f64() > total_duration + 1.0 {
            debug!("AGC settling phase timed out");
            break;
        }

        match sdr_rx.try_recv() {
            Ok(_sample) => {
                agc_samples_consumed += 1;
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => {
                debug!("SDR samples lagged during AGC settling phase");
                continue;
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Closed) => {
                debug!("SDR channel closed during AGC settling");
                break;
            }
        }
    }

    debug!(
        agc_samples_consumed = agc_samples_consumed,
        agc_duration_actual = start_time.elapsed().as_secs_f64(),
        "AGC settling phase complete"
    );

    agc_samples_consumed
}

#[allow(clippy::too_many_arguments)]
fn run_peak_detection_phase(
    sdr_rx: &mut tokio::sync::broadcast::Receiver<Complex>,
    peak_samples_needed: usize,
    mut read_buffer: Vec<Complex>,
    mut fft_buffer: Vec<rustfft::num_complex::Complex32>,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    config: &ScanningConfig,
    center_freq: f64,
    start_time: std::time::Instant,
    total_duration: f64,
) -> std::collections::BTreeMap<u64, Peak> {
    use std::collections::BTreeMap;

    debug!("Phase 2: Peak detection phase - analyzing samples for spectral peaks");
    let mut peaks_map: BTreeMap<u64, Peak> = BTreeMap::new();
    let mut peak_samples_collected = 0;

    while peak_samples_collected < peak_samples_needed {
        if start_time.elapsed().as_secs_f64() > total_duration + 1.0 {
            debug!("Peak detection phase timed out");
            break;
        }

        match sdr_rx.try_recv() {
            Ok(sample) => {
                read_buffer[peak_samples_collected % config.fft_size] = sample;
                peak_samples_collected += 1;

                if peak_samples_collected % config.fft_size == 0 {
                    let batch_peaks = process_samples_for_peaks(
                        &read_buffer,
                        config.fft_size,
                        &mut fft_buffer,
                        &fft,
                        config,
                        center_freq,
                    );

                    for peak in batch_peaks {
                        let rounded_freq = (peak.frequency_hz / 100000.0).round() as u64;
                        peaks_map
                            .entry(rounded_freq)
                            .and_modify(|e| {
                                if peak.magnitude > e.magnitude {
                                    *e = peak.clone();
                                }
                            })
                            .or_insert(peak);
                    }
                }
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => {
                thread::sleep(Duration::from_micros(100));
                continue;
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => {
                debug!("Peak collection lagged behind SDR stream");
                continue;
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Closed) => {
                debug!("SDR broadcast channel closed during peak collection");
                break;
            }
        }
    }

    debug!(
        peak_samples_collected = peak_samples_collected,
        total_duration_actual = start_time.elapsed().as_secs_f64(),
        "Peak detection phase complete"
    );

    peaks_map
}

pub fn collect_peaks(
    config: &ScanningConfig,
    mut sdr_rx: tokio::sync::broadcast::Receiver<Complex>,
    center_freq: f64,
) -> Result<Vec<Peak>> {
    let agc_settling_time = config.agc_settling_time;
    let peak_scan_duration = config.peak_scan_duration.unwrap_or(0.5);
    let total_duration = agc_settling_time + peak_scan_duration;

    debug!(
        agc_settling_seconds = agc_settling_time,
        peak_scan_seconds = peak_scan_duration,
        total_seconds = total_duration,
        "Starting AGC settling followed by peak detection scan"
    );

    let samples_per_second = config.samp_rate as usize;
    let agc_samples_needed = (samples_per_second as f64 * agc_settling_time) as usize;
    let peak_samples_needed = (samples_per_second as f64 * peak_scan_duration) as usize;
    let start_time = std::time::Instant::now();

    debug!(
        agc_samples = agc_samples_needed,
        peak_samples = peak_samples_needed,
        "Starting two-phase collection: AGC settling then peak detection"
    );

    let (fft_buffer, _planner, fft, read_buffer) = setup_peak_collection(config);
    let _agc_samples_consumed =
        run_agc_settling_phase(&mut sdr_rx, agc_samples_needed, start_time, total_duration);

    let peaks_map = run_peak_detection_phase(
        &mut sdr_rx,
        peak_samples_needed,
        read_buffer,
        fft_buffer,
        fft,
        config,
        center_freq,
        start_time,
        total_duration,
    );

    let peaks: Vec<Peak> = peaks_map.into_values().collect();
    debug!("Peak detection scan complete. Found {} peaks.", peaks.len());
    Ok(peaks)
}

/// Extract peaks from FFT magnitudes
fn extract_peaks_from_magnitudes(
    magnitudes: &[f32],
    threshold: f32,
    fft_size: usize,
    sample_rate: f64,
    center_freq: f64,
) -> Vec<Peak> {
    let mut peaks = Vec::new();

    // Detect peaks: local maxima above threshold
    for i in 1..magnitudes.len() - 1 {
        if magnitudes[i] > threshold
            && magnitudes[i] > magnitudes[i - 1]
            && magnitudes[i] > magnitudes[i + 1]
        {
            // Convert FFT bin to frequency
            let freq_offset = (i as f64 / fft_size as f64) * sample_rate;
            let freq_hz = center_freq - (sample_rate / 2.0) + freq_offset;

            // Round to nearest 100 kHz to eliminate floating point precision errors
            let freq_hz_rounded = (freq_hz / 100000.0).round() * 100000.0;

            peaks.push(Peak {
                frequency_hz: freq_hz_rounded,
                magnitude: magnitudes[i],
            });
        }
    }

    peaks
}

/// Process a batch of samples through FFT and extract peaks
fn process_samples_for_peaks(
    read_buffer: &[Complex],
    samples_read: usize,
    fft_buffer: &mut [rustfft::num_complex::Complex32],
    fft: &std::sync::Arc<dyn rustfft::Fft<f32>>,
    config: &ScanningConfig,
    center_freq: f64,
) -> Vec<Peak> {
    // Copy samples to FFT buffer
    for (i, sample) in read_buffer
        .iter()
        .take(samples_read.min(config.fft_size))
        .enumerate()
    {
        fft_buffer[i] = rustfft::num_complex::Complex32::new(sample.re, sample.im);
    }

    fft.process(fft_buffer);
    let magnitudes: Vec<f32> = fft_buffer.iter().map(|c| c.norm_sqr()).collect();

    extract_peaks_from_magnitudes(
        &magnitudes,
        config.peak_detection_threshold,
        config.fft_size,
        config.samp_rate,
        center_freq,
    )
}

/// Collect RF peaks from any SampleSource (for testing and production)
pub fn collect_peaks_from_source(
    config: &ScanningConfig,
    sample_source: &mut dyn testing::SampleSource,
) -> Result<Vec<Peak>> {
    let peak_scan_duration = sample_source.peak_scan_duration();
    debug!("Starting peak detection scan for {peak_scan_duration} seconds...",);

    // Prepare FFT processing
    use std::collections::BTreeMap;
    let mut peaks_map: BTreeMap<u64, Peak> = BTreeMap::new();
    let mut fft_buffer = vec![rustfft::num_complex::Complex32::default(); config.fft_size];
    let mut planner = rustfft::FftPlanner::new();
    let fft = planner.plan_fft_forward(config.fft_size);

    // Calculate sampling parameters
    let samples_per_second = sample_source.sample_rate() as usize;
    let total_samples_needed = (samples_per_second as f64 * peak_scan_duration) as usize;
    let mut samples_collected = 0;
    let mut read_buffer = vec![Complex::default(); config.fft_size];

    // Collect samples and perform peak detection
    while samples_collected < total_samples_needed {
        match sample_source.read_samples(&mut read_buffer) {
            Ok(samples_read) => {
                if samples_read == 0 {
                    break; // End of file reached
                }

                let batch_peaks = process_samples_for_peaks(
                    &read_buffer,
                    samples_read,
                    &mut fft_buffer,
                    &fft,
                    config,
                    sample_source.center_frequency(),
                );
                for peak in batch_peaks {
                    let rounded_freq = (peak.frequency_hz / 100000.0).round() as u64;
                    peaks_map
                        .entry(rounded_freq)
                        .and_modify(|e| {
                            if peak.magnitude > e.magnitude {
                                *e = peak.clone();
                            }
                        })
                        .or_insert(peak);
                }

                samples_collected += samples_read;
            }
            Err(e) => {
                debug!("Error reading from SDR: {}", e);
                break;
            }
        }
    }
    let peaks: Vec<Peak> = peaks_map.into_values().collect();

    sample_source.deactivate()?;
    debug!("Peak detection scan complete. Found {} peaks.", peaks.len());

    Ok(peaks)
}

/// Analyze spectral characteristics around a frequency to determine if it's a main lobe or sidelobe
/// Main lobes are wider and have characteristic spectral patterns compared to sidelobes
fn analyze_spectral_characteristics(
    peaks: &[Peak],
    target_freq_mhz: f64,
    _sample_rate: f64,
    center_freq: f64,
) -> (f32, String) {
    let target_freq_hz = target_freq_mhz * 1e6;

    // Find peaks within ±200 kHz of target frequency (wider than FM channel spacing)
    let analysis_range_hz = 200000.0;
    let nearby_peaks: Vec<&Peak> = peaks
        .iter()
        .filter(|peak| (peak.frequency_hz - target_freq_hz).abs() <= analysis_range_hz)
        .collect();

    // Debug logging for 88.9 MHz specifically
    if (target_freq_mhz - 88.9).abs() < 0.01 {
        debug!(
            "88.9 MHz analysis: found {} peaks within ±200kHz range",
            nearby_peaks.len()
        );
        for (i, peak) in nearby_peaks.iter().take(5).enumerate() {
            debug!(
                "  Peak {}: {:.3} MHz, magnitude {:.3}, offset {:.1} kHz",
                i + 1,
                peak.frequency_hz / 1e6,
                peak.magnitude,
                (peak.frequency_hz - target_freq_hz) / 1e3
            );
        }
    }

    if nearby_peaks.is_empty() {
        return (0.0, "No signal".to_string());
    }

    // Sort peaks by frequency for width analysis
    let mut sorted_peaks = nearby_peaks.clone();
    sorted_peaks.sort_by(|a, b| a.frequency_hz.partial_cmp(&b.frequency_hz).unwrap());

    // Calculate spectral width characteristics
    let peak_count = sorted_peaks.len();
    let freq_span_khz = if peak_count > 1 {
        (sorted_peaks.last().unwrap().frequency_hz - sorted_peaks.first().unwrap().frequency_hz)
            / 1000.0
    } else {
        0.0
    };

    // Find the strongest peak in the group (should be the main signal)
    let max_magnitude = sorted_peaks
        .iter()
        .map(|p| p.magnitude)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    // Calculate average magnitude
    let avg_magnitude = sorted_peaks.iter().map(|p| p.magnitude).sum::<f32>() / peak_count as f32;

    // Main lobe characteristics analysis
    let mut score: f64 = 0.0;
    let mut analysis_notes = Vec::new();

    // 1. Peak density analysis (main lobes have consistent energy distribution)
    let peak_density = peak_count as f64 / freq_span_khz.max(1.0);
    if peak_density > 20.0 && peak_density < 200.0 {
        score += 0.3;
        analysis_notes.push("Good peak density");
    } else if peak_density > 200.0 {
        score -= 0.2; // Too many peaks suggests broadband interference
        analysis_notes.push("High peak density (interference?)");
    }

    // 2. Frequency span analysis (main lobes have characteristic widths)
    // FM broadcast stations typically show energy across 150-200 kHz
    if freq_span_khz > 80.0 && freq_span_khz < 250.0 {
        score += 0.3;
        analysis_notes.push("Appropriate spectral width");
    } else if freq_span_khz < 15.0 {
        score -= 0.3; // Too narrow suggests sidelobe
        analysis_notes.push("Narrow spectral width (sidelobe?)");
    }

    // 3. Signal strength and consistency
    let magnitude_ratio = max_magnitude / avg_magnitude.max(1.0);
    if magnitude_ratio < 3.0 {
        score += 0.2; // Consistent energy suggests main lobe
        analysis_notes.push("Consistent energy");
    } else if magnitude_ratio > 10.0 {
        score -= 0.1; // Single spike suggests sidelobe
        analysis_notes.push("Sharp peak (possible sidelobe)");
    }

    // 4. Distance from center frequency (closer = more likely to be legitimate)
    let center_freq_mhz = center_freq / 1e6;
    let dist_from_center_mhz = (target_freq_mhz - center_freq_mhz).abs();
    if dist_from_center_mhz <= 0.1 {
        score += 0.4; // Strong bonus for center frequency
        analysis_notes.push("Near center freq");
    } else if dist_from_center_mhz <= 0.3 {
        score += 0.1; // Moderate bonus for nearby frequencies
    } else if dist_from_center_mhz > 0.4 {
        score -= 0.2; // Penalty for distant frequencies
        analysis_notes.push("Far from center");
    }

    // 5. Absolute signal strength
    if max_magnitude > 500.0 {
        score += 0.2;
        analysis_notes.push("Strong signal");
    } else if max_magnitude < 100.0 {
        score -= 0.1;
        analysis_notes.push("Weak signal");
    }

    let analysis_summary = analysis_notes.join(", ");

    // Additional debug for 88.9 MHz
    if (target_freq_mhz - 88.9).abs() < 0.01 {
        debug!(
            "88.9 MHz detailed analysis: peak_count={}, freq_span_khz={:.1}, max_mag={:.3}, avg_mag={:.3}, mag_ratio={:.2}, peak_density={:.1}, final_score={:.3}",
            peak_count,
            freq_span_khz,
            max_magnitude,
            avg_magnitude,
            magnitude_ratio,
            peak_density,
            score
        );
    }

    (score.clamp(0.0, 1.0) as f32, analysis_summary)
}

/// Create a candidate from peak analysis results
fn create_fm_candidate(
    frequency_mhz: f64,
    peaks: &[Peak],
    spectral_score: f32,
) -> types::Candidate {
    let signal_strength = if spectral_score > 0.8 {
        "Strong"
    } else if spectral_score > 0.6 {
        "Medium"
    } else {
        "Weak"
    };

    // Find relevant peaks for this frequency
    let tolerance_mhz = 0.1;
    let nearby_peaks: Vec<&Peak> = peaks
        .iter()
        .filter(|peak| {
            let peak_freq_mhz = peak.frequency_hz / 1e6;
            (peak_freq_mhz - frequency_mhz).abs() <= tolerance_mhz
        })
        .collect();

    let peak_count = nearby_peaks.len();
    let max_magnitude = nearby_peaks
        .iter()
        .map(|p| p.magnitude)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);
    let avg_magnitude = if peak_count > 0 {
        nearby_peaks.iter().map(|p| p.magnitude).sum::<f32>() / peak_count as f32
    } else {
        0.0
    };

    types::Candidate::Fm(Candidate {
        frequency_hz: frequency_mhz * 1e6,
        peak_count,
        max_magnitude,
        avg_magnitude,
        signal_strength: signal_strength.to_string(),
    })
}

/// Generate the next FM frequency (odd tenth increments)
fn next_fm_frequency(current_freq_mhz: f64) -> f64 {
    (current_freq_mhz * 10.0 + 2.0) / 10.0 // Next odd tenth (add 0.2)
}

/// Calculate starting FM frequency for the scan range
fn calculate_starting_fm_frequency(freq_start_mhz: f64) -> f64 {
    let mut fm_freq = (freq_start_mhz * 10.0).ceil() / 10.0;
    if (fm_freq * 10.0) as i32 % 2 == 0 {
        fm_freq += 0.1; // Make it an odd tenth
    }
    fm_freq
}

/// Detect FM radio stations using spectral analysis with main lobe vs sidelobe discrimination
/// This approach analyzes spectral characteristics like peak width, density, and shape
pub fn find_candidates(
    peaks: &[Peak],
    config: &ScanningConfig,
    center_freq: f64,
) -> Vec<types::Candidate> {
    debug!("Using spectral analysis for FM station detection with sidelobe discrimination...");

    // Calculate the frequency range we scanned based on center freq and sample rate
    let scan_range_mhz = config.samp_rate / 2e6; // Half sample rate in MHz (Nyquist)
    let freq_start_mhz = (center_freq / 1e6) - scan_range_mhz;
    let freq_end_mhz = (center_freq / 1e6) + scan_range_mhz;

    debug!(
        "Analyzing spectral patterns in range: {:.1} - {:.1} MHz",
        freq_start_mhz, freq_end_mhz
    );

    let mut candidates = Vec::new();
    let mut fm_freq = calculate_starting_fm_frequency(freq_start_mhz);

    while fm_freq <= freq_end_mhz {
        debug!("Analyzing {:.1} MHz... ", fm_freq);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let (spectral_score, analysis_summary) =
            analyze_spectral_characteristics(peaks, fm_freq, config.samp_rate, center_freq);

        debug!("score: {:.3} ({})", spectral_score, analysis_summary);

        // Only consider frequencies with significant spectral score
        if spectral_score >= config.spectral_threshold {
            candidates.push(create_fm_candidate(fm_freq, peaks, spectral_score));
        }

        fm_freq = next_fm_frequency(fm_freq);
    }

    candidates
}

// Create rustradio detection graph for signal analysis with frequency translating filter
#[allow(clippy::too_many_arguments)]
pub fn create_detection_graph(
    source_receiver: tokio::sync::broadcast::Receiver<rustradio::Complex>,
    samp_rate: f64,
    _channel_name: String,
    config: &ScanningConfig,
    center_freq: f64,
    tune_freq: f64,
    signal_tx: Option<std::sync::mpsc::SyncSender<crate::types::Signal>>,
    _device: &crate::soapy::Device,
) -> rustradio::Result<(Graph, std::sync::Arc<std::sync::atomic::AtomicU8>)> {
    let mut graph = Graph::new();

    // MPSC Source
    let (source_block, prev) = crate::broadcast::BroadcastSource::new(source_receiver);
    graph.add(Box::new(source_block));

    // Calculate frequency offset for translating filter
    let frequency_offset = tune_freq - center_freq;

    // Create frequency xlating filter using shared pipeline builder
    let (prev, decimation) = pipeline_builder::FmPipelineBuilder::create_frequency_xlating_filter(
        prev,
        &mut graph,
        frequency_offset,
        config,
        FilterPurpose::Audio,
    )?;

    // Update effective sample rate after decimation
    let decimated_samp_rate = samp_rate / decimation as f64;

    // Add IF AGC using shared pipeline builder
    let prev =
        pipeline_builder::FmPipelineBuilder::create_if_agc(prev, &mut graph, config, "detection");

    // Skip additional resampling if we're already close to desired quad rate
    let quad_rate = decimated_samp_rate as f32; // Use decimated rate directly to avoid extra resampling

    // Quadrature demodulation with reduced gain to prevent distortion
    // FM deviation for broadcast is 75kHz, so gain should account for sample rate
    let fm_gain = (quad_rate / (2.0 * 75_000.0)) * 0.8; // 0.8 factor to prevent overload
    let prev = blockchain![graph, prev, QuadratureDemod::new(prev, fm_gain)];

    // Add proper FM deemphasis to match audio pipeline processing
    // This ensures both pipelines process the FM signal identically
    let (deemphasis_block, prev) = crate::fm::deemph::Deemphasis::new(prev, quad_rate, 75.0);
    graph.add(Box::new(deemphasis_block));

    // Add audio decimation chain using shared pipeline builder
    let prev = pipeline_builder::FmPipelineBuilder::create_audio_decimation_chain(
        prev,
        &mut graph,
        quad_rate,
        config,
        "Detection",
    )?;

    // Use actual resampled rate for squelch analysis
    let analysis_rate = config.audio_sample_rate as f32; // Now matches audio pipeline exactly

    // Audio capture block (captures samples while passing them through)
    // Create audio capturer if requested - needed for test fixture generation
    let audio_capturer = if let Some(ref capture_dir) = config.capture_audio {
        let audio_config = crate::file::AudioCaptureConfig {
            output_dir: capture_dir.clone(),
            sample_rate: analysis_rate,
            capture_duration: config.capture_audio_duration,
            frequency_hz: tune_freq,
            modulation_type: crate::types::ModulationType::WFM,
        };
        match crate::file::AudioCaptureSink::new(audio_config) {
            Ok(capturer) => Some(capturer),
            Err(e) => {
                debug!("Failed to create audio capturer: {}", e);
                None
            }
        }
    } else {
        None
    };

    let (audio_capture_block, audio_capture_output) = AudioCaptureBlock::new(prev, audio_capturer);
    graph.add(Box::new(audio_capture_block));
    let prev = audio_capture_output;

    use crate::fm::squelch::SquelchConfig;
    let squelch_config = SquelchConfig {
        sample_rate: analysis_rate, // Use current rate instead of resampled audio rate
        learning_duration: config.squelch_learning_duration,
        signal_tx,
        frequency_hz: tune_freq,
        center_freq,
        squelch_disabled: config.disable_squelch,
        fft_size: config.fft_size,
    };
    let (squelch_block, decision_state) = SquelchBlock::new(prev, squelch_config);
    graph.add(Box::new(squelch_block));

    Ok((graph, decision_state))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::*;

    #[test]
    fn test_frequency_translation_scenarios() {
        let scenarios = create_frequency_test_scenarios();

        println!("\n=== Frequency Translation Analysis ===");
        for scenario in scenarios {
            let offset = scenario.simulate_frequency_translation();

            // Check if the offset is within the usable bandwidth of our FreqXlatingFir
            // Our filter has 150 kHz bandwidth, so ±75 kHz is the limit
            let max_offset = 75_000.0; // 75 kHz

            if offset.abs() > max_offset {
                println!(
                    "❌ [{}] Offset {:.1} kHz exceeds filter bandwidth (±75 kHz)!",
                    scenario.test_name,
                    offset / 1e3
                );
            } else {
                println!(
                    "✅ [{}] Offset {:.1} kHz is within filter bandwidth",
                    scenario.test_name,
                    offset / 1e3
                );
            }
        }
    }

    #[test]
    fn test_band_scanning_windows() {
        use types::Band;

        let config = ScanningConfig::default();
        let band = Band::Fm;
        let windows = band.windows(config.samp_rate, config.window_overlap);

        println!("\n=== FM Band Window Analysis ===");
        println!("Sample rate: {} MHz", config.samp_rate / 1e6);
        println!("Number of windows: {}", windows.len());

        let target_station = 88.9e6;

        for (i, window_center) in windows.iter().enumerate() {
            let window_start = window_center - (config.samp_rate * 0.8 / 2.0);
            let window_end = window_center + (config.samp_rate * 0.8 / 2.0);

            // Check if our target station falls within this window
            if target_station >= window_start && target_station <= window_end {
                let offset = target_station - window_center;
                println!(
                    "🎯 Window {}: Center {:.1} MHz covers 88.9 MHz (offset: {:.1} kHz)",
                    i + 1,
                    window_center / 1e6,
                    offset / 1e3
                );

                // This is the problematic scenario
                if offset.abs() > 75_000.0 {
                    println!("⚠️  This offset exceeds our filter bandwidth!");
                }
            }
        }
    }

    #[test]
    fn test_collect_peaks_from_mock_source() {
        let config = ScanningConfig {
            audio_buffer_size: 4096,
            audio_sample_rate: 48000,
            band: crate::types::Band::Fm,
            capture_audio_duration: 3.0,
            capture_audio: None,
            capture_duration: 2.0,
            capture_iq: None,
            debug_pipeline: false,
            duration: 1,
            sdr_gain: 24.0,
            scanning_windows: None,
            fft_size: 1024,
            peak_detection_threshold: 0.01, // Low threshold for testing
            peak_scan_duration: Some(0.1),  // Short duration for testing
            print_candidates: false,
            samp_rate: 1000000.0,
            squelch_learning_duration: 2.0,

            // Frequency tracking configuration
            frequency_tracking_method: "pll".to_string(),
            tracking_accuracy: 5000.0,
            disable_frequency_tracking: true, // Disable for test to keep existing behavior

            // Spectral analysis configuration
            spectral_threshold: 0.2,

            // AGC and window configuration
            agc_settling_time: 3.0,
            window_overlap: 0.75,
            // Squelch configuration
            disable_squelch: false,
            // IF AGC configuration
            disable_if_agc: false,
        };

        // Create mock source with a signal at +100kHz offset from center
        let mut mock_source = MockSampleSource::new(
            1000000.0,  // 1 MHz sample rate
            88900000.0, // 88.9 MHz center frequency
            100000,     // 100k samples max
            100000.0,   // 100 kHz offset signal
        );

        let peaks = collect_peaks_from_source(&config, &mut mock_source).unwrap();

        // Should detect the peak around 89.0 MHz (88.9 + 0.1)
        assert!(!peaks.is_empty(), "Should detect at least one peak");

        let target_freq = 89000000.0; // 89.0 MHz
        let found_peak = peaks
            .iter()
            .find(|p| (p.frequency_hz - target_freq).abs() < 50000.0);
        assert!(
            found_peak.is_some(),
            "Should find peak near 89.0 MHz, found peaks at: {:?}",
            peaks
                .iter()
                .map(|p| p.frequency_hz / 1e6)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_sidelobe_discrimination_rejects_legitimate_fm_signal() {
        // This test reproduces the 88.9 MHz detection failure
        // The signal has 20 kHz spectral width but gets rejected by overly strict thresholds

        // Create synthetic peaks representing a legitimate FM signal with 20 kHz span
        // This matches the real 88.9 MHz signal characteristics from Window 2
        let peaks = vec![
            Peak {
                frequency_hz: 88_700_000.0,
                magnitude: 7.672,
            }, // -200 kHz offset
            Peak {
                frequency_hz: 88_702_000.0,
                magnitude: 7.749,
            }, // -198 kHz offset
            Peak {
                frequency_hz: 88_704_000.0,
                magnitude: 7.496,
            }, // -196 kHz offset
            Peak {
                frequency_hz: 88_706_000.0,
                magnitude: 7.334,
            }, // -194 kHz offset
            Peak {
                frequency_hz: 88_708_000.0,
                magnitude: 4.706,
            }, // -192 kHz offset
            Peak {
                frequency_hz: 88_710_000.0,
                magnitude: 6.123,
            }, // -190 kHz offset
            Peak {
                frequency_hz: 88_712_000.0,
                magnitude: 5.892,
            }, // -188 kHz offset
            Peak {
                frequency_hz: 88_714_000.0,
                magnitude: 5.234,
            }, // -186 kHz offset
            Peak {
                frequency_hz: 88_716_000.0,
                magnitude: 4.987,
            }, // -184 kHz offset
            Peak {
                frequency_hz: 88_718_000.0,
                magnitude: 4.123,
            }, // -182 kHz offset
            Peak {
                frequency_hz: 88_720_000.0,
                magnitude: 3.856,
            }, // -180 kHz offset
        ];

        let target_freq_mhz = 88.9;
        let sample_rate = 2_000_000.0;
        let center_freq = 89.2e6; // Window 2 center frequency

        // Call the function that's failing
        let (score, analysis_summary) =
            analyze_spectral_characteristics(&peaks, target_freq_mhz, sample_rate, center_freq);

        // After fixing the threshold from 30 kHz to 15 kHz, the algorithm should accept this signal
        // freq_span = 88.720 - 88.700 = 20 kHz (now above the 15 kHz threshold)
        assert!(
            score > 0.0,
            "Fixed algorithm should accept legitimate FM signal with 20 kHz span. Score: {:.3}, Analysis: '{}'",
            score,
            analysis_summary
        );
        assert!(
            !analysis_summary.contains("Narrow spectral width (sidelobe?)"),
            "Should not classify 20 kHz span as narrow/sidelobe. Analysis: '{}'",
            analysis_summary
        );
    }

    #[test]
    fn test_frequency_rounding_100khz() {
        // Test that frequencies are rounded to nearest 100 kHz
        let test_cases = vec![
            // (input_hz, expected_hz)
            (87_700_000.0, 87_700_000.0), // Exact 100 kHz boundary
            (87_749_999.0, 87_700_000.0), // Just under 50 kHz threshold - round down
            (87_750_000.0, 87_800_000.0), // Exactly 50 kHz - round up
            (87_750_001.0, 87_800_000.0), // Just over 50 kHz threshold - round up
            (87_799_999.0, 87_800_000.0), // Just under next boundary - round up
            (87_800_000.0, 87_800_000.0), // Exact 100 kHz boundary
            (93_125_000.0, 93_100_000.0), // 93.125 MHz -> 93.1 MHz
            (93_175_000.0, 93_200_000.0), // 93.175 MHz -> 93.2 MHz
            (93_149_999.0, 93_100_000.0), // Just under 50 kHz threshold
            (93_150_000.0, 93_200_000.0), // Exactly 50 kHz threshold
        ];

        for (input_hz, expected_hz) in test_cases {
            let rounded = (input_hz / 100000.0f64).round() * 100000.0f64;
            assert_eq!(
                rounded, expected_hz,
                "Failed rounding {:.0} Hz to nearest 100 kHz. Expected {:.0}, got {:.0}",
                input_hz, expected_hz, rounded
            );

            // Verify the rounding is actually 100 kHz aligned
            assert_eq!(
                (rounded as u64) % 100_000,
                0,
                "Rounded frequency {:.0} Hz is not aligned to 100 kHz boundary",
                rounded
            );
        }
    }
}
