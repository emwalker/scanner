use crate::{
    file::AudioCaptureBlock,
    fm::{deemph::Deemphasis, squelch::SquelchBlock},
    freq_xlating_fir::FreqXlatingFir,
    testing,
    types::{self, Peak, Result},
};
use rustradio::fir;
use rustradio::graph::{Graph, GraphRunner};
use rustradio::window::WindowType;
use rustradio::{Complex, blockchain};
use rustradio::{blocks::QuadratureDemod, fir::FirFilter};
use std::{thread, time::Duration};
use tracing::{debug, info};

pub mod deemph;
pub mod filter_config;
pub mod squelch;

use filter_config::{FilterPurpose, FmFilterConfig};

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
        config: &crate::ScanningConfig,
        sdr_rx: tokio::sync::broadcast::Receiver<rustradio::Complex>,
        center_freq: f64,
        signal_tx: std::sync::mpsc::SyncSender<crate::types::Signal>,
    ) -> Result<()> {
        debug!(
            message = "Tuning into station",
            fequency_hz = self.frequency_hz / 1e6,
            signal_strength = self.signal_strength,
            peak_count = self.peak_count,
            self.max_magnitude = self.max_magnitude,
            self.avg_magnitude = self.avg_magnitude
        );
        let station_name = format!("{:.1}FM", self.frequency_hz / 1e6);

        // Create DSP graph to process this candidate's frequency
        // Use the provided center_freq and this candidate's tune frequency for the FreqXlatingFir
        let (mut detection_graph, noise_detected, audio_started_playing) = create_detection_graph(
            sdr_rx,
            config.samp_rate,
            station_name,
            config,
            center_freq,
            self.frequency_hz,
            Some(signal_tx.clone()), // Pass signal channel to squelch
        )?;
        let detection_cancel_token = detection_graph.cancel_token();

        debug!(
            "Processing candidate at {:.1} MHz with center freq {:.1} MHz",
            self.frequency_hz / 1e6,
            center_freq / 1e6
        );

        // Start DSP graph in a separate thread
        let frequency_hz = self.frequency_hz;
        thread::spawn(move || {
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
        let duration = config.duration;
        let noise_detected_clone = noise_detected.clone();
        let audio_started_playing_clone = audio_started_playing.clone();
        thread::spawn(move || {
            let check_interval = Duration::from_millis(100);
            let total_checks = (duration * 1000) / 100; // Total checks over duration in 100ms intervals

            let message_printed = false;

            for _ in 0..total_checks {
                thread::sleep(check_interval);

                // Check if squelch detected noise
                if noise_detected_clone.load(std::sync::atomic::Ordering::Relaxed) {
                    debug!("Squelch detected noise, exiting early");
                    detection_cancel_token.cancel();
                    return;
                }

                // If audio has started playing, terminate detection graph immediately
                if audio_started_playing_clone.load(std::sync::atomic::Ordering::Relaxed)
                    && !message_printed
                {
                    info!("found {:.1} MHz", frequency_hz / 1e6);
                    crate::logging::flush();

                    // OPTIMIZATION: Stop detection graph immediately when audio starts
                    // This frees up significant CPU resources for audio processing
                    debug!("Audio detected, terminating detection graph to optimize audio quality");
                    detection_cancel_token.cancel();
                    return; // Exit monitoring thread - audio processing takes over
                }
            }

            debug!(
                "Ran for {} seconds, continuing on to the next candidate",
                duration
            );
            detection_cancel_token.cancel();
        });

        Ok(())
    }
}

/// Collect RF peaks by consuming from a broadcast channel.
/// This performs FFT analysis to detect spectral peaks above the threshold.
pub fn collect_peaks(
    config: &crate::ScanningConfig,
    mut sdr_rx: tokio::sync::broadcast::Receiver<Complex>,
    center_freq: f64,
) -> Result<Vec<Peak>> {
    let peak_scan_duration = config.peak_scan_duration.unwrap_or(0.5);
    debug!("Starting peak detection scan for {peak_scan_duration} seconds...",);

    // Prepare FFT processing
    use std::collections::BTreeMap;
    let mut peaks_map: BTreeMap<u64, Peak> = BTreeMap::new();
    let mut fft_buffer = vec![rustfft::num_complex::Complex32::default(); config.fft_size];
    let mut planner = rustfft::FftPlanner::new();
    let fft = planner.plan_fft_forward(config.fft_size);

    // Calculate sampling parameters
    let samples_per_second = config.samp_rate as usize;
    let total_samples_needed = (samples_per_second as f64 * peak_scan_duration) as usize;
    let mut samples_collected = 0;
    let mut read_buffer = vec![Complex::default(); config.fft_size];

    let start_time = std::time::Instant::now();

    // Collect samples and perform peak detection
    while samples_collected < total_samples_needed {
        if start_time.elapsed().as_secs_f64() > peak_scan_duration + 1.0 {
            debug!("Peak collection timed out");
            break;
        }
        match sdr_rx.try_recv() {
            Ok(sample) => {
                read_buffer[samples_collected % config.fft_size] = sample;
                samples_collected += 1;

                if samples_collected % config.fft_size == 0 {
                    let batch_peaks = process_samples_for_peaks(
                        &read_buffer,
                        config.fft_size,
                        &mut fft_buffer,
                        &fft,
                        config,
                        center_freq,
                    );
                    for peak in batch_peaks {
                        let rounded_freq = (peak.frequency_hz / 1000.0).round() as u64;
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
                // No samples available, wait a bit
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

            // Round to nearest kHz to eliminate floating point precision errors
            let freq_hz_rounded = (freq_hz / 1000.0).round() * 1000.0;

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
    config: &crate::ScanningConfig,
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
    config: &crate::ScanningConfig,
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
                    let rounded_freq = (peak.frequency_hz / 1000.0).round() as u64;
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
    } else if freq_span_khz < 30.0 {
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
    config: &crate::ScanningConfig,
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
        if spectral_score > 0.3 {
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
    config: &crate::ScanningConfig,
    center_freq: f64,
    tune_freq: f64,
    signal_tx: Option<std::sync::mpsc::SyncSender<crate::types::Signal>>,
) -> rustradio::Result<(
    Graph,
    std::sync::Arc<std::sync::atomic::AtomicBool>,
    std::sync::Arc<std::sync::atomic::AtomicBool>,
)> {
    let mut graph = Graph::new();

    // MPSC Source
    let (source_block, prev) = crate::broadcast::BroadcastSource::new(source_receiver);
    graph.add(Box::new(source_block));

    // Calculate frequency offset for translating filter
    let frequency_offset = tune_freq - center_freq;

    // Get optimized filter configuration for detection stage
    let filter_config = FmFilterConfig::for_purpose(FilterPurpose::Detection, samp_rate);

    debug!(
        message = "Detection stage filter configuration",
        passband_khz = filter_config.channel_bandwidth / 1000.0,
        transition_khz = filter_config.transition_width / 1000.0,
        decimation = filter_config.decimation,
        estimated_taps = filter_config.estimated_taps,
        estimated_mflops = filter_config.estimated_mflops
    );

    // Verify we can handle the required frequency offset
    if !filter_config.can_handle_offset(frequency_offset, samp_rate) {
        debug!(
            message = "Frequency offset may exceed filter passband",
            frequency_offset_khz = frequency_offset / 1000.0,
            filter_cutoff_khz = filter_config.cutoff_frequency() / 1000.0
        );
    }

    let taps = fir::low_pass(
        samp_rate as f32,
        filter_config.cutoff_frequency(),
        filter_config.transition_width,
        &WindowType::Hamming,
    );

    // Debug: Check filter tap count vs estimation
    let tap_error_percent = ((taps.len() as f32 - filter_config.estimated_taps as f32)
        / filter_config.estimated_taps as f32
        * 100.0)
        .abs();
    debug!(
        message = "Filter tap verification",
        actual_taps = taps.len(),
        estimated_taps = filter_config.estimated_taps,
        error_percent = tap_error_percent
    );

    let decimation = filter_config.decimation;
    let (freq_xlating_block, prev) = FreqXlatingFir::with_real_taps(
        prev,
        &taps,
        frequency_offset as f32,
        samp_rate as f32,
        decimation,
    );
    graph.add(Box::new(freq_xlating_block));

    // Update effective sample rate after decimation
    let decimated_samp_rate = samp_rate / decimation as f64;

    // Skip additional resampling if we're already close to desired quad rate
    let quad_rate = decimated_samp_rate as f32; // Use decimated rate directly to avoid extra resampling

    // Quadrature demodulation with reduced gain to prevent distortion
    // FM deviation for broadcast is 75kHz, so gain should account for sample rate
    let fm_gain = (quad_rate / (2.0 * 75_000.0)) * 0.8; // 0.8 factor to prevent overload
    let prev = blockchain![graph, prev, QuadratureDemod::new(prev, fm_gain)];

    // FM Deemphasis
    let (deemphasis_block, deemphasis_output_stream) = Deemphasis::new(prev, quad_rate, 75.0);
    graph.add(Box::new(deemphasis_block));
    let prev = deemphasis_output_stream;

    // DETECTION OPTIMIZATION: Minimal audio processing for squelch analysis only
    // We don't need high-quality audio - just enough to distinguish audio from noise

    // Very light audio filter - just enough to remove high-frequency noise
    let taps = fir::low_pass(
        quad_rate,
        15_000.0f32, // Higher cutoff (less filtering)
        12_000.0f32, // Very wide transition (minimal taps)
        &WindowType::Hamming,
    );
    let prev = blockchain![graph, prev, FirFilter::new(prev, &taps)];

    // SKIP EXPENSIVE RESAMPLING: Use current rate for squelch analysis
    // The squelch block can analyze audio content at any reasonable sample rate
    let analysis_rate = quad_rate; // ~133kHz is fine for audio vs noise detection

    // Audio capture block (captures samples while passing them through)
    // Create audio capturer if requested
    // TODO: Wire up again
    #[allow(unused)]
    let audio_capturer = if let Some(ref capture_file) = config.capture_audio {
        match crate::file::AudioCaptureSink::new(
            capture_file,
            analysis_rate,
            config.capture_audio_duration,
        ) {
            Ok(capturer) => Some(capturer),
            Err(e) => {
                debug!("Failed to create audio capturer: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Audio capture block (captures samples while passing them through)
    // Create audio capturer if requested - needed for test fixture generation
    let audio_capturer = if let Some(ref capture_file) = config.capture_audio {
        match crate::file::AudioCaptureSink::new(
            capture_file,
            analysis_rate, // Use current sample rate for capture
            config.capture_audio_duration,
        ) {
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

    let (squelch_block, noise_detected, audio_started_playing) = SquelchBlock::new(
        prev,
        analysis_rate, // Use current rate instead of resampled audio rate
        config.squelch_learning_duration,
        signal_tx,
        tune_freq,
        center_freq,
    );
    graph.add(Box::new(squelch_block));

    // No MPSC sink needed - squelch block is terminal

    Ok((graph, noise_detected, audio_started_playing))
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
        use crate::Band;

        let config = crate::ScanningConfig::default();
        let band = Band::Fm;
        let windows = band.windows(config.samp_rate);

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
        let config = crate::ScanningConfig {
            audio_buffer_size: 4096,
            audio_sample_rate: 48000,
            band: crate::Band::Fm,
            capture_audio_duration: 3.0,
            capture_audio: None,
            capture_duration: 2.0,
            capture_iq: None,
            debug_pipeline: false,
            driver: "test".to_string(),
            duration: 1,
            exit_early: false,
            fft_size: 1024,
            peak_detection_threshold: 0.01, // Low threshold for testing
            peak_scan_duration: Some(0.1),  // Short duration for testing
            print_candidates: false,
            samp_rate: 1000000.0,
            squelch_learning_duration: 2.0,
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
    #[ignore]
    fn test_fm_detection_from_iq_file() {
        let (mut file_source, metadata) =
            crate::testing::load_iq_fixture("tests/data/iq/fm_88.9_MHz_1s.iq")
                .expect("Failed to load I/Q fixture");
        let mut config = crate::ScanningConfig::default();
        config.driver = "test".to_string();

        let peaks = collect_peaks_from_source(&config, &mut file_source)
            .expect("Failed to collect peaks from I/Q file");
        assert!(!peaks.is_empty(), "Expected to find peaks in the I/Q file");

        let candidates = find_candidates(&peaks, &config, metadata.center_frequency);
        let candidate_freqs: Vec<f64> = candidates
            .iter()
            .map(|c| (c.frequency_hz() / 1e6 * 10.0).round() / 10.0)
            .collect();

        assert_eq!(
            candidate_freqs.len(),
            2,
            "Expected exactly 2 candidates (88.9 and 89.3 MHz), but found {} candidates at: {:?}",
            candidate_freqs.len(),
            candidate_freqs
        );
        assert!(
            candidate_freqs.contains(&88.9),
            "Expected to find 88.9 MHz station"
        );
        assert!(
            candidate_freqs.contains(&89.3),
            "Expected to find 89.3 MHz station"
        );
    }

    #[test]
    fn test_no_candidates_1() {
        let (mut file_source, metadata) =
            crate::testing::load_iq_fixture("tests/data/iq/fm_88.5_MHz_1s.iq")
                .expect("Failed to load I/Q fixture");
        let mut config = crate::ScanningConfig::default();
        config.driver = "test".to_string();

        let peaks = collect_peaks_from_source(&config, &mut file_source)
            .expect("Failed to collect peaks from I/Q file");

        let candidates: Vec<_> = find_candidates(&peaks, &config, metadata.center_frequency);
        let candidate_freqs: Vec<f64> = candidates
            .iter()
            .map(|c| (c.frequency_hz() / 1e6 * 10.0).round() / 10.0)
            .collect();

        assert_eq!(
            candidates.len(),
            2,
            "Expected 1 candidate, but found {} candidates at: {:?}",
            candidates.len(),
            candidate_freqs
        );
    }
}
