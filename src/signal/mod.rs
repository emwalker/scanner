use crate::{
    core::types::{self, Peak, Result, ScanningConfig},
    file::AudioCaptureBlock,
    signal::squelch::SquelchBlock,
};
use rustradio::blockchain;
use rustradio::blocks::QuadratureDemod;
use rustradio::graph::{Graph, GraphRunner};
use std::{
    collections::HashSet,
    sync::{LazyLock, Mutex, mpsc::SyncSender},
};
use tracing::debug;

/// Global set of processed frequencies (rounded to nearest kHz) to avoid duplicate analysis
pub static PROCESSED_FREQUENCIES: LazyLock<Mutex<HashSet<u64>>> =
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
pub mod freq_xlating_fir;
pub mod frequency_tracking;
pub mod iq_capture;
pub mod peaks;
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
        sdr_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
        signal_tx: SyncSender<crate::core::types::Signal>,
        context: &crate::pipeline::AnalysisContext,
    ) -> Result<()> {
        // Delegate to the new testable pipeline function
        crate::pipeline::process_peak_to_signal(self.frequency_hz, sdr_rx, signal_tx, context)
    }
}

/// Collect RF peaks by consuming from a broadcast channel.
/// This performs FFT analysis to detect spectral peaks above the threshold.
#[allow(clippy::type_complexity)]
pub fn collect_peaks(
    config: &ScanningConfig,
    sdr_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
    center_freq: f64,
) -> Result<Vec<Peak>> {
    debug!(
        peak_scan_seconds = config.peak_scan_duration,
        center_freq_mhz = center_freq / 1e6,
        "Starting unified peak detection scan"
    );

    // Create SDR stream adapter to use unified peak detection
    let mut sdr_source = crate::testing::SdrStreamSource::new(
        sdr_rx,
        config.samp_rate,
        center_freq,
        config.peak_scan_duration,
    );

    // Use the unified peak detection implementation
    crate::signal::peaks::collect_peaks_from_source(config, &mut sdr_source)
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
    source_receiver: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
    samp_rate: f64,
    _channel_name: String,
    config: &ScanningConfig,
    center_freq: f64,
    tune_freq: f64,
    signal_tx: Option<SyncSender<crate::core::types::Signal>>,
    audio_analyzer: crate::audio::quality::AudioAnalyzer,
    progress_reporter: Option<std::sync::Arc<dyn crate::ui::ProgressReporter + Send + Sync>>,
    window_id: usize,
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

    // Skip additional resampling if we're already close to desired quad rate
    let quad_rate = decimated_samp_rate as f32; // Use decimated rate directly to avoid extra resampling

    // Quadrature demodulation with reduced gain to prevent distortion
    // FM deviation for broadcast is 75kHz, so gain should account for sample rate
    let fm_gain = (quad_rate / (2.0 * 75_000.0)) * 0.8; // 0.8 factor to prevent overload
    let prev = blockchain![graph, prev, QuadratureDemod::new(prev, fm_gain)];

    // Add proper FM deemphasis to match audio pipeline processing
    // This ensures both pipelines process the FM signal identically
    let (deemphasis_block, prev) = crate::signal::deemph::Deemphasis::new(prev, quad_rate, 75.0);
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
            modulation_type: crate::core::types::ModulationType::WFM,
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

    let (audio_capture_block, audio_capture_output) = AudioCaptureBlock::new(prev, None);
    graph.add(Box::new(audio_capture_block));
    let prev = audio_capture_output;

    use crate::signal::squelch::SquelchConfig;
    let squelch_config = SquelchConfig {
        sample_rate: analysis_rate, // Use current rate instead of resampled audio rate
        learning_duration: config.squelch_learning_duration,
        signal_tx,
        frequency_hz: tune_freq,
        center_freq,
        squelch_disabled: config.disable_squelch,
        threshold: config.squelch_threshold,
        fft_size: config.fft_size,
        audio_analyzer,
        audio_capturer,
        progress_reporter,
        window_id,
        tuner_id: None, // Tuner ID not available without device reference
    };
    let (squelch_block, decision_state) = SquelchBlock::new(prev, squelch_config);
    graph.add(Box::new(squelch_block));

    Ok((graph, decision_state))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::quality::AudioAnalyzer;
    use crate::testing::*;

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
            duration: 1,
            fft_size: 1024,
            peak_detection_threshold: 0.01, // Low threshold for testing
            peak_scan_duration: 0.1,        // Short duration for testing
            samp_rate: 1000000.0,
            disable_frequency_tracking: true, // Disable for test to keep existing behavior
            audio_analyzer: AudioAnalyzer::mock(),

            // Disable signal averaging and CFAR features for baseline test behavior
            enable_exponential_smoothing: false,
            enable_multi_frame_averaging: false,
            enable_coherent_integration: false,
            enable_moving_average_filter: false,
            enable_cfar_detection: false,

            // Disable spectral preprocessing for baseline test behavior
            enable_windowing: false,

            ..Default::default()
        };

        // Create mock source with a signal at +100kHz offset from center
        let mut mock_source = MockSampleSource::new(
            1000000.0,  // 1 MHz sample rate
            88900000.0, // 88.9 MHz center frequency
            100000,     // 100k samples max
            100000.0,   // 100 kHz offset signal
        );

        let peaks =
            crate::signal::peaks::collect_peaks_from_source(&config, &mut mock_source).unwrap();

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
