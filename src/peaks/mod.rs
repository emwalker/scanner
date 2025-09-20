//! Peak Detection Module
//!
//! This module contains all peak detection algorithms and related functionality.
//! It provides a unified interface for detecting RF peaks from FFT magnitude data.

pub mod averaging;
pub mod cfar;
pub mod extraction;
pub mod windowing;

use crate::testing::SampleSource;
use crate::types::{Peak, Result, ScanningConfig};
use rustfft::num_complex::Complex32;
use rustradio::Complex;
use std::collections::BTreeMap;
use std::sync::Arc;
use tracing::debug;

/// State for signal averaging algorithms
pub struct SignalAveragingState {
    pub smoothed_magnitudes: Option<Vec<f32>>,
    pub coherent_integration_accumulator: Option<Vec<f32>>,
    pub integration_cycles: usize,
    pub multi_frame_accumulator: Option<Vec<f32>>,
    pub frame_count: usize,
}

/// Main entry point for peak detection from any sample source
pub fn collect_peaks_from_source(
    config: &ScanningConfig,
    sample_source: &mut dyn SampleSource,
) -> Result<Vec<Peak>> {
    let peak_scan_duration = sample_source.peak_scan_duration();
    debug!("Starting peak detection scan for {peak_scan_duration} seconds...",);

    // Prepare FFT processing
    let mut peaks_map: BTreeMap<u64, Peak> = BTreeMap::new();
    let mut fft_buffer = vec![Complex32::default(); config.fft_size];
    let mut planner = rustfft::FftPlanner::new();
    let fft = planner.plan_fft_forward(config.fft_size);

    // Initialize signal averaging state
    let mut averaging_state = SignalAveragingState {
        smoothed_magnitudes: None,
        coherent_integration_accumulator: None,
        integration_cycles: 0,
        multi_frame_accumulator: None,
        frame_count: 0,
    };

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
                    &mut averaging_state,
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

    debug!("Peak detection scan complete. Found {} peaks.", peaks.len());

    Ok(peaks)
}

/// Process a batch of samples and extract peaks using the configured pipeline
pub fn process_samples_for_peaks(
    read_buffer: &[Complex],
    samples_read: usize,
    fft_buffer: &mut [Complex32],
    fft: &Arc<dyn rustfft::Fft<f32>>,
    config: &ScanningConfig,
    center_freq: f64,
    averaging_state: &mut SignalAveragingState,
) -> Vec<Peak> {
    // Copy samples to FFT buffer and convert to real samples for windowing
    let mut real_samples = Vec::with_capacity(config.fft_size);
    for sample in read_buffer.iter().take(samples_read.min(config.fft_size)) {
        // For windowing, we use the magnitude of complex samples
        real_samples.push((sample.re * sample.re + sample.im * sample.im).sqrt());
    }

    // Apply windowing if enabled
    if config.enable_windowing {
        windowing::apply_window(&mut real_samples, &config.window_type);
    }

    // Apply zero-padding if enabled
    if config.zero_padding_factor > 1 {
        windowing::apply_zero_padding(&mut real_samples, config.zero_padding_factor);
    }

    // Copy windowed samples to FFT buffer (convert back to complex)
    fft_buffer.fill(Complex32::new(0.0, 0.0)); // Clear buffer first
    for (i, &real_sample) in real_samples.iter().enumerate().take(fft_buffer.len()) {
        fft_buffer[i] = Complex32::new(real_sample, 0.0);
    }

    fft.process(fft_buffer);
    let mut magnitudes: Vec<f32> = fft_buffer.iter().map(|c| c.norm_sqr()).collect();

    // CFAR detection works best on raw or lightly processed magnitudes
    // Apply it early, before heavy smoothing that changes noise characteristics
    if config.enable_cfar_detection {
        return cfar::extract_peaks_with_cfar(
            &magnitudes,
            config.cfar_threshold_factor,
            config.cfar_guard_cells,
            config.cfar_reference_cells,
            config.fft_size,
            config.samp_rate,
            center_freq,
        );
    }

    // For non-CFAR detection, apply signal averaging improvements to enhance signal quality
    if config.enable_multi_frame_averaging {
        let should_extract_peaks = averaging::apply_multi_frame_averaging(
            &mut magnitudes,
            &mut averaging_state.multi_frame_accumulator,
            &mut averaging_state.frame_count,
            config.averaging_frames,
        );

        // If we haven't accumulated enough frames yet, return empty peaks
        if !should_extract_peaks {
            return Vec::new();
        }
    }

    if config.enable_moving_average_filter {
        averaging::apply_moving_average_filter(&mut magnitudes, config.moving_average_window_size);
    }

    if config.enable_coherent_integration {
        averaging::apply_coherent_integration(
            &mut magnitudes,
            &mut averaging_state.coherent_integration_accumulator,
            &mut averaging_state.integration_cycles,
        );
    }

    if config.enable_exponential_smoothing {
        averaging::apply_exponential_smoothing(
            &mut magnitudes,
            &mut averaging_state.smoothed_magnitudes,
            config.smoothing_alpha,
        );
    }

    // Use simple threshold detection for averaged magnitudes
    extraction::extract_peaks_from_magnitudes(
        &magnitudes,
        config.peak_detection_threshold,
        config.fft_size,
        config.samp_rate,
        center_freq,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::signal_generation::{PeakTestSignalGenerator, TestSignal};
    use crate::types::ScanningConfig;

    /// Integration test: Combined detection regression tests
    #[test]
    fn test_combined_phases_do_not_drastically_reduce_detection() {
        let _ = tracing_subscriber::fmt::try_init();

        let mut baseline_generator = create_multi_signal_detection_scenario();
        let baseline_config = ScanningConfig {
            audio_buffer_size: 8192,
            scanning_windows: Some(3),
            fft_size: 1024,
            peak_scan_duration: 0.5,
            audio_analyzer: crate::audio_quality::AudioAnalyzer::mock(),

            // Baseline: All features disabled
            enable_exponential_smoothing: false,
            enable_multi_frame_averaging: false,
            enable_coherent_integration: false,
            enable_moving_average_filter: false,
            enable_cfar_detection: false,

            ..Default::default()
        };

        let baseline_peaks = collect_peaks_from_source(&baseline_config, &mut baseline_generator)
            .expect("Failed to collect baseline peaks");

        // Test with both signal averaging and CFAR enabled (current defaults)
        let mut combined_generator = create_multi_signal_detection_scenario();
        let combined_config = ScanningConfig {
            // Use current defaults (both features enabled)
            ..Default::default()
        };

        let combined_peaks = collect_peaks_from_source(&combined_config, &mut combined_generator)
            .expect("Failed to collect combined peaks");

        println!("Baseline detections: {}", baseline_peaks.len());
        println!(
            "Combined (Signal Averaging + CFAR) detections: {}",
            combined_peaks.len()
        );

        let detection_ratio = combined_peaks.len() as f32 / baseline_peaks.len() as f32;
        println!(
            "Detection ratio (Combined / Baseline): {:.2}",
            detection_ratio
        );

        // This is the critical regression test - combined phases should not cause massive detection loss
        assert!(
            detection_ratio >= 0.5, // Allow up to 50% reduction, but this indicates a serious problem
            "Combined signal averaging + CFAR should not reduce detection count by more than 50%. Got {:.1}% reduction (ratio: {:.2})",
            (1.0 - detection_ratio) * 100.0,
            detection_ratio
        );

        // Warn if we see significant reduction
        if detection_ratio < 0.8 {
            println!(
                "🚨 WARNING: Combined features reduced detection by {:.1}% - investigate signal averaging/CFAR interaction",
                (1.0 - detection_ratio) * 100.0
            );
        }
    }

    fn create_multi_signal_detection_scenario() -> PeakTestSignalGenerator {
        let mut generator = PeakTestSignalGenerator::new(
            2_000_000.0,  // sample_rate
            89_000_000.0, // center_frequency
            1_000_000,    // max_samples (0.5 seconds)
            0.3,          // Moderate noise level
        );

        // Add multiple signals at different strengths to test detection sensitivity
        generator.add_signal(TestSignal::new(88_700_000.0, 0.25, "Signal1")); // Strong
        generator.add_signal(TestSignal::new(88_900_000.0, 0.15, "Signal2")); // Medium
        generator.add_signal(TestSignal::new(89_100_000.0, 0.10, "Signal3")); // Weak
        generator.add_signal(TestSignal::new(89_300_000.0, 0.08, "Signal4")); // Very weak
        generator.add_signal(TestSignal::new(89_500_000.0, 0.05, "Signal5")); // Marginal

        generator
    }
}
