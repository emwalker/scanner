//! Peak Detection Module
//!
//! This module contains all peak detection algorithms and related functionality.
//! It provides a unified interface for detecting RF peaks from FFT magnitude data.

pub mod averaging;
pub mod cfar;
pub mod extraction;
pub mod multi_frame;
pub mod noise_floor;
pub mod windowing;

use std::{collections::BTreeMap, sync::Arc};

use rustfft::num_complex::Complex32;
use rustradio::Complex;
use tracing::debug;

use crate::{
    core::types::{Peak, Result, ScanningConfig},
    testing::SampleSource,
};

/// State for signal averaging algorithms
pub struct SignalAveragingState {
    pub smoothed_magnitudes: Option<Vec<f32>>,
    pub coherent_integration_accumulator: Option<Vec<f32>>,
    pub integration_cycles: usize,
    pub multi_frame_accumulator: Option<Vec<f32>>,
    pub frame_count: usize,
}

/// State for dynamic noise floor estimation
pub struct NoiseFloorState {
    pub estimator: Option<noise_floor::NoiseFloorEstimator>,
}

/// Processing context to reduce function argument count
pub struct ProcessingContext<'a> {
    pub config: &'a ScanningConfig,
    pub center_freq: f64,
    pub averaging_state: &'a mut SignalAveragingState,
    pub noise_floor_state: &'a mut NoiseFloorState,
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
    let mut fft_buffer = vec![Complex32::default(); config.peak_detection.fft_size];
    let mut planner = rustfft::FftPlanner::new();
    let fft = planner.plan_fft_forward(config.peak_detection.fft_size);

    // Initialize signal averaging state
    let mut averaging_state = SignalAveragingState {
        smoothed_magnitudes: None,
        coherent_integration_accumulator: None,
        integration_cycles: 0,
        multi_frame_accumulator: None,
        frame_count: 0,
    };

    // Initialize dynamic noise floor state
    let mut noise_floor_state = NoiseFloorState {
        estimator: if config.peak_detection.noise_floor.enabled {
            Some(noise_floor::NoiseFloorEstimator::new(
                noise_floor::NoiseFloorConfig {
                    noise_percentile: config.peak_detection.noise_floor.percentile,
                    history_frames: config.peak_detection.noise_floor.history_frames,
                    threshold_multiplier: config.peak_detection.noise_floor.threshold_multiplier,
                    adaptation_rate: config.peak_detection.noise_floor.adaptation_rate,
                    ..Default::default()
                },
            ))
        } else {
            None
        },
    };

    // Initialize multi-frame integrator if enabled
    let mut multi_frame_integrator = if config.peak_detection.multi_frame.enabled {
        Some(multi_frame::MultiFrameIntegrator::new(
            multi_frame::MultiFrameConfig {
                history_frames: config.peak_detection.multi_frame.history_frames,
                confirmation_threshold: config.peak_detection.multi_frame.confirmation_threshold,
                frequency_tolerance: config.peak_detection.multi_frame.frequency_tolerance,
                max_frame_age: config.peak_detection.multi_frame.max_age,
            },
        ))
    } else {
        None
    };

    // Calculate sampling parameters
    let samples_per_second = sample_source.sample_rate() as usize;
    let total_samples_needed = (samples_per_second as f64 * peak_scan_duration) as usize;
    let mut samples_collected = 0;
    let mut read_buffer = vec![Complex::default(); config.peak_detection.fft_size];

    // Collect samples and perform peak detection
    while samples_collected < total_samples_needed {
        match sample_source.read_samples(&mut read_buffer) {
            Ok(samples_read) => {
                if samples_read == 0 {
                    break; // End of file reached
                }

                let mut context = ProcessingContext {
                    config,
                    center_freq: sample_source.center_frequency(),
                    averaging_state: &mut averaging_state,
                    noise_floor_state: &mut noise_floor_state,
                };

                let batch_peaks = process_samples_for_peaks(&mut PeakProcessingParams {
                    read_buffer: &read_buffer,
                    samples_read,
                    fft_buffer: &mut fft_buffer,
                    fft: &fft,
                    context: &mut context,
                });

                // Process peaks through multi-frame integration if enabled
                let final_peaks = if let Some(ref mut integrator) = multi_frame_integrator {
                    integrator.process_frame(batch_peaks)
                } else {
                    batch_peaks
                };

                for peak in final_peaks {
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

    // Log multi-frame integration statistics if enabled
    if let Some(ref integrator) = multi_frame_integrator {
        let stats = integrator.statistics();
        debug!(
            total_trackers = stats.total_trackers,
            confirmed_trackers = stats.confirmed_trackers,
            pending_trackers = stats.pending_trackers,
            current_frame = stats.current_frame,
            "Multi-frame integration statistics"
        );
    }

    debug!("Peak detection scan complete. Found {} peaks.", peaks.len());

    Ok(peaks)
}

pub struct PeakProcessingParams<'a> {
    pub read_buffer: &'a [Complex],
    pub samples_read: usize,
    pub fft_buffer: &'a mut [Complex32],
    pub fft: &'a Arc<dyn rustfft::Fft<f32>>,
    pub context: &'a mut ProcessingContext<'a>,
}

/// Process a batch of samples and extract peaks using the configured pipeline
pub fn process_samples_for_peaks(params: &mut PeakProcessingParams) -> Vec<Peak> {
    // Copy samples to FFT buffer and convert to real samples for windowing
    let mut real_samples = Vec::with_capacity(params.context.config.peak_detection.fft_size);
    for sample in params.read_buffer.iter().take(
        params
            .samples_read
            .min(params.context.config.peak_detection.fft_size),
    ) {
        // For windowing, we use the magnitude of complex samples
        real_samples.push((sample.re * sample.re + sample.im * sample.im).sqrt());
    }

    // Apply windowing if enabled
    if params.context.config.peak_detection.windowing.enabled {
        windowing::apply_window(
            &mut real_samples,
            &params.context.config.peak_detection.windowing.window_type,
        );
    }

    // Apply zero-padding if enabled
    if params
        .context
        .config
        .peak_detection
        .windowing
        .zero_padding_factor
        > 1
    {
        windowing::apply_zero_padding(
            &mut real_samples,
            params
                .context
                .config
                .peak_detection
                .windowing
                .zero_padding_factor,
        );
    }

    // Copy windowed samples to FFT buffer (convert back to complex)
    params.fft_buffer.fill(Complex32::new(0.0, 0.0)); // Clear buffer first
    for (i, &real_sample) in real_samples
        .iter()
        .enumerate()
        .take(params.fft_buffer.len())
    {
        params.fft_buffer[i] = Complex32::new(real_sample, 0.0);
    }

    params.fft.process(params.fft_buffer);
    let mut magnitudes: Vec<f32> = params.fft_buffer.iter().map(|c| c.norm_sqr()).collect();

    // Dynamic noise floor estimation takes priority over other detection methods
    if params.context.config.peak_detection.noise_floor.enabled
        && let Some(ref mut estimator) = params.context.noise_floor_state.estimator
    {
        return estimator.extract_peaks_with_dynamic_threshold(
            &magnitudes,
            params.context.config.peak_detection.fft_size,
            params.context.config.samp_rate,
            params.context.center_freq,
        );
    }

    // CFAR detection works best on raw or lightly processed magnitudes
    // Apply it early, before heavy smoothing that changes noise characteristics
    if params.context.config.peak_detection.cfar.enabled {
        return cfar::extract_peaks_with_cfar(
            &magnitudes,
            params.context.config.peak_detection.cfar.threshold_factor,
            params.context.config.peak_detection.cfar.guard_cells,
            params.context.config.peak_detection.cfar.reference_cells,
            params.context.config.peak_detection.fft_size,
            params.context.config.samp_rate,
            params.context.center_freq,
        );
    }

    // For non-CFAR detection, apply signal averaging improvements to enhance signal quality
    if params
        .context
        .config
        .peak_detection
        .averaging
        .multi_frame_averaging
        .enabled
    {
        let should_extract_peaks = averaging::apply_multi_frame_averaging(
            &mut magnitudes,
            &mut params.context.averaging_state.multi_frame_accumulator,
            &mut params.context.averaging_state.frame_count,
            params
                .context
                .config
                .peak_detection
                .averaging
                .multi_frame_averaging
                .frames,
        );

        // If we haven't accumulated enough frames yet, return empty peaks
        if !should_extract_peaks {
            return Vec::new();
        }
    }

    if params
        .context
        .config
        .peak_detection
        .averaging
        .moving_average
        .enabled
    {
        averaging::apply_moving_average_filter(
            &mut magnitudes,
            params
                .context
                .config
                .peak_detection
                .averaging
                .moving_average
                .window_size,
        );
    }

    if params
        .context
        .config
        .peak_detection
        .averaging
        .coherent_integration_enabled
    {
        averaging::apply_coherent_integration(
            &mut magnitudes,
            &mut params
                .context
                .averaging_state
                .coherent_integration_accumulator,
            &mut params.context.averaging_state.integration_cycles,
        );
    }

    if params
        .context
        .config
        .peak_detection
        .averaging
        .exponential_smoothing
        .enabled
    {
        averaging::apply_exponential_smoothing(
            &mut magnitudes,
            &mut params.context.averaging_state.smoothed_magnitudes,
            params
                .context
                .config
                .peak_detection
                .averaging
                .exponential_smoothing
                .alpha,
        );
    }

    // Use simple threshold detection for averaged magnitudes
    extraction::extract_peaks_from_magnitudes(
        &magnitudes,
        params.context.config.peak_detection.threshold,
        params.context.config.peak_detection.fft_size,
        params.context.config.samp_rate,
        params.context.center_freq,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        audio::quality::AudioAnalyzer,
        core::types::ScanningConfig,
        testing::signal_generation::{PeakTestSignalGenerator, TestSignal},
    };

    /// Integration test: Combined detection regression tests
    #[test]
    fn test_combined_phases_do_not_drastically_reduce_detection() {
        let mut baseline_generator = create_multi_signal_detection_scenario();
        let mut baseline_config = ScanningConfig::default();
        baseline_config.audio.buffer_size = 8192;
        baseline_config.audio.analyzer = AudioAnalyzer::mock();
        baseline_config.scanning_windows = Some(3);
        baseline_config.peak_detection.fft_size = 1024;
        baseline_config.peak_detection.scan_duration = 0.5;

        // Baseline: All features disabled
        baseline_config
            .peak_detection
            .averaging
            .exponential_smoothing
            .enabled = false;
        baseline_config
            .peak_detection
            .averaging
            .multi_frame_averaging
            .enabled = false;
        baseline_config
            .peak_detection
            .averaging
            .coherent_integration_enabled = false;
        baseline_config
            .peak_detection
            .averaging
            .moving_average
            .enabled = false;
        baseline_config.peak_detection.cfar.enabled = false;
        baseline_config.peak_detection.windowing.enabled = false;
        baseline_config.peak_detection.multi_frame.enabled = false;

        let baseline_peaks = collect_peaks_from_source(&baseline_config, &mut baseline_generator)
            .expect("Failed to collect baseline peaks");

        // Test with both signal averaging and CFAR enabled (exclude newer features)
        let mut combined_generator = create_multi_signal_detection_scenario();
        let mut combined_config = ScanningConfig::default();
        combined_config.audio.buffer_size = 8192;
        combined_config.audio.analyzer = AudioAnalyzer::mock();
        combined_config.scanning_windows = Some(3);
        combined_config.peak_detection.fft_size = 1024;
        combined_config.peak_detection.scan_duration = 0.5;

        // Test combination: Signal averaging + CFAR enabled, newer features disabled
        combined_config
            .peak_detection
            .averaging
            .exponential_smoothing
            .enabled = true;
        combined_config
            .peak_detection
            .averaging
            .multi_frame_averaging
            .enabled = true;
        combined_config
            .peak_detection
            .averaging
            .coherent_integration_enabled = true;
        combined_config
            .peak_detection
            .averaging
            .moving_average
            .enabled = true;
        combined_config.peak_detection.cfar.enabled = true;
        combined_config.peak_detection.windowing.enabled = false;
        combined_config.peak_detection.multi_frame.enabled = false;

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

        // This is the critical regression test - combined phases should not cause massive detection
        // loss
        assert!(
            detection_ratio >= 0.5, /* Allow up to 50% reduction, but this indicates a serious
                                     * problem */
            "Combined signal averaging + CFAR should not reduce detection count by more than 50%. \
             Got {:.1}% reduction (ratio: {:.2})",
            (1.0 - detection_ratio) * 100.0,
            detection_ratio
        );

        // Warn if we see significant reduction
        if detection_ratio < 0.8 {
            println!(
                "🚨 WARNING: Combined features reduced detection by {:.1}% - investigate signal \
                 averaging/CFAR interaction",
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
