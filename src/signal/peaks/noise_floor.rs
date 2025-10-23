//! Dynamic Noise Floor Estimation
//!
//! This module implements dynamic noise floor estimation using percentile-based
//! statistical methods to adapt detection thresholds to varying RF environments.

use std::{cmp::Ordering, collections::VecDeque};

use tracing::debug;

use crate::core::types::Peak;

/// Configuration for dynamic noise floor estimation
#[derive(Debug, Clone)]
pub struct NoiseFloorConfig {
    /// Percentile for noise floor calculation (0.0-1.0, typical: 0.1 for 10th percentile)
    pub noise_percentile: f32,
    /// Number of frames to maintain in noise history
    pub history_frames: usize,
    /// Minimum threshold multiplier above noise floor
    pub threshold_multiplier: f32,
    /// Maximum allowed threshold to prevent over-suppression
    pub max_threshold: f32,
    /// Minimum allowed threshold to maintain sensitivity
    pub min_threshold: f32,
    /// Smoothing factor for threshold adaptation (0.0-1.0)
    pub adaptation_rate: f32,
}

impl Default for NoiseFloorConfig {
    fn default() -> Self {
        Self {
            noise_percentile: 0.25,    // 25th percentile (less conservative)
            history_frames: 8,         // Track last 8 frames (faster adaptation)
            threshold_multiplier: 1.6, // 1.6x above noise floor (more aggressive)
            max_threshold: 25.0,       // Lower maximum threshold cap for sensitivity
            min_threshold: 0.5,        // Lower minimum threshold floor for weak signals
            adaptation_rate: 0.35,     // 35% adaptation rate (faster learning)
        }
    }
}

/// Dynamic noise floor estimator
pub struct NoiseFloorEstimator {
    config: NoiseFloorConfig,
    magnitude_history: VecDeque<Vec<f32>>,
    current_noise_floor: f32,
    current_threshold: f32,
    frame_count: u64,
}

impl NoiseFloorEstimator {
    pub fn new(config: NoiseFloorConfig) -> Self {
        Self {
            current_noise_floor: config.min_threshold,
            current_threshold: config.min_threshold * config.threshold_multiplier,
            magnitude_history: VecDeque::new(),
            frame_count: 0,
            config,
        }
    }

    /// Update noise floor estimation with new magnitude data
    pub fn update_noise_floor(&mut self, magnitudes: &[f32]) {
        self.frame_count += 1;

        // Add current frame to history
        self.magnitude_history.push_back(magnitudes.to_vec());

        // Maintain history size
        while self.magnitude_history.len() > self.config.history_frames {
            self.magnitude_history.pop_front();
        }

        // Calculate new noise floor if we have sufficient history
        if self.magnitude_history.len() >= 3 {
            let new_noise_floor = self.calculate_percentile_noise_floor();

            // Smooth the noise floor adaptation
            self.current_noise_floor = self.current_noise_floor
                * (1.0 - self.config.adaptation_rate)
                + new_noise_floor * self.config.adaptation_rate;

            // Update threshold based on noise floor
            let new_threshold = self.current_noise_floor * self.config.threshold_multiplier;
            self.current_threshold = new_threshold
                .max(self.config.min_threshold)
                .min(self.config.max_threshold);

            debug!(
                frame = self.frame_count,
                noise_floor = self.current_noise_floor,
                threshold = self.current_threshold,
                "Updated dynamic noise floor"
            );
        }
    }

    /// Get current adaptive threshold
    pub fn current_threshold(&self) -> f32 {
        self.current_threshold
    }

    /// Get current noise floor estimate
    pub fn current_noise_floor(&self) -> f32 {
        self.current_noise_floor
    }

    /// Calculate percentile-based noise floor from magnitude history
    fn calculate_percentile_noise_floor(&self) -> f32 {
        if self.magnitude_history.is_empty() {
            return self.config.min_threshold;
        }

        // Flatten all magnitude samples from history
        let mut all_magnitudes: Vec<f32> =
            self.magnitude_history.iter().flatten().copied().collect();

        if all_magnitudes.is_empty() {
            return self.config.min_threshold;
        }

        // Sort for percentile calculation
        all_magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        // Calculate percentile index
        let percentile_index =
            ((all_magnitudes.len() - 1) as f32 * self.config.noise_percentile) as usize;

        all_magnitudes[percentile_index].max(self.config.min_threshold)
    }

    /// Extract peaks using dynamic threshold
    pub fn extract_peaks_with_dynamic_threshold(
        &mut self,
        magnitudes: &[f32],
        fft_size: usize,
        sample_rate: f64,
        center_freq: f64,
    ) -> Vec<Peak> {
        // Update noise floor with current data
        self.update_noise_floor(magnitudes);

        // Use current threshold for peak detection
        let threshold = self.current_threshold();

        // Apply local background subtraction for enhanced detection
        let processed_magnitudes = self.apply_local_background_subtraction(magnitudes);

        // Extract peaks using processed magnitudes and dynamic threshold
        self.extract_peaks_from_processed_magnitudes(
            &processed_magnitudes,
            threshold,
            fft_size,
            sample_rate,
            center_freq,
        )
    }

    /// Apply local background subtraction to enhance peak detection
    fn apply_local_background_subtraction(&self, magnitudes: &[f32]) -> Vec<f32> {
        let window_size = 11; // Local background estimation window
        let mut processed = Vec::with_capacity(magnitudes.len());

        for i in 0..magnitudes.len() {
            // Define local window around current bin
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(magnitudes.len());

            // Calculate local background (exclude center bins to avoid signal contamination)
            let mut local_background = Vec::new();
            for (idx, &magnitude) in magnitudes.iter().enumerate().take(end).skip(start) {
                if (idx as i32 - i as i32).abs() > 2 {
                    // Exclude center ±2 bins
                    local_background.push(magnitude);
                }
            }

            // Calculate median of local background
            if !local_background.is_empty() {
                local_background.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                let median_idx = local_background.len() / 2;
                let local_median = local_background[median_idx];

                // Subtract background, ensuring non-negative result
                processed.push((magnitudes[i] - local_median).max(0.0));
            } else {
                processed.push(magnitudes[i]);
            }
        }

        processed
    }

    /// Extract peaks from background-subtracted magnitudes
    fn extract_peaks_from_processed_magnitudes(
        &self,
        magnitudes: &[f32],
        threshold: f32,
        fft_size: usize,
        sample_rate: f64,
        center_freq: f64,
    ) -> Vec<Peak> {
        let mut peaks = Vec::new();

        // Detect peaks: local maxima above dynamic threshold
        for i in 1..magnitudes.len() - 1 {
            if magnitudes[i] > threshold
                && magnitudes[i] > magnitudes[i - 1]
                && magnitudes[i] > magnitudes[i + 1]
            {
                // Convert FFT bin to frequency
                let freq_offset = (i as f64 / fft_size as f64) * sample_rate;
                let freq_hz = center_freq - (sample_rate / 2.0) + freq_offset;

                // Round to nearest 100 kHz
                let freq_hz_rounded = (freq_hz / 100000.0).round() * 100000.0;

                peaks.push(Peak {
                    frequency_hz: freq_hz_rounded,
                    magnitude: magnitudes[i],
                });
            }
        }

        peaks
    }

    /// Get statistics for monitoring and debugging
    pub fn statistics(&self) -> NoiseFloorStatistics {
        NoiseFloorStatistics {
            current_noise_floor: self.current_noise_floor,
            current_threshold: self.current_threshold,
            frame_count: self.frame_count,
            history_size: self.magnitude_history.len(),
            threshold_multiplier: self.config.threshold_multiplier,
        }
    }
}

/// Statistics for noise floor estimation monitoring
#[derive(Debug)]
pub struct NoiseFloorStatistics {
    pub current_noise_floor: f32,
    pub current_threshold: f32,
    pub frame_count: u64,
    pub history_size: usize,
    pub threshold_multiplier: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that percentile-based noise floor tracks actual noise
    #[test]
    fn test_percentile_noise_floor_tracking() {
        let config = NoiseFloorConfig {
            noise_percentile: 0.1,
            history_frames: 5,
            threshold_multiplier: 3.0,
            max_threshold: 50.0,
            min_threshold: 1.0,
            adaptation_rate: 0.5, // Faster adaptation for testing
        };

        let mut estimator = NoiseFloorEstimator::new(config);

        // Simulate frames with different noise levels

        // Low noise scenario
        for _ in 0..3 {
            let low_noise_magnitudes: Vec<f32> =
                (0..100).map(|_| 2.0 + rand::random::<f32>()).collect();
            estimator.update_noise_floor(&low_noise_magnitudes);
        }

        let low_noise_threshold = estimator.current_threshold();
        let low_noise_floor = estimator.current_noise_floor();

        // High noise scenario
        for _ in 0..5 {
            let high_noise_magnitudes: Vec<f32> =
                (0..100).map(|_| 8.0 + rand::random::<f32>()).collect();
            estimator.update_noise_floor(&high_noise_magnitudes);
        }

        let high_noise_threshold = estimator.current_threshold();
        let high_noise_floor = estimator.current_noise_floor();

        // Noise floor should adapt to higher noise level
        assert!(
            high_noise_floor > low_noise_floor,
            "Noise floor should increase with higher noise: {} > {}",
            high_noise_floor,
            low_noise_floor
        );

        // Threshold should increase proportionally
        assert!(
            high_noise_threshold > low_noise_threshold,
            "Threshold should increase with noise floor: {} > {}",
            high_noise_threshold,
            low_noise_threshold
        );

        println!(
            "Low noise: floor={:.2}, threshold={:.2}",
            low_noise_floor, low_noise_threshold
        );
        println!(
            "High noise: floor={:.2}, threshold={:.2}",
            high_noise_floor, high_noise_threshold
        );
    }

    /// Test that local background subtraction works correctly
    #[test]
    fn test_local_background_subtraction() {
        let config = NoiseFloorConfig::default();
        let estimator = NoiseFloorEstimator::new(config);

        // Create magnitude spectrum with signal on noise background
        let mut magnitudes = vec![2.0; 100]; // Uniform noise background
        magnitudes[50] = 10.0; // Strong signal
        magnitudes[51] = 8.0; // Signal leakage
        magnitudes[49] = 8.0; // Signal leakage

        let processed = estimator.apply_local_background_subtraction(&magnitudes);

        println!("Original signal magnitude: {}", magnitudes[50]);
        println!("Processed signal magnitude: {}", processed[50]);
        println!("Original background: {}", magnitudes[10]);
        println!("Processed background: {}", processed[10]);

        // Signal should be preserved or enhanced (local background is ~2.0, so signal becomes 10.0
        // - 2.0 = 8.0)
        assert!(
            processed[50] >= 6.0,
            "Signal should be preserved after background subtraction: got {}",
            processed[50]
        );

        // Background areas should be reduced to near zero (2.0 - 2.0 = 0.0)
        assert!(
            processed[10] <= 1.0,
            "Background should be significantly reduced: got {}",
            processed[10]
        );
    }

    /// Test performance consistency across RF environments
    #[test]
    fn test_performance_across_environments() {
        let config = NoiseFloorConfig {
            adaptation_rate: 0.3,
            ..Default::default()
        };

        let _estimator = NoiseFloorEstimator::new(config);

        // Test different RF environments
        let environments = vec![
            ("clean", 1.0, 2.0),        // Clean environment (low noise)
            ("urban", 5.0, 8.0),        // Urban environment (medium noise)
            ("industrial", 10.0, 15.0), // Industrial environment (high noise)
        ];

        for (env_name, base_noise, noise_variance) in environments {
            // Reset estimator for new environment
            let mut local_estimator = NoiseFloorEstimator::new(NoiseFloorConfig {
                adaptation_rate: 0.3,
                ..Default::default()
            });

            // Simulate several frames in this environment
            let mut thresholds = Vec::new();
            for _ in 0..10 {
                let magnitudes: Vec<f32> = (0..100)
                    .map(|_| base_noise + rand::random::<f32>() * noise_variance)
                    .collect();

                local_estimator.update_noise_floor(&magnitudes);
                thresholds.push(local_estimator.current_threshold());
            }

            // Check that threshold stabilizes (low variance in later frames)
            let late_thresholds: Vec<f32> = thresholds.iter().skip(5).copied().collect();
            let mean = late_thresholds.iter().sum::<f32>() / late_thresholds.len() as f32;
            let variance = late_thresholds
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>()
                / late_thresholds.len() as f32;
            let coefficient_of_variation = variance.sqrt() / mean;

            println!(
                "{} environment: mean_threshold={:.2}, CV={:.3}",
                env_name, mean, coefficient_of_variation
            );

            // Threshold should be stable (low coefficient of variation)
            assert!(
                coefficient_of_variation < 0.2,
                "Threshold should be stable in {} environment (CV: {:.3})",
                env_name,
                coefficient_of_variation
            );
        }
    }

    /// Test handling of sudden interference changes
    #[test]
    fn test_sudden_interference_handling() {
        let config = NoiseFloorConfig {
            adaptation_rate: 0.6, // Very fast adaptation for testing
            max_threshold: 100.0,
            history_frames: 5, // Shorter history for faster adaptation
            ..Default::default()
        };

        let mut estimator = NoiseFloorEstimator::new(config);

        // Establish baseline in clean environment (use deterministic values)
        for _ in 0..5 {
            let clean_magnitudes: Vec<f32> = vec![1.5; 100]; // Deterministic clean signal
            estimator.update_noise_floor(&clean_magnitudes);
        }

        let baseline_threshold = estimator.current_threshold();

        // Sudden interference spike (use deterministic values)
        for _ in 0..8 {
            let interference_magnitudes: Vec<f32> = vec![20.0; 100]; // High interference
            estimator.update_noise_floor(&interference_magnitudes);
        }

        let interference_threshold = estimator.current_threshold();

        println!("Baseline: {:.2}", baseline_threshold);
        println!("Interference: {:.2}", interference_threshold);

        // System should adapt to higher interference level (more lenient check)
        assert!(
            interference_threshold > baseline_threshold * 1.5,
            "Should adapt to interference: {} > {}",
            interference_threshold,
            baseline_threshold * 1.5
        );

        // Return to clean environment
        for _ in 0..8 {
            let clean_magnitudes: Vec<f32> = vec![1.5; 100]; // Back to clean
            estimator.update_noise_floor(&clean_magnitudes);
        }

        let recovered_threshold = estimator.current_threshold();

        // Should adapt back down (more lenient check)
        assert!(
            recovered_threshold < interference_threshold * 0.8,
            "Should adapt back down: {} < {}",
            recovered_threshold,
            interference_threshold * 0.8
        );

        println!("Recovered: {:.2}", recovered_threshold);
    }
}
