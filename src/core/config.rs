mod audio;
mod capture;
mod debug;
mod peak_detection;
mod signal_processing;

pub use audio::{AudioConfig, SquelchConfig};
pub use capture::CaptureConfig;
pub use debug::DebugConfig;
use num::integer::gcd;
pub use peak_detection::{
    AveragingConfig, CfarConfig, ExponentialSmoothingConfig, MovingAverageConfig,
    MultiFrameAveragingConfig, MultiFrameConfig, NoiseFloorConfig, PeakDetectionConfig, WindowType,
    WindowingConfig,
};
pub use signal_processing::{FrequencyTrackingConfig, SignalProcessingConfig};

use crate::core::bands::Band;

/// Configuration for scanning operations
#[derive(Clone)]
pub struct ScanningConfig {
    // Core scanning settings
    pub band: Band,
    pub duration: u64,
    pub samp_rate: f64,
    pub sdr_gain: f64,
    pub scanning_windows: Option<usize>,

    // Grouped configurations
    pub audio: AudioConfig,
    pub peak_detection: PeakDetectionConfig,
    pub signal_processing: SignalProcessingConfig,
    pub capture: CaptureConfig,
    pub debug: DebugConfig,
}

impl Default for ScanningConfig {
    fn default() -> Self {
        Self {
            band: Band::Fm,
            duration: 3,
            samp_rate: 2_000_000.0,
            sdr_gain: 24.0,
            scanning_windows: None,
            audio: AudioConfig::default(),
            peak_detection: PeakDetectionConfig::default(),
            signal_processing: SignalProcessingConfig::default(),
            capture: CaptureConfig::default(),
            debug: DebugConfig::default(),
        }
    }
}

impl ScanningConfig {
    /// Calculate optimal rational resampler ratios for converting from input_rate to
    /// audio_sample_rate Returns (interpolation, decimation) factors for the rational resampler
    pub fn calculate_resampler_ratios(&self, input_rate: f32) -> (usize, usize) {
        let target_rate = self.audio.sample_rate as f32;

        // Find the best rational approximation using continued fractions
        // For efficiency, we'll use a simpler approach: scale by 1000 and find GCD
        let scaled_target = (target_rate * 1000.0).round() as u32;
        let scaled_input = (input_rate * 1000.0).round() as u32;

        // Calculate GCD to reduce the fraction
        let gcd_value = gcd(scaled_target, scaled_input);
        let interp = (scaled_target / gcd_value) as usize;
        let deci = (scaled_input / gcd_value) as usize;

        // Ensure the ratios are reasonable (not too large)
        if interp > 10000 || deci > 10000 {
            // Fall back to a simpler approximation
            let simplified_ratio = (target_rate / input_rate * 1000.0).round() as usize;
            (simplified_ratio, 1000)
        } else {
            (interp, deci)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resampler_ratios_exact_case() {
        let config = ScanningConfig {
            audio: AudioConfig {
                sample_rate: 48000,
                ..Default::default()
            },
            ..Default::default()
        };

        // Test the exact case from our FM demodulation: 153846.15 Hz -> 48000 Hz
        let input_rate = 153846.15;
        let (interp, deci) = config.calculate_resampler_ratios(input_rate);

        // Should get 312:1000 ratio (or equivalent reduced fraction)
        let actual_output = input_rate * (interp as f32 / deci as f32);
        let error = (actual_output - 48000.0).abs();

        assert!(
            error < 1.0,
            "Resampling error should be < 1 Hz, got {:.1} Hz",
            error
        );
        assert_eq!(interp, 312);
        assert_eq!(deci, 1000);
    }

    #[test]
    fn test_resampler_ratios_common_cases() {
        let config = ScanningConfig {
            audio: AudioConfig {
                sample_rate: 48000,
                ..Default::default()
            },
            ..Default::default()
        };

        // Test 44.1 kHz -> 48 kHz (common audio conversion)
        let (interp, deci) = config.calculate_resampler_ratios(44100.0);
        let actual_output = 44100.0 * (interp as f32 / deci as f32);
        let error = (actual_output - 48000.0).abs();
        assert!(
            error < 10.0,
            "44.1->48 kHz error should be < 10 Hz, got {:.1} Hz",
            error
        );

        // Test 96 kHz -> 48 kHz (simple 2:1 ratio)
        let (interp, deci) = config.calculate_resampler_ratios(96000.0);
        let actual_output = 96000.0 * (interp as f32 / deci as f32);
        let error = (actual_output - 48000.0).abs();
        assert!(
            error < 1.0,
            "96->48 kHz error should be < 1 Hz, got {:.1} Hz",
            error
        );
    }

    #[test]
    fn test_resampler_ratios_different_target_rates() {
        // Test with 44.1 kHz target
        let config_44k = ScanningConfig {
            audio: AudioConfig {
                sample_rate: 44100,
                ..Default::default()
            },
            ..Default::default()
        };

        let (interp, deci) = config_44k.calculate_resampler_ratios(48000.0);
        let actual_output = 48000.0 * (interp as f32 / deci as f32);
        let error = (actual_output - 44100.0).abs();
        assert!(
            error < 10.0,
            "48->44.1 kHz error should be < 10 Hz, got {:.1} Hz",
            error
        );

        // Test with 96 kHz target
        let config_96k = ScanningConfig {
            audio: AudioConfig {
                sample_rate: 96000,
                ..Default::default()
            },
            ..Default::default()
        };

        let (interp, deci) = config_96k.calculate_resampler_ratios(48000.0);
        let actual_output = 48000.0 * (interp as f32 / deci as f32);
        let error = (actual_output - 96000.0).abs();
        assert!(
            error < 1.0,
            "48->96 kHz error should be < 1 Hz, got {:.1} Hz",
            error
        );
    }

    #[test]
    fn test_resampler_ratios_fallback() {
        let config = ScanningConfig {
            audio: AudioConfig {
                sample_rate: 48000,
                ..Default::default()
            },
            ..Default::default()
        };

        // Test a case that might produce very large ratios
        let input_rate = 44099.99; // Slightly off from 44.1 kHz
        let (interp, deci) = config.calculate_resampler_ratios(input_rate);

        // Should use fallback if ratios become too large
        assert!(
            interp <= 10000,
            "Interpolation factor should be <= 10000, got {}",
            interp
        );
        assert!(
            deci <= 10000,
            "Decimation factor should be <= 10000, got {}",
            deci
        );

        let actual_output = input_rate * (interp as f32 / deci as f32);
        let error = (actual_output - 48000.0).abs();
        assert!(
            error < 100.0,
            "Fallback error should be reasonable, got {:.1} Hz",
            error
        );
    }

    #[test]
    fn test_resampler_ratios_unity_case() {
        let config = ScanningConfig {
            audio: AudioConfig {
                sample_rate: 48000,
                ..Default::default()
            },
            ..Default::default()
        };

        // Test 1:1 ratio (no resampling needed)
        let (interp, deci) = config.calculate_resampler_ratios(48000.0);
        let actual_output = 48000.0 * (interp as f32 / deci as f32);
        let error = (actual_output - 48000.0).abs();

        assert!(
            error < 0.1,
            "Unity ratio should have minimal error, got {:.3} Hz",
            error
        );
    }

    #[test]
    fn test_resampler_ratios_reduced_fractions() {
        let config = ScanningConfig {
            audio: AudioConfig {
                sample_rate: 48000,
                ..Default::default()
            },
            ..Default::default()
        };

        // Test that fractions are properly reduced
        let (interp, deci) = config.calculate_resampler_ratios(24000.0); // 2:1 ratio

        // Should get a simple ratio like 2:1, not 2000:1000
        assert!(
            interp <= 10 && deci <= 10,
            "Simple ratios should be reduced: got {}:{}",
            interp,
            deci
        );

        let actual_output = 24000.0 * (interp as f32 / deci as f32);
        let error = (actual_output - 48000.0).abs();
        assert!(
            error < 1.0,
            "Simple ratio error should be < 1 Hz, got {:.1} Hz",
            error
        );
    }
}
