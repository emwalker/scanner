use crate::audio::quality::AudioQuality;
use crate::core::bands::Band;

#[derive(Debug, Clone)]
pub enum WindowType {
    Rectangular,
    Hamming,
    Hanning,
    BlackmanHarris,
}

/// Configuration for scanning operations
#[derive(Clone)]
pub struct ScanningConfig {
    pub audio_buffer_size: u32,
    pub audio_sample_rate: u32,
    pub band: Band,
    pub capture_audio_duration: f64,
    pub capture_audio: Option<String>,
    pub capture_duration: f64,
    pub capture_iq: Option<String>,
    pub debug_pipeline: bool,
    pub duration: u64,
    pub sdr_gain: f64,
    pub scanning_windows: Option<usize>,
    pub fft_size: usize,
    pub peak_detection_threshold: f32,
    pub peak_scan_duration: f64,
    pub print_candidates: bool,
    pub samp_rate: f64,
    pub squelch_learning_duration: f32,

    // Frequency tracking configuration
    pub frequency_tracking_method: String,
    pub tracking_accuracy: f64,
    pub disable_frequency_tracking: bool,

    // Spectral analysis configuration
    pub spectral_threshold: f32,

    // AGC and window configuration
    pub agc_settling_time: f64,
    pub window_overlap: f64,

    // Sample batching configuration
    pub packet_size: usize,

    // Squelch configuration
    pub disable_squelch: bool,
    pub squelch_threshold: AudioQuality,
    // IF AGC configuration
    pub disable_if_agc: bool,
    // Audio quality analyzer
    pub audio_analyzer: crate::audio::quality::AudioAnalyzer,

    // Signal averaging and smoothing configuration
    pub enable_exponential_smoothing: bool,
    pub smoothing_alpha: f32,
    pub enable_multi_frame_averaging: bool,
    pub averaging_frames: usize,
    pub enable_coherent_integration: bool,
    pub enable_moving_average_filter: bool,
    pub moving_average_window_size: usize,

    // CFAR detection configuration
    pub enable_cfar_detection: bool,
    pub cfar_threshold_factor: f32,
    pub cfar_guard_cells: usize,
    pub cfar_reference_cells: usize,
    pub cfar_false_alarm_rate: f32,

    // Dynamic noise floor estimation configuration
    pub enable_dynamic_noise_floor: bool,
    pub noise_floor_percentile: f32,
    pub noise_floor_history_frames: usize,
    pub noise_floor_threshold_multiplier: f32,
    pub noise_floor_adaptation_rate: f32,

    // Spectral preprocessing configuration
    pub enable_windowing: bool,
    pub window_type: WindowType,
    pub zero_padding_factor: usize,
    pub window_overlap_percent: f32,

    // Multi-frame integration configuration
    pub enable_multi_frame_integration: bool,
    pub multi_frame_history_frames: usize,
    pub multi_frame_confirmation_threshold: usize,
    pub multi_frame_frequency_tolerance: f64,
    pub multi_frame_max_age: f64,
}

impl Default for ScanningConfig {
    fn default() -> Self {
        Self {
            audio_buffer_size: 4096,
            audio_sample_rate: 48000,
            band: Band::Fm,
            capture_audio: None,
            capture_audio_duration: 3.0,
            capture_duration: 2.0,
            capture_iq: None,
            debug_pipeline: false,
            duration: 3,
            sdr_gain: 24.0,
            scanning_windows: None,
            fft_size: 1024,
            peak_detection_threshold: 1.0,
            peak_scan_duration: 1.5,
            print_candidates: false,
            samp_rate: 2_000_000.0,
            squelch_learning_duration: 1.0,

            // Frequency tracking defaults
            frequency_tracking_method: "pll".to_string(),
            tracking_accuracy: 5000.0,
            disable_frequency_tracking: false,

            // Spectral analysis defaults
            spectral_threshold: 0.2,

            // AGC and window defaults
            agc_settling_time: 0.45,
            window_overlap: 0.75,

            // Sample batching defaults
            packet_size: 16384,

            // Squelch defaults
            disable_squelch: false,
            squelch_threshold: AudioQuality::Moderate,
            // IF AGC defaults
            disable_if_agc: false,
            // Audio analyzer default (pass-through for testing)
            audio_analyzer: crate::audio::quality::AudioAnalyzer::pass_through(),

            // Signal averaging defaults (enabled by default for improved performance)
            enable_exponential_smoothing: true,
            smoothing_alpha: 0.3, // 30% smoothing factor
            enable_multi_frame_averaging: true,
            averaging_frames: 3, // Average over 3 frames (reduced from 8 to prevent excessive frame discarding)
            enable_coherent_integration: true,
            enable_moving_average_filter: true,
            moving_average_window_size: 5, // 5-point moving average

            // CFAR detection defaults (enabled by default for improved performance)
            enable_cfar_detection: true,
            cfar_threshold_factor: 10.0, // 10 dB above noise floor
            cfar_guard_cells: 10,        // Guard cells around target
            cfar_reference_cells: 50,    // Reference cells for noise estimation
            cfar_false_alarm_rate: 0.01, // 1% false alarm rate

            // Dynamic noise floor estimation defaults (disabled by default until properly tuned)
            enable_dynamic_noise_floor: false,
            noise_floor_percentile: 0.25, // 25th percentile (less conservative)
            noise_floor_history_frames: 8, // Track last 8 frames (faster adaptation)
            noise_floor_threshold_multiplier: 1.6, // 1.6x above noise floor (more aggressive)
            noise_floor_adaptation_rate: 0.35, // 35% adaptation rate (faster learning)

            // Spectral preprocessing defaults
            enable_windowing: true,
            window_type: WindowType::BlackmanHarris,
            zero_padding_factor: 2,
            window_overlap_percent: 0.0,

            // Multi-frame integration defaults (disabled by default until properly tuned)
            enable_multi_frame_integration: false,
            multi_frame_history_frames: 5, // Track last 5 frames
            multi_frame_confirmation_threshold: 2, // Default threshold (adaptive)
            multi_frame_frequency_tolerance: 25_000.0, // 25 kHz tolerance
            multi_frame_max_age: 10.0,     // 10 second timeout
        }
    }
}

impl ScanningConfig {
    /// Calculate optimal rational resampler ratios for converting from input_rate to audio_sample_rate
    /// Returns (interpolation, decimation) factors for the rational resampler
    pub fn calculate_resampler_ratios(&self, input_rate: f32) -> (usize, usize) {
        let target_rate = self.audio_sample_rate as f32;

        // Find the best rational approximation using continued fractions
        // For efficiency, we'll use a simpler approach: scale by 1000 and find GCD
        let scaled_target = (target_rate * 1000.0).round() as u32;
        let scaled_input = (input_rate * 1000.0).round() as u32;

        // Calculate GCD to reduce the fraction
        let gcd = Self::gcd(scaled_target, scaled_input);
        let interp = (scaled_target / gcd) as usize;
        let deci = (scaled_input / gcd) as usize;

        // Ensure the ratios are reasonable (not too large)
        if interp > 10000 || deci > 10000 {
            // Fall back to a simpler approximation
            let simplified_ratio = (target_rate / input_rate * 1000.0).round() as usize;
            (simplified_ratio, 1000)
        } else {
            (interp, deci)
        }
    }

    /// Calculate Greatest Common Divisor using Euclidean algorithm
    fn gcd(mut a: u32, mut b: u32) -> u32 {
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd_basic_cases() {
        assert_eq!(ScanningConfig::gcd(48, 18), 6);
        assert_eq!(ScanningConfig::gcd(1000, 312), 8);
        assert_eq!(ScanningConfig::gcd(312, 1000), 8);
        assert_eq!(ScanningConfig::gcd(100, 25), 25);
        assert_eq!(ScanningConfig::gcd(17, 13), 1); // coprime numbers
    }

    #[test]
    fn test_gcd_edge_cases() {
        assert_eq!(ScanningConfig::gcd(0, 5), 5);
        assert_eq!(ScanningConfig::gcd(5, 0), 5);
        assert_eq!(ScanningConfig::gcd(1, 1), 1);
        assert_eq!(ScanningConfig::gcd(1000, 1000), 1000);
    }

    #[test]
    fn test_resampler_ratios_exact_case() {
        let config = ScanningConfig {
            audio_sample_rate: 48000,
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
            audio_sample_rate: 48000,
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
            audio_sample_rate: 44100,
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
            audio_sample_rate: 96000,
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
            audio_sample_rate: 48000,
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
            audio_sample_rate: 48000,
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
            audio_sample_rate: 48000,
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
