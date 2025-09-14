use crate::fm;
use thiserror::Error;

pub trait ConsoleWriter {
    fn write_info(&self, message: &str);
    fn write_debug(&self, message: &str);
}

#[derive(Error, Debug)]
pub enum ScannerError {
    #[error(transparent)]
    Sdr(#[from] soapysdr::Error),
    #[error("Error: {0}")]
    Custom(String),
    #[error(transparent)]
    Audio(#[from] cpal::SupportedStreamConfigsError),
    #[error(transparent)]
    AudioBuild(#[from] cpal::BuildStreamError),
    #[error(transparent)]
    AudioPlay(#[from] cpal::PlayStreamError),
    #[error(transparent)]
    AudioDevice(#[from] cpal::DefaultStreamConfigError),
    #[error(transparent)]
    AudioDeviceName(#[from] cpal::DeviceNameError),
    #[error(transparent)]
    RustRadio(#[from] rustradio::Error),
    #[error(transparent)]
    Stderr(#[from] log::SetLoggerError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("I/Q capture error: {0}")]
    IqCapture(String),
    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, ScannerError>;

pub trait Logger {
    fn init(&self) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct Peak {
    pub frequency_hz: f64,
    pub magnitude: f32,
}

#[derive(Debug)]
pub enum Candidate {
    Fm(fm::Candidate),
}

/// Represents a successfully detected and demodulated signal
#[derive(Debug, Clone)]
pub struct Signal {
    /// Center frequency of the signal in Hz
    pub frequency_hz: f64,
    /// Signal strength/power measurement
    pub signal_strength: f32,
    /// Estimated bandwidth of the signal in Hz
    pub bandwidth_hz: f32,
    /// Type of modulation detected
    pub modulation: ModulationType,
    /// Audio sample rate for this signal
    pub audio_sample_rate: u32,
    /// Timestamp when signal was detected
    pub detected_at: std::time::SystemTime,
    /// Duration of analysis period that led to detection
    pub analysis_duration_ms: u32,
    /// Center frequency used by SDR during detection (needed for audio processing offset calculation)
    pub detection_center_freq: f64,
    /// Audio quality assessment
    pub audio_quality: crate::audio_quality::AudioQuality,
}

#[derive(Debug, Clone)]
pub enum ModulationType {
    WFM,
    // Future: NFM, Am, Digital, etc.
}

impl Signal {
    pub fn new_fm(
        frequency_hz: f64,
        signal_strength: f32,
        bandwidth_hz: f32,
        audio_sample_rate: u32,
        analysis_duration_ms: u32,
        detection_center_freq: f64,
        audio_quality: crate::audio_quality::AudioQuality,
    ) -> Self {
        Self {
            frequency_hz,
            signal_strength,
            bandwidth_hz,
            modulation: ModulationType::WFM,
            audio_sample_rate,
            detected_at: std::time::SystemTime::now(),
            analysis_duration_ms,
            detection_center_freq,
            audio_quality,
        }
    }
}

impl Candidate {
    pub fn frequency_hz(&self) -> f64 {
        match self {
            Candidate::Fm(candidate) => candidate.frequency_hz,
        }
    }

    pub fn analyze(
        &self,
        config: &ScanningConfig,
        sdr_rx: tokio::sync::broadcast::Receiver<rustradio::Complex>,
        center_freq: f64,
        signal_tx: std::sync::mpsc::SyncSender<Signal>,
        device: &crate::soapy::Device,
    ) -> Result<()> {
        match self {
            Candidate::Fm(candidate) => {
                candidate.analyze(config, sdr_rx, center_freq, signal_tx, device)
            }
        }
    }
}

// Frequency bands for scanning
use clap::ValueEnum;

#[derive(ValueEnum, Copy, Clone, Debug)]
pub enum Band {
    /// FM broadcast band (88-108 MHz)
    Fm,
    /// VHF aircraft band (108-137 MHz)
    Aircraft,
    /// 2-meter amateur band (144-148 MHz)
    Ham2m,
    /// NOAA weather radio (162-163 MHz)
    Weather,
    /// Marine VHF band (156-162 MHz)
    Marine,
}

impl std::fmt::Display for Band {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Band::Fm => write!(f, "fm"),
            Band::Aircraft => write!(f, "aircraft"),
            Band::Ham2m => write!(f, "ham2m"),
            Band::Weather => write!(f, "weather"),
            Band::Marine => write!(f, "marine"),
        }
    }
}

impl Band {
    pub fn frequency_range(&self) -> (f64, f64) {
        match self {
            Band::Fm => (88.0e6, 108.0e6),
            Band::Aircraft => (108.0e6, 137.0e6),
            Band::Ham2m => (144.0e6, 148.0e6),
            Band::Weather => (162.0e6, 163.0e6),
            Band::Marine => (156.0e6, 162.0e6),
        }
    }

    pub fn windows(&self, sample_rate: f64, overlap: f64) -> Vec<f64> {
        let (start_freq, end_freq) = self.frequency_range();
        let usable_bandwidth = sample_rate * 0.8; // Use 80% of bandwidth to avoid edge effects
        let step_size = usable_bandwidth * (1.0 - overlap); // Step size based on overlap percentage

        let mut windows = Vec::new();
        let mut center_freq = start_freq + (usable_bandwidth / 2.0);

        while center_freq - (usable_bandwidth / 2.0) < end_freq {
            windows.push(center_freq);
            center_freq += step_size;
        }

        windows
    }
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
    pub peak_scan_duration: Option<f64>,
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
    // Squelch configuration
    pub disable_squelch: bool,
    // IF AGC configuration
    pub disable_if_agc: bool,
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
            peak_scan_duration: None,
            print_candidates: false,
            samp_rate: 2_000_000.0,
            squelch_learning_duration: 2.0,

            // Frequency tracking defaults
            frequency_tracking_method: "pll".to_string(),
            tracking_accuracy: 5000.0,
            disable_frequency_tracking: false,

            // Spectral analysis defaults
            spectral_threshold: 0.2,

            // AGC and window defaults
            agc_settling_time: 3.0,
            window_overlap: 0.75,
            // Squelch defaults
            disable_squelch: false,
            // IF AGC defaults
            disable_if_agc: false,
        }
    }
}

impl ScanningConfig {
    /// Calculate optimal rational resampler ratios for converting from input_rate to audio_sample_rate
    /// Returns (interpolation, decimation) factors for the rational resampler
    pub fn calculate_resampler_ratios(&self, input_rate: f32) -> (usize, usize) {
        let target_rate = self.audio_sample_rate as f32;
        let ratio = target_rate / input_rate;

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
            let simplified_ratio = (ratio * 1000.0).round() as usize;
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

#[derive(ValueEnum, Copy, Clone, Debug)]
pub enum Format {
    /// JSON structured logging format
    Json,
    /// Simple text logging format
    Text,
    /// Standard log format with timestamps and levels
    Log,
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Format::Json => write!(f, "json"),
            Format::Text => write!(f, "text"),
            Format::Log => write!(f, "log"),
        }
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
