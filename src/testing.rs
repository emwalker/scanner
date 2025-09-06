#![allow(dead_code)]
use rustradio::Complex;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::{
    file::IqFileMetadata,
    types::{Result, ScanningConfig},
};
use std::io::Read;
use std::{fs::File, io::BufReader};

/// Metadata for audio fixture files
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AudioFileMetadata {
    pub sample_rate: f32,
    pub squelch_learning_duration: f32, // Renamed from duration for clarity
    pub total_samples: usize,
    pub format: String,                    // e.g., "f32_le"
    pub expected_squelch_decision: String, // "audio" or "noise"
    pub description: String,
    pub frequency_hz: f64, // The frequency being monitored
    pub center_freq: f64,  // The SDR center frequency
    pub driver: String,    // SDR driver used (e.g., "driver=sdrplay")
}

impl AudioFileMetadata {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sample_rate: f32,
        squelch_learning_duration: f32,
        total_samples: usize,
        expected_squelch_decision: String,
        description: String,
        frequency_hz: f64,
        center_freq: f64,
        driver: String,
    ) -> Self {
        Self {
            sample_rate,
            squelch_learning_duration,
            total_samples,
            format: "f32_le".to_string(),
            expected_squelch_decision,
            description,
            frequency_hz,
            center_freq,
            driver,
        }
    }

    /// Load metadata from a JSON file
    #[cfg(test)]
    pub fn from_file(metadata_path: &str) -> Result<Self> {
        let file = File::open(metadata_path)?;
        let metadata: AudioFileMetadata = serde_json::from_reader(file)?;
        Ok(metadata)
    }

    /// Save metadata to a JSON file
    pub fn to_file(&self, metadata_path: &str) -> Result<()> {
        let file = File::create(metadata_path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }
}
/// Abstraction for sources of I/Q samples
pub trait SampleSource {
    /// Read samples into the provided buffer
    /// Returns the number of samples actually read
    fn read_samples(&mut self, buffer: &mut [Complex]) -> Result<usize>;

    /// Get the configured sample rate
    fn sample_rate(&self) -> f64;

    /// Get the configured center frequency  
    fn center_frequency(&self) -> f64;

    /// Clean up resources when done
    fn deactivate(&mut self) -> Result<()>;

    fn peak_scan_duration(&self) -> f64;

    fn device_args(&self) -> &str;
}

impl IqFileMetadata {
    /// Load metadata from a JSON file
    pub fn from_file(metadata_path: &str) -> Result<Self> {
        let file = File::open(metadata_path)?;
        let metadata: IqFileMetadata = serde_json::from_reader(file)?;
        Ok(metadata)
    }
}

/// File-based sample source for testing
pub struct FileSampleSource {
    reader: BufReader<File>,
    sample_rate: f64,
    center_frequency: f64,
    samples_remaining: usize,
}

impl FileSampleSource {
    pub fn new(file_path: &str, sample_rate: f64, center_frequency: f64) -> Result<Self> {
        use std::fs::File;

        let file = File::open(file_path)?;

        // Get file size to estimate number of samples (8 bytes per complex sample: f32 real + f32 imag)
        let file_size = file.metadata()?.len() as usize;
        let samples_remaining = file_size / 8; // 2 f32s per complex sample

        Ok(Self {
            reader: BufReader::new(file),
            sample_rate,
            center_frequency,
            samples_remaining,
        })
    }
}

impl SampleSource for FileSampleSource {
    fn read_samples(&mut self, buffer: &mut [Complex]) -> Result<usize> {
        let samples_to_read = buffer.len().min(self.samples_remaining);
        if samples_to_read == 0 {
            return Ok(0);
        }

        // Read raw bytes for f32 pairs
        let bytes_to_read = samples_to_read * 8; // 8 bytes per complex sample
        let mut byte_buffer = vec![0u8; bytes_to_read];

        match self.reader.read_exact(&mut byte_buffer) {
            Ok(_) => {
                // Convert bytes to Complex<f32> samples
                for (i, sample) in buffer.iter_mut().take(samples_to_read).enumerate() {
                    let real_bytes = &byte_buffer[i * 8..i * 8 + 4];
                    let imag_bytes = &byte_buffer[i * 8 + 4..i * 8 + 8];

                    let real = f32::from_le_bytes([
                        real_bytes[0],
                        real_bytes[1],
                        real_bytes[2],
                        real_bytes[3],
                    ]);
                    let imag = f32::from_le_bytes([
                        imag_bytes[0],
                        imag_bytes[1],
                        imag_bytes[2],
                        imag_bytes[3],
                    ]);

                    *sample = Complex::new(real, imag);
                }

                self.samples_remaining -= samples_to_read;
                Ok(samples_to_read)
            }
            Err(e) => Err(e.into()),
        }
    }

    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    fn center_frequency(&self) -> f64 {
        self.center_frequency
    }

    fn deactivate(&mut self) -> Result<()> {
        // Nothing to deactivate for file source
        Ok(())
    }

    fn device_args(&self) -> &str {
        ""
    }

    fn peak_scan_duration(&self) -> f64 {
        1.0
    }
}

/// Mock sample source for testing that generates a simple sine wave
pub struct MockSampleSource {
    sample_rate: f64,
    center_frequency: f64,
    samples_generated: usize,
    max_samples: usize,
    phase: f32,
    frequency_offset: f32, // Hz offset from center frequency
}

impl MockSampleSource {
    pub fn new(
        sample_rate: f64,
        center_frequency: f64,
        max_samples: usize,
        signal_freq_offset: f32,
    ) -> Self {
        Self {
            sample_rate,
            center_frequency,
            samples_generated: 0,
            max_samples,
            phase: 0.0,
            frequency_offset: signal_freq_offset,
        }
    }
}

impl SampleSource for MockSampleSource {
    fn read_samples(&mut self, buffer: &mut [Complex]) -> Result<usize> {
        let samples_to_generate = buffer.len().min(self.max_samples - self.samples_generated);
        if samples_to_generate == 0 {
            return Ok(0);
        }

        let angular_freq =
            2.0 * std::f32::consts::PI * self.frequency_offset / self.sample_rate as f32;
        debug!(
            "MockSampleSource: freq_offset={}, angular_freq={}",
            self.frequency_offset, angular_freq
        );

        for sample in buffer.iter_mut().take(samples_to_generate) {
            // Generate a single complex exponential at the specified frequency offset
            // This creates a pure tone at center_freq + frequency_offset
            // e^(j*phase) = cos(phase) + j*sin(phase)

            *sample = Complex::new(
                self.phase.cos() * 0.5, // I component
                self.phase.sin() * 0.5, // Q component
            );

            // Update phase
            self.phase += angular_freq;

            // Wrap phase to avoid accumulation errors
            if self.phase > 2.0 * std::f32::consts::PI {
                self.phase -= 2.0 * std::f32::consts::PI;
            }
        }

        self.samples_generated += samples_to_generate;
        Ok(samples_to_generate)
    }

    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    fn center_frequency(&self) -> f64 {
        self.center_frequency
    }

    fn deactivate(&mut self) -> Result<()> {
        Ok(())
    }

    fn device_args(&self) -> &str {
        "test"
    }

    fn peak_scan_duration(&self) -> f64 {
        1.0
    }
}

/// Test helper to load both I/Q file and metadata in one call
pub fn load_iq_fixture(iq_file_path: &str) -> Result<(FileSampleSource, IqFileMetadata)> {
    // Derive metadata file path by replacing .iq extension with .json
    let metadata_path = iq_file_path.replace(".iq", ".json");
    let metadata = IqFileMetadata::from_file(&metadata_path)?;

    let file_source = FileSampleSource::new(
        iq_file_path,
        metadata.sample_rate,
        metadata.center_frequency,
    )?;

    Ok((file_source, metadata))
}

/// File-based audio source for testing squelch functionality
pub struct AudioFileSource {
    reader: BufReader<File>,
    #[allow(dead_code)]
    sample_rate: f32,
    samples_remaining: usize,
}

impl AudioFileSource {
    pub fn new(file_path: &str, sample_rate: f32) -> Result<Self> {
        let file = File::open(file_path)?;

        // Get file size to estimate number of samples (4 bytes per f32 sample)
        let file_size = file.metadata()?.len() as usize;
        let samples_remaining = file_size / 4; // 4 bytes per f32 sample

        Ok(Self {
            reader: BufReader::new(file),
            sample_rate,
            samples_remaining,
        })
    }

    /// Read audio samples from file
    pub fn read_audio_samples(&mut self, buffer: &mut [f32]) -> Result<usize> {
        let samples_to_read = buffer.len().min(self.samples_remaining);
        if samples_to_read == 0 {
            return Ok(0);
        }

        // Read raw bytes for f32 samples
        let bytes_to_read = samples_to_read * 4; // 4 bytes per f32 sample
        let mut byte_buffer = vec![0u8; bytes_to_read];

        match self.reader.read_exact(&mut byte_buffer) {
            Ok(_) => {
                // Convert bytes to f32 samples
                for (i, sample) in buffer.iter_mut().take(samples_to_read).enumerate() {
                    let sample_bytes = &byte_buffer[i * 4..i * 4 + 4];
                    *sample = f32::from_le_bytes([
                        sample_bytes[0],
                        sample_bytes[1],
                        sample_bytes[2],
                        sample_bytes[3],
                    ]);
                }

                self.samples_remaining -= samples_to_read;
                Ok(samples_to_read)
            }
            Err(e) => Err(e.into()),
        }
    }
}

/// Test helper to load both audio file and metadata in one call
#[cfg(test)]
pub fn load_audio_fixture(audio_file_path: &str) -> Result<(AudioFileSource, AudioFileMetadata)> {
    // Derive metadata file path by replacing .audio extension with .json
    let metadata_path = audio_file_path.replace(".audio", ".json");
    let metadata = AudioFileMetadata::from_file(&metadata_path)?;
    let audio_source = AudioFileSource::new(audio_file_path, metadata.sample_rate)?;
    Ok((audio_source, metadata))
}

/// Test framework for verifying frequency translation behavior
#[derive(Debug)]
pub struct FrequencyTest {
    pub test_name: String,
    pub sdr_center_freq: f64,
    pub target_station_freq: f64,
    pub expected_offset: f64,
}

impl FrequencyTest {
    pub fn new(test_name: &str, sdr_center_freq: f64, target_station_freq: f64) -> Self {
        let expected_offset = target_station_freq - sdr_center_freq;
        Self {
            test_name: test_name.to_string(),
            sdr_center_freq,
            target_station_freq,
            expected_offset,
        }
    }

    /// Simulate the frequency translation that would occur in the FreqXlatingFir
    pub fn simulate_frequency_translation(&self) -> f64 {
        // This is what the FreqXlatingFir receives
        let frequency_offset = self.target_station_freq - self.sdr_center_freq;
        info!(
            "[{}] SDR Center: {:.1} MHz, Target: {:.1} MHz, Offset: {:.1} kHz",
            self.test_name,
            self.sdr_center_freq / 1e6,
            self.target_station_freq / 1e6,
            frequency_offset / 1e3
        );
        frequency_offset
    }
}

/// Create test scenarios that match the current scanning behavior
pub fn create_frequency_test_scenarios() -> Vec<FrequencyTest> {
    vec![
        // Scenario 1: --stations mode (direct tuning)
        FrequencyTest::new("stations_mode_88.9", 88.9e6, 88.9e6),
        // Scenario 2: --band fm mode with realistic window centers
        // With 1 MHz sample rate, windows should be about 1 MHz wide
        // Realistic scenario: 88.9 MHz in a window centered at 89.1 MHz (200 kHz offset)
        FrequencyTest::new("band_mode_88.9_window_89.1", 89.1e6, 88.9e6),
        // Scenario 3: Different realistic window center
        // 88.9 MHz in a window centered at 88.7 MHz (200 kHz offset)
        FrequencyTest::new("band_mode_88.9_window_88.7", 88.7e6, 88.9e6),
        // Scenario 4: Edge case within Nyquist limit
        // 88.9 MHz in a window centered at 89.3 MHz (400 kHz offset, still within 500 kHz limit)
        FrequencyTest::new("band_mode_88.9_window_89.3", 89.3e6, 88.9e6),
    ]
}

/// Helper function to verify FreqXlatingFir parameters
pub fn verify_freq_xlating_params(center_freq: f64, tune_freq: f64) -> (f64, bool) {
    let frequency_offset = tune_freq - center_freq;
    let sample_rate = 1_000_000.0;
    let nyquist_limit = sample_rate / 2.0;
    let is_valid = frequency_offset.abs() <= nyquist_limit;

    info!(
        "FreqXlatingFir params: center={:.3} MHz, tune={:.3} MHz, offset={:.1} kHz, valid={}",
        center_freq / 1e6,
        tune_freq / 1e6,
        frequency_offset / 1e3,
        is_valid
    );

    (frequency_offset, is_valid)
}

/// Create a test candidate to verify DSP pipeline behavior
#[cfg(test)]
pub fn create_test_candidate(freq: f64, center_freq: f64) -> (crate::fm::Candidate, f64, f64) {
    let candidate = crate::fm::Candidate {
        frequency_hz: freq,
        peak_count: 1,
        max_magnitude: 1000.0,
        avg_magnitude: 500.0,
        signal_strength: "Test".to_string(),
    };
    (candidate, center_freq, freq)
}

/// Test helper for isolating peak detection with known I/Q signals
pub fn test_peak_detection_isolated(
    iq_file_path: &str,
    expected_peaks: &[f64],
    config: &ScanningConfig,
    debug: bool,
) -> crate::types::Result<TestPeakResult> {
    let (mut sample_source, metadata) = load_iq_fixture(iq_file_path)?;

    if debug {
        debug!(
            message = "Peak detection test started",
            iq_file = iq_file_path,
            sample_rate_mhz = metadata.sample_rate / 1e6,
            center_frequency_mhz = metadata.center_frequency / 1e6,
            expected_peaks_count = expected_peaks.len()
        );

        for (idx, expected_freq) in expected_peaks.iter().enumerate() {
            debug!(
                message = "Expected peak",
                peak_index = idx,
                frequency_mhz = expected_freq / 1e6
            );
        }
    }

    let peaks = crate::fm::collect_peaks_from_source(config, &mut sample_source)?;

    if debug {
        debug!(
            message = "Peak detection completed",
            peaks_found = peaks.len()
        );

        for (idx, peak) in peaks.iter().enumerate() {
            debug!(
                message = "Peak found",
                peak_index = idx,
                frequency_mhz = peak.frequency_hz / 1e6,
                magnitude = peak.magnitude
            );
        }
    }

    // Analyze peak detection accuracy
    let mut found_expected = Vec::new();
    let tolerance_hz = 50_000.0; // 50 kHz tolerance

    for expected_freq in expected_peaks {
        let found = peaks
            .iter()
            .any(|peak| (peak.frequency_hz - expected_freq).abs() <= tolerance_hz);
        found_expected.push(found);

        if debug {
            debug!(
                message = "Peak detection validation",
                expected_frequency_mhz = expected_freq / 1e6,
                found = found,
                tolerance_khz = tolerance_hz / 1e3
            );
        }
    }

    let all_expected_found = found_expected.iter().all(|&found| found);

    Ok(TestPeakResult {
        peaks,
        metadata,
        expected_found: found_expected,
        all_expected_found,
    })
}

/// Test helper for isolating frequency translation pipeline
pub fn test_frequency_translation_isolated(
    center_freq: f64,
    tune_freq: f64,
    debug: bool,
) -> FrequencyTranslationResult {
    let frequency_offset = tune_freq - center_freq;

    // Check if offset is within reasonable bounds
    let sample_rate = 1_000_000.0; // 1 MHz default
    let nyquist_limit = sample_rate / 2.0;
    let within_nyquist = frequency_offset.abs() <= nyquist_limit;

    // Check filter bandwidth compatibility
    let channel_bandwidth = 150_000.0; // Current filter bandwidth
    let filter_cutoff = channel_bandwidth / 2.0;

    if debug {
        debug!(
            message = "Frequency translation test",
            center_frequency_mhz = center_freq / 1e6,
            tune_frequency_mhz = tune_freq / 1e6,
            frequency_offset_khz = frequency_offset / 1e3,
            nyquist_limit_khz = nyquist_limit / 1e3,
            within_nyquist = within_nyquist,
            filter_bandwidth_khz = channel_bandwidth / 1e3
        );
    }

    FrequencyTranslationResult {
        center_freq,
        tune_freq,
        frequency_offset,
        within_nyquist,
        filter_bandwidth: channel_bandwidth,
        filter_cutoff,
        translation_valid: within_nyquist,
    }
}

/// Complete end-to-end pipeline test with debugging
pub fn test_complete_pipeline_debug(
    iq_file_path: &str,
    expected_station_freq: f64,
    scanning_mode: ScanningMode,
    config: &ScanningConfig,
) -> crate::types::Result<PipelineTestResult> {
    info!("\n=== Complete Pipeline Debug Test ===");

    // Step 1: Test peak detection
    info!("\nStep 1: Peak Detection");
    let peak_result =
        test_peak_detection_isolated(iq_file_path, &[expected_station_freq], config, true)?;

    // Step 2: Test candidate creation
    info!("\nStep 2: Candidate Creation");
    let center_freq = match scanning_mode {
        ScanningMode::Stations(freq) => freq,
        ScanningMode::BandWindow(window_center) => window_center,
    };

    let candidates = crate::fm::find_candidates(&peak_result.peaks, config, center_freq);
    info!("  Using center freq: {:.3} MHz", center_freq / 1e6);
    info!("  Created {} candidates", candidates.len());

    for candidate in &candidates {
        let signal_strength = match candidate {
            crate::types::Candidate::Fm(fm_candidate) => &fm_candidate.signal_strength,
        };
        info!(
            "    {:.3} MHz (strength: {})",
            candidate.frequency_hz() / 1e6,
            signal_strength
        );
    }

    // Step 3: Test frequency translation for target candidate
    info!("\nStep 3: Frequency Translation");
    let mut translation_results = Vec::new();
    let mut target_candidate_found = false;

    for candidate in &candidates {
        let result =
            test_frequency_translation_isolated(center_freq, candidate.frequency_hz(), false);

        // Check if this is our target candidate
        if (candidate.frequency_hz() - expected_station_freq).abs() < 50_000.0 {
            target_candidate_found = true;
            info!(
                "  Target candidate: {:.3} MHz",
                candidate.frequency_hz() / 1e6
            );
            info!("    Offset: {:.1} kHz", result.frequency_offset / 1e3);
            info!("    Translation valid: {}", result.translation_valid);
        }

        translation_results.push(result);
    }

    if !target_candidate_found {
        info!(
            "  ✗ Target station {:.3} MHz not found in candidates!",
            expected_station_freq / 1e6
        );
    }

    Ok(PipelineTestResult {
        peak_result,
        candidates,
        translation_results,
        target_found: target_candidate_found,
        scanning_mode,
    })
}

#[derive(Debug)]
pub struct TestPeakResult {
    pub peaks: Vec<crate::types::Peak>,
    pub metadata: crate::file::IqFileMetadata,
    pub expected_found: Vec<bool>,
    pub all_expected_found: bool,
}

#[derive(Debug)]
pub struct FrequencyTranslationResult {
    pub center_freq: f64,
    pub tune_freq: f64,
    pub frequency_offset: f64,
    pub within_nyquist: bool,
    pub filter_bandwidth: f64,
    pub filter_cutoff: f64,
    pub translation_valid: bool,
}

#[derive(Debug)]
pub struct PipelineTestResult {
    pub peak_result: TestPeakResult,
    pub candidates: Vec<crate::types::Candidate>,
    pub translation_results: Vec<FrequencyTranslationResult>,
    pub target_found: bool,
    pub scanning_mode: ScanningMode,
}

#[derive(Debug, Clone)]
pub enum ScanningMode {
    Stations(f64),   // Direct station frequency
    BandWindow(f64), // Window center frequency
}

/// Test-safe logging initialization that captures logs for analysis
/// Returns a LogBuffer that can be used to retrieve captured log messages
pub fn init_test_logging(
    verbose: bool,
    format: crate::Format,
) -> crate::types::Result<crate::logging::LogBuffer> {
    use tracing::Level;
    use tracing_subscriber::FmtSubscriber;

    let level = if verbose { Level::DEBUG } else { Level::INFO };
    let log_buffer = crate::logging::LogBuffer::default();

    match format {
        crate::Format::Json => {
            let subscriber = FmtSubscriber::builder()
                .json()
                .with_max_level(level)
                .with_writer(log_buffer.clone())
                .finish();
            tracing::subscriber::set_global_default(subscriber).map_err(|_| {
                crate::types::ScannerError::Custom("Failed to set subscriber".to_string())
            })?;
        }
        crate::Format::Text => {
            let subscriber = FmtSubscriber::builder()
                .with_max_level(level)
                .with_writer(log_buffer.clone())
                .without_time()
                .with_target(false)
                .with_level(false)
                .finish();
            tracing::subscriber::set_global_default(subscriber).map_err(|_| {
                crate::types::ScannerError::Custom("Failed to set subscriber".to_string())
            })?;
        }
        crate::Format::Log => {
            let subscriber = FmtSubscriber::builder()
                .with_max_level(level)
                .with_writer(log_buffer.clone())
                .with_target(false)
                .finish();
            tracing::subscriber::set_global_default(subscriber).map_err(|_| {
                crate::types::ScannerError::Custom("Failed to set subscriber".to_string())
            })?;
        }
    }

    Ok(log_buffer)
}

/// Test helper that runs a function with captured logging and returns both result and logs
pub fn with_captured_logs<F, R>(
    verbose: bool,
    format: crate::Format,
    test_fn: F,
) -> crate::types::Result<(R, String)>
where
    F: FnOnce() -> crate::types::Result<R>,
{
    let log_buffer = init_test_logging(verbose, format)?;
    let result = test_fn()?;
    let logs = log_buffer.get_string();
    Ok((result, logs))
}

/// Enhanced pipeline test that captures and returns debug logs
pub fn test_complete_pipeline_with_logs(
    iq_file_path: &str,
    expected_station_freq: f64,
    scanning_mode: ScanningMode,
    config: &ScanningConfig,
) -> crate::types::Result<(PipelineTestResult, String)> {
    with_captured_logs(true, crate::Format::Json, || {
        test_complete_pipeline_debug(
            iq_file_path,
            expected_station_freq,
            scanning_mode.clone(),
            config,
        )
    })
}

/// Test helper for comparing scanning modes with captured logs
pub fn compare_scanning_modes_with_logs(
    iq_file_path: &str,
    station_freq: f64,
    window_center_freq: f64,
    config: &ScanningConfig,
) -> crate::types::Result<(PipelineTestResult, PipelineTestResult, String, String)> {
    // Test stations mode
    let (stations_result, stations_logs) = test_complete_pipeline_with_logs(
        iq_file_path,
        station_freq,
        ScanningMode::Stations(station_freq),
        config,
    )?;

    // Test band window mode
    let (band_result, band_logs) = test_complete_pipeline_with_logs(
        iq_file_path,
        station_freq,
        ScanningMode::BandWindow(window_center_freq),
        config,
    )?;

    Ok((stations_result, band_result, stations_logs, band_logs))
}
