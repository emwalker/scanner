//! Core testing utilities and helper functions

use rustradio::Complex;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::{
    file::IqFileMetadata,
    types::{Format, Result, ScannerError, ScanningConfig},
};
use std::f32::consts::PI;
use std::io::Read;
use std::{fs::File, io::BufReader};
use tokio::sync::broadcast::error::TryRecvError;

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

        let angular_freq = 2.0 * PI * self.frequency_offset / self.sample_rate as f32;
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
            if self.phase > 2.0 * PI {
                self.phase -= 2.0 * PI;
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

    let peaks = crate::peaks::collect_peaks_from_source(config, &mut sample_source)?;

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
    format: Format,
) -> crate::types::Result<crate::logging::LogBuffer> {
    use tracing::Level;
    use tracing_subscriber::FmtSubscriber;

    let level = if verbose { Level::DEBUG } else { Level::INFO };
    let log_buffer = crate::logging::LogBuffer::default();

    match format {
        Format::Json => {
            let subscriber = FmtSubscriber::builder()
                .json()
                .with_max_level(level)
                .with_writer(log_buffer.clone())
                .finish();
            tracing::subscriber::set_global_default(subscriber)?;
        }
        Format::Text => {
            let subscriber = FmtSubscriber::builder()
                .with_max_level(level)
                .with_writer(log_buffer.clone())
                .without_time()
                .with_target(false)
                .with_level(false)
                .finish();
            tracing::subscriber::set_global_default(subscriber)?;
        }
        Format::Log => {
            let subscriber = FmtSubscriber::builder()
                .with_max_level(level)
                .with_writer(log_buffer.clone())
                .with_target(false)
                .finish();
            tracing::subscriber::set_global_default(subscriber)?;
        }
    }

    Ok(log_buffer)
}

/// Test helper that runs a function with captured logging and returns both result and logs
pub fn with_captured_logs<F, R>(
    verbose: bool,
    format: Format,
    test_fn: F,
) -> crate::types::Result<(R, String)>
where
    F: FnOnce() -> crate::types::Result<R>,
{
    let log_buffer = init_test_logging(verbose, format)?;
    let result = test_fn()?;
    let logs = log_buffer.into_string();
    Ok((result, logs))
}

/// Test helper function to assert that a classifier correctly classifies audio samples
///
/// # Arguments
/// * `classifier` - An instantiated classifier implementing the Classifier trait
/// * `overrides` - List of (filename, expected_quality) tuples for cases where the classifier
///   is expected to deviate from the training dataset. An empty list means
///   the classifier should have perfect accuracy against the training data.
///
/// # Usage
/// A poor classifier will need many overrides to pass the test, while a good classifier
/// will need minimal or no overrides. This captures the current behavior and protects
/// against regressions.
pub fn assert_classifies_audio(
    classifier: &dyn crate::audio_quality::Classifier,
    overrides: &[(&str, crate::audio_quality::AudioQuality)],
) -> crate::types::Result<()> {
    use std::collections::HashMap;

    // Convert overrides to a HashMap for quick lookup
    let override_map: HashMap<&str, crate::audio_quality::AudioQuality> =
        overrides.iter().cloned().collect();

    // Get the training dataset
    let training_data = crate::audio_quality::training_dataset();

    let mut total_tests = 0;
    let mut correct_classifications = 0;
    let mut failed_files = Vec::new();
    let mut unnecessary_overrides = Vec::new();

    for (filename, training_quality) in training_data.iter() {
        // Check for unnecessary overrides (override matches training dataset expectation)
        if let Some(override_quality) = override_map.get(filename)
            && override_quality == training_quality
        {
            unnecessary_overrides.push((filename.to_string(), *training_quality));
        }

        // Check if there's an override for this file
        let expected_quality = override_map.get(filename).unwrap_or(training_quality);

        // Construct the path to the audio file
        let wav_path = std::path::PathBuf::from("tests/data/audio/quality").join(filename);

        // Skip files that don't exist (similar to training logic)
        if !wav_path.exists() {
            debug!(filename = %filename, "Audio file not found, skipping test");
            continue;
        }

        // Load the audio file
        let audio_samples = match crate::wave::load_file(&wav_path) {
            Ok(samples) => samples,
            Err(e) => {
                debug!(filename = %filename, error = %e, "Failed to load audio file, skipping");
                continue;
            }
        };

        // Analyze with the classifier
        match classifier.analyze(&audio_samples, 48000.0) {
            Ok(result) => {
                total_tests += 1;

                if result.quality == *expected_quality {
                    correct_classifications += 1;
                    debug!(
                        filename = %filename,
                        expected = %expected_quality.to_human_string(),
                        actual = %result.quality.to_human_string(),
                        confidence = result.confidence,
                        "Classification correct"
                    );
                } else {
                    failed_files.push((
                        filename.to_string(),
                        *expected_quality,
                        result.quality,
                        result.confidence,
                    ));
                    debug!(
                        filename = %filename,
                        expected = %expected_quality.to_human_string(),
                        actual = %result.quality.to_human_string(),
                        confidence = result.confidence,
                        "Classification mismatch"
                    );
                }
            }
            Err(e) => {
                debug!(filename = %filename, error = %e, "Classification failed");
                failed_files.push((
                    filename.to_string(),
                    *expected_quality,
                    crate::audio_quality::AudioQuality::Unknown,
                    0.0,
                ));
            }
        }
    }

    // Report results
    debug!(
        classifier = classifier.name(),
        total_tests = total_tests,
        correct = correct_classifications,
        accuracy_percent = if total_tests > 0 {
            (correct_classifications as f32 / total_tests as f32) * 100.0
        } else {
            0.0
        },
        "Classification test completed"
    );

    // Check for unnecessary overrides first
    if !unnecessary_overrides.is_empty() {
        let mut error_message = format!(
            "Classifier '{}' has {} unnecessary override(s) that match the training dataset:\n",
            classifier.name(),
            unnecessary_overrides.len()
        );

        for (filename, quality) in unnecessary_overrides {
            error_message.push_str(&format!(
                "  {} - Override specifies {}, but training dataset already expects {}\n",
                filename,
                quality.to_human_string(),
                quality.to_human_string()
            ));
        }

        error_message
            .push_str("\nRemove these unnecessary overrides to keep the override list minimal.\n");

        return Err(ScannerError::Custom(error_message));
    }

    // Assert that all classifications were correct
    if !failed_files.is_empty() {
        let mut error_message = format!(
            "Classifier '{}' failed {} out of {} tests:\n",
            classifier.name(),
            failed_files.len(),
            total_tests
        );

        for (filename, expected, actual, confidence) in failed_files {
            error_message.push_str(&format!(
                "  {} - Expected: {}, Got (possibly via an override): {} (confidence: {:.2})\n",
                filename,
                expected.to_human_string(),
                actual.to_human_string(),
                confidence
            ));
        }

        return Err(ScannerError::Custom(error_message));
    }

    Ok(())
}

/// Adapter to make SDR broadcast receiver compatible with SampleSource trait
/// This allows the unified peak detection code to work with both testing sources and real SDR streams
pub struct SdrStreamSource {
    sdr_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
    sample_rate: f64,
    center_frequency: f64,
    peak_scan_duration: f64,
    timeout_us: u64,
}

impl SdrStreamSource {
    pub fn new(
        sdr_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
        sample_rate: f64,
        center_frequency: f64,
        peak_scan_duration: f64,
    ) -> Self {
        Self {
            sdr_rx,
            sample_rate,
            center_frequency,
            peak_scan_duration,
            timeout_us: 100, // 100μs timeout between read attempts
        }
    }
}

impl SampleSource for SdrStreamSource {
    fn read_samples(&mut self, buffer: &mut [rustradio::Complex]) -> crate::types::Result<usize> {
        use std::thread;
        use std::time::Duration;

        let mut samples_read = 0;
        while samples_read < buffer.len() {
            match self.sdr_rx.try_recv() {
                Ok(packet) => {
                    let samples = packet.as_slice();
                    let to_copy = samples.len().min(buffer.len() - samples_read);
                    buffer[samples_read..samples_read + to_copy]
                        .copy_from_slice(&samples[..to_copy]);
                    samples_read += to_copy;
                }
                Err(TryRecvError::Empty) => {
                    if samples_read > 0 {
                        break;
                    }
                    thread::sleep(Duration::from_micros(self.timeout_us));
                    continue;
                }
                Err(TryRecvError::Lagged(_)) => {
                    continue;
                }
                Err(TryRecvError::Closed) => {
                    break;
                }
            }
        }
        Ok(samples_read)
    }

    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    fn center_frequency(&self) -> f64 {
        self.center_frequency
    }

    fn peak_scan_duration(&self) -> f64 {
        self.peak_scan_duration
    }

    fn deactivate(&mut self) -> crate::types::Result<()> {
        // Nothing to deactivate for SDR stream source
        Ok(())
    }

    fn device_args(&self) -> &str {
        ""
    }
}
