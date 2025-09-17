use crate::types::{ModulationType, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use tracing::debug;

/// Metadata for I/Q files
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IqFileMetadata {
    pub sample_rate: f64,
    pub center_frequency: f64,
    pub capture_duration: f64,
    pub total_samples: usize,
    pub format: String,                // e.g., "f32_le_complex"
    pub expected_candidates: Vec<f64>, // Expected station frequencies in Hz

    // Peak detection parameters used during scanning
    pub fft_size: usize,
    pub peak_detection_threshold: f32,
    pub peak_scan_duration: Option<f64>,
    pub driver: String, // SDR driver used (e.g., "driver=sdrplay")
}

impl IqFileMetadata {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sample_rate: f64,
        center_frequency: f64,
        capture_duration: f64,
        total_samples: usize,
        fft_size: usize,
        peak_detection_threshold: f32,
        peak_scan_duration: Option<f64>,
        driver: String,
    ) -> Self {
        Self {
            sample_rate,
            center_frequency,
            capture_duration,
            total_samples,
            format: "f32_le_complex".to_string(),
            expected_candidates: Vec::new(),
            fft_size,
            peak_detection_threshold,
            peak_scan_duration,
            driver,
        }
    }

    /// Save metadata to a JSON file
    pub fn to_file(&self, metadata_path: &str) -> Result<()> {
        let file = File::create(metadata_path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }
}

/// Configuration for AudioCaptureSink creation
pub struct AudioCaptureConfig {
    pub output_dir: String,
    pub sample_rate: f32,
    pub capture_duration: f64,
    pub frequency_hz: f64,
    pub modulation_type: ModulationType,
}

/// Capturing wrapper for audio samples - buffers samples and creates WAV file only after squelch passes
pub struct AudioCaptureSink {
    samples_captured: usize,
    max_samples: usize,
    writer: Option<crate::wave::BufWriter>,
    config: AudioCaptureConfig,
    buffer: Vec<f32>,
    file_created: bool,
}

impl AudioCaptureSink {
    /// Create a new buffered audio capture sink
    pub fn new(config: AudioCaptureConfig) -> crate::types::Result<Self> {
        let max_samples = (config.sample_rate * config.capture_duration as f32) as usize;

        debug!(
            message = "Starting buffered audio capture",
            capture_duration = config.capture_duration,
            max_samples = max_samples,
            sample_rate = config.sample_rate,
            frequency_mhz = config.frequency_hz / 1e6
        );

        Ok(Self {
            samples_captured: 0,
            max_samples,
            writer: None,
            config,
            buffer: Vec::with_capacity(max_samples),
            file_created: false,
        })
    }

    /// Generate filename with frequency formatting and auto-increment
    fn generate_filename(
        output_dir: &str,
        frequency_hz: f64,
        modulation_type: &ModulationType,
    ) -> crate::types::Result<String> {
        // Format frequency with zero-padding and dot separators
        let freq_str = Self::format_frequency(frequency_hz);

        // Format modulation type
        let mod_str = match modulation_type {
            ModulationType::WFM => "wfm",
        };

        // Find next available test number
        let mut test_num = 1;
        loop {
            let filename = format!(
                "{}/{}-{}-{:03}.wav",
                output_dir, freq_str, mod_str, test_num
            );

            if !Path::new(&filename).exists() {
                return Ok(filename);
            }

            test_num += 1;
            if test_num > 999 {
                return Err(crate::types::ScannerError::Custom(
                    "Maximum test number (999) exceeded for frequency".to_string(),
                ));
            }
        }
    }

    /// Format frequency with zero-padding and dot separators
    /// Example: 88900000.0 -> "000.088.900.000Hz"
    fn format_frequency(frequency_hz: f64) -> String {
        let freq_hz = frequency_hz as u64;

        // Zero-pad to 12 digits (supports up to 999.999 GHz)
        let padded = format!("{:012}", freq_hz);

        // Insert dots every 3 digits from the right
        let mut result = String::new();
        for (i, ch) in padded.chars().enumerate() {
            if i > 0 && (padded.len() - i) % 3 == 0 {
                result.push('.');
            }
            result.push(ch);
        }
        result.push_str("Hz");
        result
    }

    /// Buffer audio samples in memory
    pub fn capture_samples(&mut self, samples: &[f32]) -> crate::types::Result<()> {
        if self.samples_captured >= self.max_samples {
            return Ok(()); // Already captured enough samples
        }

        let samples_to_capture = (self.max_samples - self.samples_captured).min(samples.len());

        if self.file_created {
            // File already created, write directly
            if let Some(ref mut writer) = self.writer {
                for sample in samples.iter().take(samples_to_capture) {
                    writer.write_all(&sample.to_le_bytes())?;
                }
            }
        } else {
            // Buffer samples in memory
            self.buffer
                .extend_from_slice(&samples[..samples_to_capture]);
        }

        self.samples_captured += samples_to_capture;
        Ok(())
    }

    /// Create the WAV file and write buffered samples - call when squelch analysis passes
    pub fn create_file_and_flush_buffer(&mut self) -> crate::types::Result<()> {
        if self.file_created {
            return Ok(()); // File already created
        }

        // Generate filename
        let output_file = Self::generate_filename(
            &self.config.output_dir,
            self.config.frequency_hz,
            &self.config.modulation_type,
        )?;

        // Create directory if it doesn't exist
        if let Some(parent) = Path::new(&output_file).parent() {
            fs::create_dir_all(parent)?;
        }

        let file = File::create(&output_file)?;
        let mut writer = crate::wave::BufWriter::new(file);

        // Write WAV header
        writer.write_header(self.config.sample_rate, self.max_samples)?;

        // Write all buffered samples
        for sample in &self.buffer {
            writer.write_all(&sample.to_le_bytes())?;
        }

        debug!(
            message = "Created WAV file and wrote buffered samples",
            output_file = output_file,
            buffered_samples = self.buffer.len(),
            frequency_mhz = self.config.frequency_hz / 1e6
        );

        self.writer = Some(writer);
        self.file_created = true;

        // Clear buffer to save memory
        self.buffer.clear();
        self.buffer.shrink_to_fit();

        Ok(())
    }

    /// Discard buffered samples without creating file - call when squelch analysis fails
    pub fn discard_buffer(&mut self) {
        if !self.file_created {
            debug!(
                message = "Discarding buffered audio samples - squelch failed",
                discarded_samples = self.buffer.len(),
                frequency_mhz = self.config.frequency_hz / 1e6
            );
            self.buffer.clear();
            self.buffer.shrink_to_fit();
        }
    }

    /// Check if file has been created
    pub fn is_file_created(&self) -> bool {
        self.file_created
    }
}

impl Drop for AudioCaptureSink {
    fn drop(&mut self) {
        if !self.buffer.is_empty() && !self.file_created {
            debug!(
                "Dropping buffered audio capture with {} unsaved samples",
                self.buffer.len()
            );
        }
        if let Some(writer) = self.writer.take() {
            debug!("Finalizing WAV file");
            if let Err(e) = writer.into_inner() {
                tracing::error!("Failed to flush audio capture file on drop: {}", e);
            }
        }
    }
}

/// Audio capture block that saves audio samples to file while passing them through unchanged
pub struct AudioCaptureBlock {
    input: rustradio::stream::ReadStream<rustradio::Float>,
    output: rustradio::stream::WriteStream<rustradio::Float>,
    audio_capturer: Option<AudioCaptureSink>,
}

impl AudioCaptureBlock {
    pub fn new(
        input: rustradio::stream::ReadStream<rustradio::Float>,
        audio_capturer: Option<AudioCaptureSink>,
    ) -> (Self, rustradio::stream::ReadStream<rustradio::Float>) {
        let (output, output_stream) = rustradio::stream::WriteStream::new();

        let block = Self {
            input,
            output,
            audio_capturer,
        };

        (block, output_stream)
    }

    /// Get mutable reference to the audio capturer for external coordination
    pub fn audio_capturer_mut(&mut self) -> Option<&mut AudioCaptureSink> {
        self.audio_capturer.as_mut()
    }
}

impl rustradio::block::BlockName for AudioCaptureBlock {
    fn block_name(&self) -> &str {
        "AudioCaptureBlock"
    }
}

impl rustradio::block::BlockEOF for AudioCaptureBlock {
    fn eof(&mut self) -> bool {
        self.input.eof()
    }
}

impl rustradio::block::Block for AudioCaptureBlock {
    fn work(&mut self) -> rustradio::Result<rustradio::block::BlockRet<'_>> {
        let (input_buf, _) = self.input.read_buf()?;
        let input_samples = input_buf.slice();

        if input_samples.is_empty() {
            return Ok(rustradio::block::BlockRet::WaitForStream(&self.input, 1));
        }

        // Get output buffer
        let mut output_buf = self.output.write_buf()?;
        let to_copy = input_samples.len().min(output_buf.len());

        // Pass through all samples unchanged
        output_buf.slice()[..to_copy].copy_from_slice(&input_samples[..to_copy]);

        // Capture samples if requested
        if let Some(ref mut capturer) = self.audio_capturer
            && let Err(e) = capturer.capture_samples(&input_samples[..to_copy])
        {
            tracing::debug!("Audio capture error: {}", e);
        }

        input_buf.consume(to_copy);
        output_buf.produce(to_copy, &[]);

        Ok(rustradio::block::BlockRet::Again)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModulationType;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_config(output_dir: &str) -> AudioCaptureConfig {
        AudioCaptureConfig {
            output_dir: output_dir.to_string(),
            sample_rate: 48000.0,
            capture_duration: 1.0,    // 1 second
            frequency_hz: 88900000.0, // 88.9 MHz
            modulation_type: ModulationType::WFM,
        }
    }

    #[test]
    fn test_buffered_capture_and_file_creation() -> crate::types::Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(temp_dir.path().to_str().unwrap());

        let mut sink = AudioCaptureSink::new(config)?;

        // Initially no file should be created
        assert!(!sink.is_file_created());

        // Buffer some samples
        let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        sink.capture_samples(&samples)?;

        // Still no file created
        assert!(!sink.is_file_created());

        // Create file and flush buffer (simulating squelch pass)
        sink.create_file_and_flush_buffer()?;

        // Now file should be created
        assert!(sink.is_file_created());

        // Check that file exists
        let files: Vec<_> = fs::read_dir(temp_dir.path()).unwrap().collect();
        assert_eq!(files.len(), 1);

        let file_path = files[0].as_ref().unwrap().path();
        assert!(file_path.to_str().unwrap().contains("000.088.900.000Hz"));
        assert!(file_path.to_str().unwrap().ends_with(".wav"));

        Ok(())
    }

    #[test]
    fn test_buffered_capture_discard() -> crate::types::Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(temp_dir.path().to_str().unwrap());

        let mut sink = AudioCaptureSink::new(config)?;

        // Buffer some samples
        let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        sink.capture_samples(&samples)?;

        // Discard buffer (simulating squelch fail)
        sink.discard_buffer();

        // No file should be created
        assert!(!sink.is_file_created());

        // Check that no files exist
        let files: Vec<_> = fs::read_dir(temp_dir.path()).unwrap().collect();
        assert_eq!(files.len(), 0);

        Ok(())
    }

    #[test]
    fn test_capture_after_file_creation() -> crate::types::Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(temp_dir.path().to_str().unwrap());

        let mut sink = AudioCaptureSink::new(config)?;

        // Buffer some samples
        let samples1 = vec![0.1, 0.2];
        sink.capture_samples(&samples1)?;

        // Create file and flush buffer
        sink.create_file_and_flush_buffer()?;

        // Add more samples after file creation (should write directly)
        let samples2 = vec![0.3, 0.4];
        sink.capture_samples(&samples2)?;

        // File should still exist
        assert!(sink.is_file_created());

        Ok(())
    }
}
