use crate::types::{Result, SampleSource};
use rustradio::Complex;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
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
}

impl IqFileMetadata {
    pub fn new(
        sample_rate: f64,
        center_frequency: f64,
        capture_duration: f64,
        total_samples: usize,
    ) -> Self {
        Self {
            sample_rate,
            center_frequency,
            capture_duration,
            total_samples,
            format: "f32_le_complex".to_string(),
            expected_candidates: Vec::new(),
        }
    }

    /// Load metadata from a JSON file
    pub fn from_file(metadata_path: &str) -> Result<Self> {
        let file = File::open(metadata_path)?;
        let metadata: IqFileMetadata = serde_json::from_reader(file)?;
        Ok(metadata)
    }

    /// Save metadata to a JSON file
    pub fn to_file(&self, metadata_path: &str) -> Result<()> {
        let file = File::create(metadata_path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }
}

/// File-based sample source for testing
#[allow(dead_code)]
pub struct FileSampleSource {
    reader: BufReader<File>,
    sample_rate: f64,
    center_frequency: f64,
    samples_remaining: usize,
}

impl FileSampleSource {
    #[allow(dead_code)]
    pub fn new(file_path: &str, sample_rate: f64, center_frequency: f64) -> Result<Self> {
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

    /// Create FileSampleSource from I/Q file and its metadata
    #[allow(dead_code)]
    pub fn from_metadata(iq_file_path: &str) -> Result<Self> {
        // Derive metadata file path by replacing .iq extension with .json
        let metadata_path = iq_file_path.replace(".iq", ".json");
        let metadata = IqFileMetadata::from_file(&metadata_path)?;

        Self::new(
            iq_file_path,
            metadata.sample_rate,
            metadata.center_frequency,
        )
    }
}

/// Test helper to load both I/Q file and metadata in one call
#[cfg(test)]
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

/// Capturing wrapper that saves I/Q samples to a file while passing them through
pub struct CapturingSampleSource {
    inner: Box<dyn SampleSource>,
    writer: Option<BufWriter<File>>,
    samples_captured: usize,
    max_samples: usize,
    sample_rate: f64,
    center_frequency: f64,
    output_file: String,
    capture_duration: f64,
}

impl CapturingSampleSource {
    pub fn new(
        inner: Box<dyn SampleSource>,
        output_file: &str,
        capture_duration: f64,
    ) -> Result<Self> {
        let sample_rate = inner.sample_rate();
        let center_frequency = inner.center_frequency();
        let max_samples = (sample_rate * capture_duration) as usize;

        let file = File::create(output_file)?;
        let writer = BufWriter::new(file);

        debug!(
            message = "Starting I/Q capture",
            output_file = output_file,
            capture_duration = capture_duration,
            max_samples = max_samples,
            sample_rate = sample_rate
        );

        Ok(Self {
            inner,
            writer: Some(writer),
            samples_captured: 0,
            max_samples,
            sample_rate,
            center_frequency,
            output_file: output_file.to_string(),
            capture_duration,
        })
    }

    /// Save metadata for the captured I/Q file
    fn save_metadata(&self) -> Result<()> {
        let metadata = IqFileMetadata::new(
            self.sample_rate,
            self.center_frequency,
            self.capture_duration,
            self.samples_captured,
        );

        let metadata_path = self.output_file.replace(".iq", ".json");
        metadata.to_file(&metadata_path)?;

        debug!(
            message = "I/Q metadata saved",
            metadata_file = metadata_path,
            total_samples = self.samples_captured
        );

        Ok(())
    }
}

impl SampleSource for CapturingSampleSource {
    fn read_samples(&mut self, buffer: &mut [Complex]) -> Result<usize> {
        let samples_read = self.inner.read_samples(buffer)?;

        // Capture samples if we haven't reached the limit
        if let Some(ref mut writer) = self.writer
            && self.samples_captured < self.max_samples
        {
            let samples_to_capture = (self.max_samples - self.samples_captured).min(samples_read);

            for sample in buffer.iter().take(samples_to_capture) {
                // Write as f32 little-endian pairs (real, imaginary)
                writer.write_all(&sample.re.to_le_bytes())?;
                writer.write_all(&sample.im.to_le_bytes())?;
            }

            self.samples_captured += samples_to_capture;

            // Close file when done capturing
            if self.samples_captured >= self.max_samples
                && let Some(writer) = self.writer.take()
            {
                writer.into_inner().map_err(|e| {
                    crate::types::ScannerError::IqCapture(format!(
                        "Failed to flush capture file: {}",
                        e
                    ))
                })?;
                debug!(
                    message = "I/Q capture complete",
                    samples_captured = self.samples_captured
                );
                self.save_metadata()?;
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

    fn deactivate(&mut self) -> Result<()> {
        // Flush and close capture file if still open
        if let Some(writer) = self.writer.take() {
            writer.into_inner().map_err(|e| {
                crate::types::ScannerError::IqCapture(format!(
                    "Failed to close capture file: {}",
                    e
                ))
            })?;
            debug!(
                message = "I/Q capture finished",
                samples_captured = self.samples_captured
            );
            self.save_metadata()?;
        }

        self.inner.deactivate()
    }

    fn device_args(&self) -> &str {
        "test"
    }

    fn peak_scan_duration(&self) -> f64 {
        1.0
    }
}
