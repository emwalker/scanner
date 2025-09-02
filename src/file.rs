use crate::types::{Result, SampleSource};
use rustradio::Complex;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
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

    /// Save metadata to a JSON file
    pub fn to_file(&self, metadata_path: &str) -> Result<()> {
        let file = File::create(metadata_path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }
}

/// Metadata for audio fixture files
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AudioFileMetadata {
    pub sample_rate: f32,
    pub duration: f32,
    pub total_samples: usize,
    pub format: String,                    // e.g., "f32_le"
    pub expected_squelch_decision: String, // "audio" or "noise"
    pub description: String,
}

impl AudioFileMetadata {
    pub fn new(
        sample_rate: f32,
        duration: f32,
        total_samples: usize,
        expected_squelch_decision: String,
        description: String,
    ) -> Self {
        Self {
            sample_rate,
            duration,
            total_samples,
            format: "f32_le".to_string(),
            expected_squelch_decision,
            description,
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

/// Capturing wrapper for audio samples - saves demodulated audio to file for squelch testing
pub struct AudioCaptureSink {
    samples_captured: usize,
    max_samples: usize,
    sample_rate: f32,
    output_file: String,
    capture_duration: f64,
    writer: Option<BufWriter<File>>,
}

impl AudioCaptureSink {
    pub fn new(
        output_file: &str,
        sample_rate: f32,
        capture_duration: f64,
    ) -> crate::types::Result<Self> {
        let max_samples = (sample_rate * capture_duration as f32) as usize;

        let file = File::create(output_file)?;
        let writer = BufWriter::new(file);

        debug!(
            message = "Starting audio capture",
            output_file = output_file,
            capture_duration = capture_duration,
            max_samples = max_samples,
            sample_rate = sample_rate
        );

        Ok(Self {
            samples_captured: 0,
            max_samples,
            sample_rate,
            output_file: output_file.to_string(),
            capture_duration,
            writer: Some(writer),
        })
    }

    /// Capture audio samples to file
    pub fn capture_samples(&mut self, samples: &[f32]) -> crate::types::Result<()> {
        if let Some(ref mut writer) = self.writer
            && self.samples_captured < self.max_samples
        {
            let samples_to_capture = (self.max_samples - self.samples_captured).min(samples.len());

            for sample in samples.iter().take(samples_to_capture) {
                writer.write_all(&sample.to_le_bytes())?;
            }

            self.samples_captured += samples_to_capture;
        }

        Ok(())
    }

    /// Save metadata for the captured audio file
    fn save_metadata(&self) -> crate::types::Result<()> {
        let metadata = AudioFileMetadata::new(
            self.sample_rate,
            self.capture_duration as f32,
            self.samples_captured,
            "unknown".to_string(), // Will need to be set by caller based on squelch decision
            format!(
                "Captured demodulated audio at {} Hz for {} seconds",
                self.sample_rate, self.capture_duration
            ),
        );

        let metadata_path = self.output_file.replace(".audio", ".json");
        metadata.to_file(&metadata_path)?;

        debug!(
            message = "Audio metadata saved",
            metadata_file = metadata_path,
            total_samples = self.samples_captured
        );

        Ok(())
    }
}

impl Drop for AudioCaptureSink {
    fn drop(&mut self) {
        if let Some(writer) = self.writer.take() {
            debug!("Saving audio file and metadata");
            if let Err(e) = writer.into_inner() {
                tracing::error!("Failed to flush audio capture file on drop: {}", e);
            }
            if let Err(e) = self.save_metadata() {
                tracing::error!("Failed to save audio metadata on drop: {}", e);
            }
        }
    }
}

/// Capturing wrapper that saves I/Q samples to a file while passing them through
pub struct SampleCaptureSink {
    inner: Box<dyn SampleSource>,
    writer: Option<BufWriter<File>>,
    samples_captured: usize,
    max_samples: usize,
    sample_rate: f64,
    center_frequency: f64,
    output_file: String,
    capture_duration: f64,
}

impl SampleCaptureSink {
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

impl SampleSource for SampleCaptureSink {
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
        if let Some(writer) = self.writer.take() {
            if let Err(e) = writer.into_inner() {
                tracing::error!("Failed to flush I/Q capture file on drop: {}", e);
            }
            if let Err(e) = self.save_metadata() {
                tracing::error!("Failed to save I/Q metadata on drop: {}", e);
            }
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
