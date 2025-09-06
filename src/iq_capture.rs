use crate::file::IqFileMetadata;
use rustradio::{
    Complex, Result,
    block::{Block, BlockEOF, BlockName, BlockRet},
    stream::{ReadStream, WriteStream},
};
use std::fs::File;
use std::io::{BufWriter, Write};
use tracing::debug;

/// I/Q capture block that saves samples to file while passing them through unchanged
pub struct IqCaptureBlock {
    input: ReadStream<Complex>,
    output: WriteStream<Complex>,
    writer: Option<BufWriter<File>>,
    samples_captured: usize,
    max_samples: usize,

    // Metadata for the capture file
    output_file: String,
    sample_rate: f64,
    center_frequency: f64,
    capture_duration: f64,
    fft_size: usize,
    peak_detection_threshold: f32,
    peak_scan_duration: Option<f64>,
    driver: String,
}

impl IqCaptureBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: ReadStream<Complex>,
        output_file: String,
        sample_rate: f64,
        center_frequency: f64,
        capture_duration: f64,
        fft_size: usize,
        peak_detection_threshold: f32,
        peak_scan_duration: Option<f64>,
        driver: String,
    ) -> Result<(Self, ReadStream<Complex>)> {
        let max_samples = (sample_rate * capture_duration) as usize;

        let file = File::create(&output_file)?;
        let writer = BufWriter::new(file);

        let (output, output_stream) = WriteStream::new();

        debug!(
            message = "Starting I/Q capture",
            output_file = output_file,
            capture_duration = capture_duration,
            max_samples = max_samples,
            sample_rate = sample_rate,
            center_freq_mhz = center_frequency / 1e6
        );

        let block = Self {
            input,
            output,
            writer: Some(writer),
            samples_captured: 0,
            max_samples,
            output_file,
            sample_rate,
            center_frequency,
            capture_duration,
            fft_size,
            peak_detection_threshold,
            peak_scan_duration,
            driver,
        };

        Ok((block, output_stream))
    }

    /// Save metadata for the captured I/Q file
    fn save_metadata(&self) -> Result<()> {
        let metadata = IqFileMetadata::new(
            self.sample_rate,
            self.center_frequency,
            self.capture_duration,
            self.samples_captured,
            self.fft_size,
            self.peak_detection_threshold,
            self.peak_scan_duration,
            self.driver.clone(),
        );

        let metadata_path = self.output_file.replace(".iq", ".json");
        if let Err(e) = metadata.to_file(&metadata_path) {
            debug!("Failed to save I/Q metadata: {}", e);
        }

        debug!(
            message = "I/Q metadata saved",
            metadata_file = metadata_path,
            total_samples = self.samples_captured
        );

        Ok(())
    }
}

impl BlockName for IqCaptureBlock {
    fn block_name(&self) -> &str {
        "IqCaptureBlock"
    }
}

impl BlockEOF for IqCaptureBlock {
    fn eof(&mut self) -> bool {
        self.input.eof()
    }
}

impl Block for IqCaptureBlock {
    fn work(&mut self) -> Result<BlockRet<'_>> {
        let (input_buf, _) = self.input.read_buf()?;
        let input_samples = input_buf.slice();

        if input_samples.is_empty() {
            return Ok(BlockRet::WaitForStream(&self.input, 1));
        }

        // Get output buffer
        let mut output_buf = self.output.write_buf()?;
        let to_copy = input_samples.len().min(output_buf.len());

        // Pass through all samples unchanged
        output_buf.slice()[..to_copy].copy_from_slice(&input_samples[..to_copy]);

        // Capture samples to file if we haven't reached the limit
        if let Some(ref mut writer) = self.writer
            && self.samples_captured < self.max_samples
        {
            let samples_to_capture = (self.max_samples - self.samples_captured).min(to_copy);

            for sample in input_samples.iter().take(samples_to_capture) {
                // Write as f32 little-endian pairs (real, imaginary)
                if let Err(e) = writer.write_all(&sample.re.to_le_bytes()) {
                    debug!("I/Q capture write error (real): {}", e);
                    break;
                }
                if let Err(e) = writer.write_all(&sample.im.to_le_bytes()) {
                    debug!("I/Q capture write error (imag): {}", e);
                    break;
                }
            }

            self.samples_captured += samples_to_capture;

            // Close file when done capturing
            if self.samples_captured >= self.max_samples
                && let Some(writer) = self.writer.take()
            {
                if let Err(e) = writer.into_inner() {
                    debug!("Failed to flush I/Q capture file: {}", e);
                } else {
                    debug!(
                        message = "I/Q capture complete",
                        samples_captured = self.samples_captured
                    );
                    if let Err(e) = self.save_metadata() {
                        debug!("Failed to save I/Q metadata: {}", e);
                    }
                }
            }
        }

        input_buf.consume(to_copy);
        output_buf.produce(to_copy, &[]);

        Ok(BlockRet::Again)
    }
}

impl Drop for IqCaptureBlock {
    fn drop(&mut self) {
        if let Some(writer) = self.writer.take() {
            debug!("Saving I/Q file and metadata on drop");
            if let Err(e) = writer.into_inner() {
                tracing::error!("Failed to flush I/Q capture file on drop: {}", e);
            } else if let Err(e) = self.save_metadata() {
                tracing::error!("Failed to save I/Q metadata on drop: {}", e);
            }
        }
    }
}
