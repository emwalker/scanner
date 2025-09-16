use hound::{SampleFormat, WavReader};
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use tracing::debug;

/// Load WAV file and return audio samples as normalized f32 values
pub fn load_file<P: AsRef<Path>>(path: P) -> crate::types::Result<Vec<f32>> {
    let mut reader = WavReader::open(path.as_ref()).map_err(|e| {
        crate::types::ScannerError::Custom(format!("Failed to open WAV file: {}", e))
    })?;

    let spec = reader.spec();
    debug!(
        path = %path.as_ref().display(),
        channels = spec.channels,
        sample_rate = spec.sample_rate,
        bits_per_sample = spec.bits_per_sample,
        sample_format = ?spec.sample_format,
        "Loading WAV file"
    );

    let mut samples = Vec::new();
    match spec.sample_format {
        SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                samples.push(sample.map_err(|e| {
                    crate::types::ScannerError::Custom(format!("Failed to read sample: {}", e))
                })?);
            }
        }
        SampleFormat::Int => {
            let max_val = (1i32 << (spec.bits_per_sample - 1)) as f32;
            for sample in reader.samples::<i32>() {
                let s = sample.map_err(|e| {
                    crate::types::ScannerError::Custom(format!("Failed to read sample: {}", e))
                })?;
                samples.push(s as f32 / max_val);
            }
        }
    }

    // Convert to mono if stereo by averaging channels
    if spec.channels == 2 {
        let mono_samples: Vec<f32> = samples
            .chunks(2)
            .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
            .collect();
        samples = mono_samples;
    }

    debug!(sample_count = samples.len(), "Loaded WAV samples");
    Ok(samples)
}

/// Wrapper around std::io::BufWriter for WAV file operations
pub struct BufWriter {
    inner: io::BufWriter<File>,
}

impl BufWriter {
    /// Create a new WAV BufWriter
    pub fn new(file: File) -> Self {
        Self {
            inner: io::BufWriter::new(file),
        }
    }

    /// Get the inner BufWriter, consuming the wrapper
    pub fn into_inner(self) -> io::Result<File> {
        self.inner.into_inner().map_err(|e| e.into_error())
    }

    /// Write WAV file header (RIFF format with 32-bit IEEE float)
    pub fn write_header(
        &mut self,
        sample_rate: f32,
        total_samples: usize,
    ) -> crate::types::Result<()> {
        let sample_rate = sample_rate as u32;
        let channels = 1u16;
        let bits_per_sample = 32u16;
        let bytes_per_sample = bits_per_sample / 8;
        let byte_rate = sample_rate * channels as u32 * bytes_per_sample as u32;
        let block_align = channels * bytes_per_sample;
        let data_size = total_samples as u32 * bytes_per_sample as u32;
        let file_size = 36 + data_size;

        // RIFF header
        self.inner.write_all(b"RIFF")?;
        self.inner.write_all(&file_size.to_le_bytes())?;
        self.inner.write_all(b"WAVE")?;

        // fmt chunk
        self.inner.write_all(b"fmt ")?;
        self.inner.write_all(&16u32.to_le_bytes())?; // chunk size
        self.inner.write_all(&3u16.to_le_bytes())?; // format (3 = IEEE float)
        self.inner.write_all(&channels.to_le_bytes())?;
        self.inner.write_all(&sample_rate.to_le_bytes())?;
        self.inner.write_all(&byte_rate.to_le_bytes())?;
        self.inner.write_all(&block_align.to_le_bytes())?;
        self.inner.write_all(&bits_per_sample.to_le_bytes())?;

        // data chunk header
        self.inner.write_all(b"data")?;
        self.inner.write_all(&data_size.to_le_bytes())?;

        Ok(())
    }
}

impl Write for BufWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.inner.write_all(buf)
    }
}

impl From<BufWriter> for io::BufWriter<File> {
    fn from(wrapper: BufWriter) -> Self {
        wrapper.inner
    }
}
