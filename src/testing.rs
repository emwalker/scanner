use crate::types::{Result, SampleSource};
use rustradio::Complex;
use std::fs::File;
use std::io::{BufReader, Read};

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
        let file = File::open(file_path).map_err(|e| {
            crate::types::ScannerError::Custom(format!(
                "Failed to open I/Q file {}: {}",
                file_path, e
            ))
        })?;

        // Get file size to estimate number of samples (8 bytes per complex sample: f32 real + f32 imag)
        let file_size = file
            .metadata()
            .map_err(|e| {
                crate::types::ScannerError::Custom(format!("Failed to get file metadata: {}", e))
            })?
            .len() as usize;
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
            Err(e) => Err(crate::types::ScannerError::Custom(format!(
                "File read error: {}",
                e
            ))),
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
}
