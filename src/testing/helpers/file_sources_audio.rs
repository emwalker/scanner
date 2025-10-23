use std::{
    fs::File,
    io::{BufReader, Read},
};

use crate::core::types::Result;

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
