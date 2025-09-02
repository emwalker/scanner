use rustradio::Complex;

use crate::file::AudioFileMetadata;
use crate::types::SampleSource;
use crate::{file::IqFileMetadata, types::Result};
use std::io::Read;
use std::{fs::File, io::BufReader};

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
