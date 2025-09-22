//! SDR (Software Defined Radio) abstraction layer

pub mod sample_source;

use crate::types::{Result, ScanningConfig};
use rustradio::Complex;
use tokio::sync::broadcast;

// Re-export commonly used items
pub use sample_source::{FileSampleSource, MockSampleSource, SampleSource, SdrStreamSource};

pub trait Segment {
    fn audio_subscriber(&self) -> broadcast::Receiver<Complex>;
}

pub trait Device {
    fn tune(&self, config: &ScanningConfig, center_freq: f64) -> Result<Box<dyn Segment>>;
}
