use crate::types::{Result, ScanningConfig};
use rustradio::Complex;
use tokio::sync::broadcast;

pub trait Segment {
    fn audio_subscriber(&self) -> broadcast::Receiver<Complex>;
}

pub trait Device {
    fn tune(&self, config: &ScanningConfig, center_freq: f64) -> Result<Box<dyn Segment>>;
}
