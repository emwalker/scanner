use rustradio::Complex;

use crate::{core::types::Result, hardware::DeviceId};

/// Actual hardware configuration values after device configuration
///
/// Hardware may not support exact requested values, so we return actual values.
#[derive(Debug, Clone)]
pub struct ActualConfig {
    pub freq_hz: f64,
    pub sample_rate: f64,
    pub gain_db: f64,
}

/// Device streaming interface for subprocess workers
///
/// This trait provides direct sample streaming without rustradio graph dependency.
/// Used by device worker subprocesses to stream I/Q data to parent process.
pub trait StreamingDevice: Send {
    /// Stable device identifier
    fn device_id(&self) -> &DeviceId;

    /// Number of RX channels (e.g., RSPduo has 2)
    fn channels(&self) -> usize;

    /// Configure RX channel with frequency, sample rate, and gain
    ///
    /// Returns actual values set by hardware (may differ from requested).
    fn configure_rx(
        &mut self,
        channel: usize,
        freq: f64,
        rate: f64,
        gain: f64,
    ) -> Result<ActualConfig>;

    /// Start streaming on a channel
    ///
    /// Must call configure_rx() first. Channel must not already be streaming.
    fn start_stream(&mut self, channel: usize) -> Result<()>;

    /// Read samples from a streaming channel
    ///
    /// Returns number of samples read (may be less than buffer size).
    /// timeout_us: microsecond timeout for read operation
    fn read_samples(
        &mut self,
        channel: usize,
        buffer: &mut [Complex],
        timeout_us: i64,
    ) -> Result<usize>;

    /// Stop streaming on a channel
    fn stop_stream(&mut self, channel: usize) -> Result<()>;
}
