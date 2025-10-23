//! Backend trait for SDR hardware abstraction

use super::{DeviceInfo, DeviceTrait, pool::TunerId, streaming::StreamingDevice};
use crate::core::types::Result;

/// Abstraction over different SDR backend implementations
///
/// This trait allows the application to work with multiple SDR backends
/// (SoapySDR, Seify, native drivers) through a unified interface.
///
/// # Examples
///
/// ```no_run
/// use scanner::hardware::{Backend, DeviceId, Soapy, pool::TunerId};
///
/// let backend = Soapy;
/// let devices = backend.enumerate_devices()?;
/// let device_id = DeviceId::from_serial("sdrplay", "2301034E34");
/// let tuner_id = TunerId::new(device_id, 0);
/// let device = backend.open_tuner(&tuner_id)?;
/// # Ok::<(), scanner::core::types::ScannerError>(())
/// ```
pub trait Backend: Send + Sync {
    /// Enumerate all devices this backend can access
    ///
    /// Returns a list of available devices that can be opened with this backend.
    fn enumerate_devices(&self) -> Result<Vec<DeviceInfo>>;

    /// Open specific tuner for rustradio graph-based processing
    fn open_tuner(&self, tuner_id: &TunerId) -> Result<Box<dyn DeviceTrait>>;

    /// Open specific tuner for direct sample streaming (used by device worker subprocess)
    fn open_streaming_tuner(&self, tuner_id: &TunerId) -> Result<Box<dyn StreamingDevice>>;

    /// Backend identifier (e.g., "SoapySDR", "Seify", "rtl-sdr-rs")
    fn name(&self) -> &str;
}
