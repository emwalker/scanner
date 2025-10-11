//! Backend trait for SDR hardware abstraction

use super::{DeviceId, DeviceInfo, DeviceTrait, streaming::StreamingDevice};
use crate::core::types::Result;

/// Abstraction over different SDR backend implementations
///
/// This trait allows the application to work with multiple SDR backends
/// (SoapySDR, Seify, native drivers) through a unified interface.
///
/// # Examples
///
/// ```no_run
/// use scanner::hardware::{Backend, Soapy};
///
/// let backend = Soapy;
/// let devices = backend.enumerate_devices()?;
/// let device = backend.open_device(&devices[0].id)?;
/// # Ok::<(), scanner::core::types::ScannerError>(())
/// ```
pub trait Backend: Send + Sync {
    /// Enumerate all devices this backend can access
    ///
    /// Returns a list of available devices that can be opened with this backend.
    fn enumerate_devices(&self) -> Result<Vec<DeviceInfo>>;

    /// Open device for rustradio graph-based processing
    fn open_device(&self, id: &DeviceId) -> Result<Box<dyn DeviceTrait>>;

    /// Open device for direct sample streaming (used by device worker subprocess)
    fn open_streaming_device(&self, id: &DeviceId) -> Result<Box<dyn StreamingDevice>>;

    /// Backend identifier (e.g., "SoapySDR", "Seify", "rtl-sdr-rs")
    fn name(&self) -> &str;
}
