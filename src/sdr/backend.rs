//! Backend trait for SDR hardware abstraction

use super::{DeviceId, DeviceInfo, DeviceTrait};
use crate::types::Result;

/// Abstraction over different SDR backend implementations
///
/// This trait allows the application to work with multiple SDR backends
/// (SoapySDR, Seify, native drivers) through a unified interface.
///
/// # Examples
///
/// ```no_run
/// use scanner::sdr::{Backend, Soapy};
///
/// let backend = Soapy;
/// let devices = backend.enumerate_devices()?;
/// let device = backend.open_device(&devices[0].id)?;
/// # Ok::<(), scanner::types::ScannerError>(())
/// ```
pub trait Backend: Send + Sync {
    /// Enumerate all devices this backend can access
    ///
    /// Returns a list of available devices that can be opened with this backend.
    fn enumerate_devices(&self) -> Result<Vec<DeviceInfo>>;

    /// Open a specific device by ID
    ///
    /// The device ID should come from a previous call to `enumerate_devices()`.
    fn open_device(&self, id: &DeviceId) -> Result<Box<dyn DeviceTrait>>;

    /// Backend identifier (e.g., "SoapySDR", "Seify", "rtl-sdr-rs")
    fn name(&self) -> &str;
}
