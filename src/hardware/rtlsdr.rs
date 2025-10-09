//! RTL-SDR native backend stub (future optimization)
//!
//! This stub provides direct RTL-SDR access without SoapySDR.
//! Implementation is optional and depends on whether Seify provides
//! sufficient performance for RTL-SDR devices.

use super::{Backend, DeviceError, DeviceErrorKind, DeviceId, DeviceInfo, DeviceTrait};
use crate::core::types::Result;

/// RTL-SDR native backend (future optimization)
pub struct RtlSdr;

impl Backend for RtlSdr {
    fn enumerate_devices(&self) -> Result<Vec<DeviceInfo>> {
        // When rtl-sdr-rs is integrated:
        // let devices = rtlsdr::get_device_count()?;
        Ok(vec![]) // Stub for now
    }

    fn open_device(&self, _id: &DeviceId) -> Result<Box<dyn DeviceTrait>> {
        Err(DeviceError::new(
            DeviceErrorKind::Unsupported,
            "RtlSdr",
            "rtl-sdr-rs backend not yet implemented (may use Seify instead)",
        )
        .into())
    }

    fn name(&self) -> &str {
        "rtl-sdr-rs (native)"
    }
}

// Future implementation notes:
//
// This backend would provide direct access to RTL-SDR devices without
// going through SoapySDR. However, Seify may provide better performance
// and features, so this implementation may not be necessary.
//
// If implemented, it would:
// 1. Use rtl-sdr-rs crate for direct device access
// 2. Create a bridge to rustradio blocks
// 3. Potentially provide better performance than SoapySDR
//
// Decision: Wait for Seify to mature and benchmark before implementing this.
