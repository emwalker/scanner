//! SDR backend abstraction layer
//!
//! This module provides a hardware-independent abstraction over different SDR backends
//! (SoapySDR, Seify, native drivers) allowing the application to work with any supported
//! SDR hardware through a unified interface.

pub mod backend;
pub mod device;
pub mod types;

// Backend implementations
pub mod mock;
pub mod soapy;

// Future backend stubs
#[cfg(feature = "seify")]
pub mod seify;

#[cfg(feature = "rtlsdr")]
pub mod rtlsdr;

// Legacy modules (will be removed in future refactoring)
pub mod sample_source;

use crate::types::{Result, ScanningConfig};

// Re-export commonly used types
pub use backend::Backend;
pub use device::DeviceTrait;
pub use types::{Capabilities, DeviceError, DeviceErrorKind, DeviceId, DeviceInfo};

// Re-export backend implementations
pub use mock::Mock;
pub use soapy::Soapy;

#[cfg(feature = "seify")]
pub use seify::Seify;

#[cfg(feature = "rtlsdr")]
pub use rtlsdr::RtlSdr;

// Re-export legacy sample source types
pub use sample_source::SampleSource;

// Keep old Device trait name for backward compatibility during migration
pub trait Device {
    fn tune(
        &self,
        config: &ScanningConfig,
        center_freq: f64,
    ) -> Result<Box<dyn crate::pool::SegmentTrait>>;
}
