//! SDR backend abstraction layer
//!
//! This module provides a hardware-independent abstraction over different SDR backends
//! (SoapySDR, Seify, native drivers) allowing the application to work with any supported
//! SDR hardware through a unified interface.

pub mod backend;
pub mod device;
pub mod pool;
pub mod streaming;
pub mod types;

// Backend implementations
pub mod mock;
pub mod soapy;
pub mod usb;

// Future backend stubs
#[cfg(feature = "seify")]
pub mod seify;

#[cfg(feature = "rtlsdr")]
pub mod rtlsdr;

// Legacy modules (will be removed in future refactoring)
pub mod sample_source;

// Re-export commonly used types
pub use backend::Backend;
pub use device::DeviceTrait;
// Re-export backend implementations
pub use mock::Mock;
#[cfg(feature = "rtlsdr")]
pub use rtlsdr::RtlSdr;
// Re-export legacy sample source types
pub use sample_source::SampleSource;
#[cfg(feature = "seify")]
pub use seify::Seify;
pub use soapy::Soapy;
pub use streaming::{ActualConfig, StreamingDevice};
pub use types::{Capabilities, DeviceError, DeviceErrorKind, DeviceId, DeviceInfo};
pub use usb::Usb;
