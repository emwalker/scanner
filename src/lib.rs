pub mod broadcast;
pub mod file;
pub mod fm;
pub mod freq_xlating_fir;
pub mod frequency_tracking;
pub mod iq_capture;
pub mod logging;
pub mod mpsc;
pub mod soapy;
pub mod testing;
pub mod types;

pub use crate::logging::LogBuffer;
pub use crate::types::{Band, Candidate, Peak, Result, ScannerError, ScanningConfig};

// Re-export main types for testing
use clap::ValueEnum;

#[derive(ValueEnum, Copy, Clone, Debug)]
pub enum Format {
    /// JSON structured logging format
    Json,
    /// Simple text logging format
    Text,
    /// Standard log format with timestamps and levels
    Log,
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Format::Json => write!(f, "json"),
            Format::Text => write!(f, "text"),
            Format::Log => write!(f, "log"),
        }
    }
}
