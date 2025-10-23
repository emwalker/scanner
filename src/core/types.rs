use clap::ValueEnum;

// Re-export types from other modules for backward compatibility
pub use crate::core::bands::Band;
pub use crate::core::{
    config::{ScanningConfig, WindowType},
    errors::{Result, ScannerError, TEST_FREQUENCY_HZ},
    signals::{Candidate, ModulationType, Peak, Signal},
};

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
