pub mod bands;
pub mod config;
pub mod errors;
pub mod signals;
pub mod types;

// Re-export commonly used types for convenience
pub use bands::Band;
pub use config::{ScanningConfig, WindowType};
pub use errors::{Result, ScannerError, TEST_FREQUENCY_HZ};
pub use signals::{Candidate, ModulationType, Peak, Signal};
pub use types::{ConsoleWriter, Format, Logger};
