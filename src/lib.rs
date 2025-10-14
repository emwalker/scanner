pub mod audio;
pub mod broadcast;
pub mod cli;
pub mod core;
pub mod discovery;
pub mod file;
pub mod hardware;
pub mod ipc;
pub mod logging;
pub mod main_thread;
pub mod mpsc;
pub mod pipeline;
pub mod scanner_state;
pub mod scanning;
pub mod shutdown;
pub mod signal;
pub mod task;
pub mod testing;
pub mod ui;

// Re-export commonly used types for convenience (backward compatibility)
pub use audio::quality as audio_quality;
pub use core::types;
pub use signal::peaks;
