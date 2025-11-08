pub mod audio;
pub mod broadcast;
pub mod cli;
pub mod core;
pub mod discovery;
pub mod ecs;
pub mod file;
pub mod hardware;
pub mod ipc;
pub mod logging;
pub mod main_thread;
pub mod mpsc;
pub mod pause_signal;
pub mod persistence;
pub mod pipeline;
pub mod scanning;
pub mod shutdown;
pub mod signal;
pub mod task;
pub mod testing;
pub mod ui;

// Re-export commonly used types for convenience (backward compatibility)
pub use core::types;

pub use audio::quality as audio_quality;
pub use signal::peaks;
