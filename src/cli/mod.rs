mod args;
mod config;
mod discovery;
mod headless_mode;
mod model;
mod scan;
mod signals;
mod train;
mod tui_mode;
mod worker;
pub(crate) mod worker_logging;

pub use args::{AudioClassifier, Cli, Commands, ScanArgs, TrainArgs, WorkerCommand};
pub use scan::handle_scan_command;
pub use train::handle_train_command;
pub use worker::{handle_device_command, handle_enumerate_command};
