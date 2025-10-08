mod args;
mod config;
mod discovery;
mod headless_mode;
mod model;
mod scan;
mod signals;
mod train;
mod tui_mode;

pub use args::{AudioClassifier, Cli, Commands, ScanArgs, TrainArgs};
pub use scan::handle_scan_command;
pub use train::handle_train_command;
