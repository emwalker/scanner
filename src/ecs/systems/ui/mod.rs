//! UI update systems

mod signals_table;
mod update;

pub use signals_table::SignalsTableSystem;
pub use update::{SignalData, UIUpdateSystem};
