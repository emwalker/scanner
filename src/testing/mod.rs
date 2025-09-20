//! Testing infrastructure for peak detection validation and performance measurement

#![allow(dead_code)]

pub mod benchmark_datasets;
pub mod performance_regression;
pub mod phase1_tests;
pub mod signal_generation;
pub mod statistical_validation;
pub mod test_helpers;
pub mod variance_measurement;

// Re-export commonly used items
pub use benchmark_datasets::*;
pub use performance_regression::*;
pub use signal_generation::*;
pub use statistical_validation::*;
pub use test_helpers::*;
pub use variance_measurement::*;
