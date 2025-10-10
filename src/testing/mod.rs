//! Testing infrastructure for peak detection validation and performance measurement

#![allow(dead_code)]

pub mod benchmark_datasets;
#[cfg(test)]
pub mod detection_regression_tests;
pub mod helpers;
pub mod performance_regression;
pub mod signal_generation;
pub mod statistical_validation;
pub mod variance_measurement;

// Re-export commonly used items
pub use benchmark_datasets::*;
pub use helpers::*;
pub use performance_regression::*;
pub use signal_generation::*;
pub use statistical_validation::*;
pub use variance_measurement::*;
