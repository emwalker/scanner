//! Audio Quality Analysis Module
//!
//! This module provides various approaches to audio quality assessment:
//! - `legacy`: Original audio quality analyzer used in production
//! - `normalized`: New normalized, gain-invariant metrics for robust testing
//!
//! The module is designed to support migration from the legacy system to the
//! normalized system while preserving calibration results and enabling comprehensive testing.

pub mod legacy;
pub mod normalized;

// Re-export the main types from legacy for backward compatibility
pub use legacy::{AudioQuality, AudioQualityAnalyzer};

// Re-export normalized types for new code
pub use normalized::{AudioQualityMetrics, QualityResult};
