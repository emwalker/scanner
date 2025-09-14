//! Audio Quality Analysis Module
//!
//! This module provides various approaches to audio quality assessment:
//! - `legacy`: Original audio quality analyzer used in production
//! - `normalized`: New normalized, gain-invariant metrics for robust testing
//!
//! The module is designed to support migration from the legacy system to the
//! normalized system while preserving calibration results and enabling comprehensive testing.

pub mod legacy;
pub mod ml_regression;
pub mod normalized;

#[derive(Debug, Clone, PartialEq)]
pub enum AudioQuality {
    Good,     // High quality, minimal distortion
    Moderate, // Audible content with some distortion/noise
    Poor,     // Weak signal with significant distortion but still audible
    Static,   // Primarily noise, no clear audio content
    Unknown,  // Unable to determine quality (insufficient data)
}

impl AudioQuality {
    pub fn to_human_string(&self) -> &'static str {
        match self {
            AudioQuality::Good => "good audio",
            AudioQuality::Moderate => "moderate audio",
            AudioQuality::Poor => "poor audio",
            AudioQuality::Static => "static",
            AudioQuality::Unknown => "unknown quality",
        }
    }

    /// Returns true if this quality level represents audible audio content (not just noise/static)
    pub fn is_audio(&self) -> bool {
        matches!(
            self,
            AudioQuality::Good | AudioQuality::Moderate | AudioQuality::Poor
        )
    }
}

// Re-export the main types for backward compatibility
pub use legacy::AudioQualityAnalyzer;

// Re-export normalized types for new code
pub use normalized::{AudioQualityMetrics, QualityResult};

// Re-export ML regression types
pub use ml_regression::{MLAudioQualityAnalyzer, QualityScore, TrainingSample};
