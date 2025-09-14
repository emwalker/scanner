//! Audio Quality Analysis Module
//!
//! This module provides various approaches to audio quality assessment:
//! - `legacy`: Original audio quality analyzer used in production
//! - `normalized`: New normalized, gain-invariant metrics for robust testing
//!
//! The module is designed to support migration from the legacy system to the
//! normalized system while preserving calibration results and enabling comprehensive testing.

pub mod heuristic2;
pub mod legacy;
pub mod normalized;
pub mod random_forest;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioQuality {
    Good = 4,     // High quality, minimal distortion
    Moderate = 3, // Audible content with some distortion/noise
    Poor = 2,     // Weak signal with significant distortion but still audible
    NoAudio = 1,  // Signal present but no discernible audio content
    Static = 0,   // Primarily noise, no clear audio content
    Unknown = -1, // Unable to determine quality (insufficient data) - treat as separate value
}

impl AudioQuality {
    pub fn to_human_string(&self) -> &'static str {
        match self {
            AudioQuality::Good => "good audio",
            AudioQuality::Moderate => "moderate audio",
            AudioQuality::Poor => "poor audio",
            AudioQuality::NoAudio => "no audio",
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

    /// Convert to numerical value for ML regression
    pub fn to_score(&self) -> f64 {
        *self as i32 as f64
    }

    /// Create from ML prediction score using thresholds calibrated for our actual ML model output
    pub fn from_score(score: f64) -> Self {
        if score < -0.5 {
            AudioQuality::Unknown
        } else if score < 0.25 {
            AudioQuality::Static
        } else if score < 0.4 {
            AudioQuality::NoAudio
        } else if score < 0.55 {
            AudioQuality::Poor
        } else if score < 0.7 {
            AudioQuality::Moderate
        } else {
            AudioQuality::Good
        }
    }
}

// Re-export the main types for backward compatibility
pub use legacy::AudioQualityAnalyzer;

// Re-export normalized types for new code
pub use normalized::{AudioQualityMetrics, QualityResult};

// Re-export heuristic types for rule-based classification
pub use heuristic2::{
    AudioFeatures, AudioQualityClassifier as HeuristicClassifier, QualityResult as HeuristicResult,
};

// Re-export Random Forest types for actual machine learning
pub use random_forest::{
    AudioQualityClassifier as RandomForestClassifier, QualityResult as RandomForestResult,
};

/// Shared training dataset with human-rated audio quality samples
/// This centralized list is used by both ML regression and normalized quality analysis
pub fn get_training_dataset() -> Vec<(&'static str, AudioQuality)> {
    vec![
        // Static samples (from our calibration)
        ("000.087.700.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.088.099.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.088.299.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.088.300.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.088.499.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.089.500.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.091.702.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.093.100.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.094.500.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.095.500.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.099.500.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.101.100.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.101.100.000Hz-wfm-002.wav", AudioQuality::Static),
        ("000.089.299.000Hz-wfm-001.wav", AudioQuality::Static), // Reclassified based on ML analysis
        // NoAudio samples (signal present but no discernible audio content)
        ("000.096.300.000Hz-wfm-001.wav", AudioQuality::NoAudio),
        ("000.088.700.000Hz-wfm-001.wav", AudioQuality::NoAudio), // Reclassified based on ML analysis
        ("000.089.099.000Hz-wfm-001.wav", AudioQuality::NoAudio), // Reclassified based on ML analysis
        // Poor samples
        ("000.090.302.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.092.101.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.093.300.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.093.500.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.093.700.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.094.100.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.093.300.000Hz-wfm-002.wav", AudioQuality::Poor),
        ("000.094.700.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.095.700.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.097.100.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.100.300.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.103.500.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.107.100.000Hz-wfm-001.wav", AudioQuality::Poor),
        // Moderate samples
        ("000.089.700.000Hz-wfm-001.wav", AudioQuality::Moderate),
        ("000.090.101.000Hz-wfm-001.wav", AudioQuality::Moderate),
        ("000.091.100.000Hz-wfm-001.wav", AudioQuality::Moderate),
        ("000.091.500.000Hz-wfm-001.wav", AudioQuality::Moderate),
        ("000.096.900.000Hz-wfm-001.wav", AudioQuality::Moderate),
        ("000.098.500.000Hz-wfm-001.wav", AudioQuality::Moderate),
        ("000.102.500.000Hz-wfm-001.wav", AudioQuality::Moderate),
        // Good samples
        ("000.088.900.000Hz-wfm-001.wav", AudioQuality::Good),
        ("000.099.100.000Hz-wfm-001.wav", AudioQuality::Good),
        ("000.100.700.000Hz-wfm-001.wav", AudioQuality::Good),
        ("000.103.900.000Hz-wfm-001.wav", AudioQuality::Good),
    ]
}
