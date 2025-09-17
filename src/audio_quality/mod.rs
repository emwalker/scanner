//! Audio Quality Analysis Module
//!
//! This module provides a unified interface for audio quality assessment with multiple
//! classifier implementations:
//! - `heuristic1`: Original statistical audio quality analyzer
//! - `heuristic2`: Fast normalized, gain-invariant metrics
//! - `heuristic3`: Rule-based classification with feature extraction
//! - `random_forest`: Machine learning classification using Random Forest

pub mod heuristic1;
pub mod heuristic2;
pub mod heuristic3;
pub mod random_forest;

use crate::types::Result;

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

    /// Returns true if this quality level meets or exceeds the minimum threshold
    pub fn meets_threshold(&self, min_threshold: AudioQuality) -> bool {
        (*self as i32) >= (min_threshold as i32)
    }

    /// Convert to numerical value for ML regression
    pub fn to_score(&self) -> f64 {
        *self as i32 as f64
    }

    /// Create from ML prediction score using thresholds calibrated for our actual ML model output
    pub fn from_ml_score(score: f64) -> Self {
        if score >= 3.5 {
            AudioQuality::Good
        } else if score >= 2.5 {
            AudioQuality::Moderate
        } else if score >= 1.5 {
            AudioQuality::Poor
        } else if score >= 0.5 {
            AudioQuality::NoAudio
        } else if score >= -0.5 {
            AudioQuality::Static
        } else {
            AudioQuality::Unknown
        }
    }
}

/// Comprehensive audio features for quality assessment
#[derive(Debug, Clone)]
pub struct AudioFeatures {
    // Energy-based features
    pub rms_energy: f32,
    pub peak_amplitude: f32,
    pub dynamic_range: f32,

    // Spectral features
    pub spectral_centroid: f32,
    pub spectral_rolloff: f32,
    pub spectral_flux: f32,
    pub high_freq_energy: f32,

    // Temporal features
    pub zero_crossing_rate: f32,
    pub silence_ratio: f32,

    // Quality indicators
    pub snr_estimate: f32,
    pub harmonic_ratio: f32,
}

/// Unified result structure for all audio quality classifiers
#[derive(Debug, Clone)]
pub struct QualityResult {
    /// Primary audio quality classification
    pub quality: AudioQuality,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Signal strength (0.0 to 1.0)
    pub signal_strength: f32,
    /// Detailed audio features (when available)
    pub features: Option<AudioFeatures>,
}

/// Trait for audio quality classifiers
pub trait Classifier: Send + Sync {
    /// Analyze audio samples and return quality assessment
    fn analyze(&self, samples: &[f32], sample_rate: f32) -> Result<QualityResult>;

    /// Get the name of this classifier
    fn name(&self) -> &'static str;
}

/// Main audio analyzer with pluggable classifier backend
#[derive(Clone)]
pub struct AudioAnalyzer {
    classifier: std::sync::Arc<dyn Classifier>,
}

impl AudioAnalyzer {
    /// Create new analyzer with specified classifier
    pub fn new(classifier: Box<dyn Classifier>) -> Self {
        Self {
            classifier: std::sync::Arc::from(classifier),
        }
    }

    /// Analyze audio samples using the configured classifier
    pub fn analyze(&self, samples: &[f32], sample_rate: f32) -> Result<QualityResult> {
        self.classifier.analyze(samples, sample_rate)
    }

    /// Get the name of the current classifier
    pub fn classifier_name(&self) -> &'static str {
        self.classifier.name()
    }

    /// Create a pass-through analyzer that always reports good audio quality
    /// Used when squelch is disabled
    pub fn pass_through() -> Self {
        Self {
            classifier: std::sync::Arc::new(PassThroughClassifier),
        }
    }

    /// Create a mock analyzer for testing
    pub fn mock() -> Self {
        Self {
            classifier: std::sync::Arc::new(MockClassifier),
        }
    }
}

/// Trivial classifier that always reports good audio quality
/// Used when squelch is disabled
struct PassThroughClassifier;

impl Classifier for PassThroughClassifier {
    fn analyze(&self, samples: &[f32], _sample_rate: f32) -> Result<QualityResult> {
        let rms_energy = if samples.is_empty() {
            0.0
        } else {
            (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt()
        };

        Ok(QualityResult {
            quality: AudioQuality::Good,
            confidence: 1.0,
            signal_strength: rms_energy,
            features: None,
        })
    }

    fn name(&self) -> &'static str {
        "pass_through"
    }
}

/// Mock classifier for testing that makes realistic quality decisions
/// Used in unit tests to avoid complex ML dependencies
struct MockClassifier;

impl Classifier for MockClassifier {
    fn analyze(&self, samples: &[f32], _sample_rate: f32) -> Result<QualityResult> {
        let rms_energy = if samples.is_empty() {
            0.0
        } else {
            (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt()
        };

        // Simple mock logic based on signal strength
        let quality = if rms_energy < 0.05 {
            AudioQuality::Static
        } else if rms_energy < 0.1 {
            AudioQuality::Poor
        } else if rms_energy < 0.3 {
            AudioQuality::Moderate
        } else {
            AudioQuality::Good
        };

        let confidence = if rms_energy > 0.1 { 0.8 } else { 0.6 };

        Ok(QualityResult {
            quality,
            confidence,
            signal_strength: rms_energy,
            features: None,
        })
    }

    fn name(&self) -> &'static str {
        "mock"
    }
}

/// Shared training dataset with human-rated audio quality samples
/// This centralized list is used by ML classifiers for training
pub fn training_dataset() -> Vec<(&'static str, AudioQuality)> {
    vec![
        ("000.087.700.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.088.099.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.088.299.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.088.300.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.088.499.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.088.700.000Hz-wfm-001.wav", AudioQuality::NoAudio),
        ("000.088.900.000Hz-wfm-001.wav", AudioQuality::Good),
        ("000.088.900.000Hz-wfm-002.wav", AudioQuality::Moderate),
        ("000.088.900.000Hz-wfm-003.wav", AudioQuality::Good),
        ("000.089.099.000Hz-wfm-001.wav", AudioQuality::NoAudio),
        ("000.089.299.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.089.500.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.089.700.000Hz-wfm-001.wav", AudioQuality::Moderate),
        ("000.089.700.000Hz-wfm-003.wav", AudioQuality::Moderate),
        ("000.090.100.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.090.100.000Hz-wfm-002.wav", AudioQuality::Moderate),
        ("000.090.101.000Hz-wfm-001.wav", AudioQuality::Moderate),
        ("000.090.302.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.090.900.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.091.100.000Hz-wfm-001.wav", AudioQuality::Moderate),
        ("000.091.500.000Hz-wfm-001.wav", AudioQuality::Moderate),
        ("000.091.500.000Hz-wfm-002.wav", AudioQuality::Moderate),
        ("000.091.702.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.092.100.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.092.101.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.092.300.000Hz-wfm-001.wav", AudioQuality::Moderate),
        ("000.092.300.000Hz-wfm-002.wav", AudioQuality::Moderate),
        ("000.092.500.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.092.900.000Hz-wfm-001.wav", AudioQuality::Moderate),
        ("000.093.100.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.093.300.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.093.300.000Hz-wfm-002.wav", AudioQuality::Poor),
        ("000.093.500.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.093.700.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.093.900.000Hz-wfm-001.wav", AudioQuality::Moderate),
        ("000.093.900.000Hz-wfm-002.wav", AudioQuality::Good),
        ("000.094.100.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.094.500.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.094.700.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.094.900.000Hz-wfm-001.wav", AudioQuality::Moderate),
        ("000.094.900.000Hz-wfm-002.wav", AudioQuality::Moderate),
        ("000.095.100.000Hz-wfm-004.wav", AudioQuality::Static),
        ("000.095.500.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.095.700.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.095.700.000Hz-wfm-002.wav", AudioQuality::Moderate),
        ("000.095.700.000Hz-wfm-003.wav", AudioQuality::Moderate),
        ("000.096.100.000Hz-wfm-001.wav", AudioQuality::Good),
        ("000.096.100.000Hz-wfm-002.wav", AudioQuality::Moderate),
        ("000.096.300.000Hz-wfm-001.wav", AudioQuality::NoAudio),
        ("000.096.500.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.096.700.000Hz-wfm-004.wav", AudioQuality::Poor),
        ("000.096.900.000Hz-wfm-001.wav", AudioQuality::Moderate),
        ("000.096.900.000Hz-wfm-002.wav", AudioQuality::Moderate),
        ("000.096.900.000Hz-wfm-003.wav", AudioQuality::Moderate),
        ("000.097.100.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.097.300.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.098.300.000Hz-wfm-004.wav", AudioQuality::Static),
        ("000.098.500.000Hz-wfm-001.wav", AudioQuality::Moderate),
        ("000.098.500.000Hz-wfm-002.wav", AudioQuality::Static),
        ("000.098.900.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.099.100.000Hz-wfm-001.wav", AudioQuality::Good),
        ("000.099.100.000Hz-wfm-002.wav", AudioQuality::Moderate),
        ("000.099.100.000Hz-wfm-003.wav", AudioQuality::Good),
        ("000.099.500.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.100.100.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.100.300.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.100.700.000Hz-wfm-001.wav", AudioQuality::Good),
        ("000.100.700.000Hz-wfm-002.wav", AudioQuality::Moderate),
        ("000.100.900.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.101.100.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.101.100.000Hz-wfm-002.wav", AudioQuality::Static),
        ("000.101.300.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.101.300.000Hz-wfm-002.wav", AudioQuality::Moderate),
        ("000.101.700.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.102.500.000Hz-wfm-001.wav", AudioQuality::Moderate),
        ("000.102.500.000Hz-wfm-002.wav", AudioQuality::Moderate),
        ("000.102.500.000Hz-wfm-003.wav", AudioQuality::Moderate),
        ("000.103.500.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.103.500.000Hz-wfm-002.wav", AudioQuality::Moderate),
        ("000.103.500.000Hz-wfm-003.wav", AudioQuality::Moderate),
        ("000.103.700.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.103.900.000Hz-wfm-001.wav", AudioQuality::Good),
        ("000.104.500.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.105.100.000Hz-wfm-001.wav", AudioQuality::Moderate),
        ("000.105.100.000Hz-wfm-002.wav", AudioQuality::Moderate),
        ("000.105.500.000Hz-wfm-001.wav", AudioQuality::Moderate),
        ("000.106.500.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.107.100.000Hz-wfm-001.wav", AudioQuality::Poor),
    ]
}
