use std::time::SystemTime;

use crate::signal;

#[derive(Debug, Clone)]
pub struct Peak {
    pub frequency_hz: f64,
    pub magnitude: f32,
}

#[derive(Debug)]
pub enum Candidate {
    Fm(signal::Candidate),
}

/// Represents a successfully detected and demodulated signal
#[derive(Debug, Clone)]
pub struct Signal {
    /// Center frequency of the signal in Hz
    pub frequency_hz: f64,
    /// Signal strength/power measurement
    pub signal_strength: f32,
    /// Estimated bandwidth of the signal in Hz
    pub bandwidth_hz: f32,
    /// Type of modulation detected
    pub modulation: ModulationType,
    /// Audio sample rate for this signal
    pub audio_sample_rate: u32,
    /// Timestamp when signal was detected
    pub detected_at: std::time::SystemTime,
    /// Duration of analysis period that led to detection
    pub analysis_duration_ms: u32,
    /// Center frequency used by SDR during detection (needed for audio processing offset
    /// calculation)
    pub detection_center_freq: f64,
    /// Audio quality assessment
    pub audio_quality: crate::audio::quality::AudioQuality,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum ModulationType {
    WFM, // Wideband FM
    NFM, // Narrowband FM
    AM,  // Amplitude Modulation
    LSB, // Lower Side Band
    USB, // Upper Side Band
}

impl<'de> serde::Deserialize<'de> for ModulationType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "WFM" => Ok(ModulationType::WFM),
            "NFM" => Ok(ModulationType::NFM),
            "AM" => Ok(ModulationType::AM),
            "LSB" => Ok(ModulationType::LSB),
            "USB" => Ok(ModulationType::USB),
            // Handle legacy "FM" by mapping to WFM
            "FM" => Ok(ModulationType::WFM),
            _ => Err(serde::de::Error::unknown_variant(&s, &[
                "WFM", "NFM", "AM", "LSB", "USB", "FM",
            ])),
        }
    }
}

impl std::fmt::Display for ModulationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModulationType::WFM => write!(f, "WFM"),
            ModulationType::NFM => write!(f, "NFM"),
            ModulationType::AM => write!(f, "AM"),
            ModulationType::LSB => write!(f, "LSB"),
            ModulationType::USB => write!(f, "USB"),
        }
    }
}

impl Signal {
    pub fn new_fm(
        frequency_hz: f64,
        signal_strength: f32,
        bandwidth_hz: f32,
        audio_sample_rate: u32,
        analysis_duration_ms: u32,
        detection_center_freq: f64,
        audio_quality: crate::audio::quality::AudioQuality,
    ) -> Self {
        Self {
            frequency_hz,
            signal_strength,
            bandwidth_hz,
            modulation: ModulationType::WFM,
            audio_sample_rate,
            detected_at: SystemTime::now(),
            analysis_duration_ms,
            detection_center_freq,
            audio_quality,
        }
    }
}

impl Candidate {
    pub fn frequency_hz(&self) -> f64 {
        match self {
            Candidate::Fm(signal) => signal.frequency_hz,
        }
    }

    pub fn analyze(
        &self,
        sdr_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
        context: &crate::pipeline::AnalysisContext,
    ) -> crate::core::errors::Result<()> {
        match self {
            Candidate::Fm(signal) => signal.analyze(sdr_rx, context),
        }
    }
}
