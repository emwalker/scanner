/// Signal processing configuration
#[derive(Clone)]
pub struct SignalProcessingConfig {
    pub agc_settling_time: f64,
    pub disable_if_agc: bool,
    pub packet_size: usize,
    pub window_overlap: f64,
    pub frequency_tracking: FrequencyTrackingConfig,
}

impl Default for SignalProcessingConfig {
    fn default() -> Self {
        Self {
            agc_settling_time: 0.45,
            disable_if_agc: false,
            packet_size: 16384,
            window_overlap: 0.75,
            frequency_tracking: FrequencyTrackingConfig::default(),
        }
    }
}

/// Frequency tracking configuration
#[derive(Clone)]
pub struct FrequencyTrackingConfig {
    pub disabled: bool,
    pub method: String,
    pub accuracy: f64,
}

impl Default for FrequencyTrackingConfig {
    fn default() -> Self {
        Self {
            disabled: false,
            method: "pll".to_string(),
            accuracy: 5000.0,
        }
    }
}
