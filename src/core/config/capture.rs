/// Capture configuration for debugging and analysis
#[derive(Clone)]
pub struct CaptureConfig {
    pub audio_path: Option<String>,
    pub audio_duration: f64,
    pub iq_path: Option<String>,
    pub iq_duration: f64,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            audio_path: None,
            audio_duration: 3.0,
            iq_path: None,
            iq_duration: 2.0,
        }
    }
}
