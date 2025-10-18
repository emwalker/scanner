//! Status component - tracks current tuner activity and tuning state

/// Activity type for a tuner
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TunerActivity {
    /// Tuner is idle
    Idle,
    /// Tuner is scanning for signals
    Scanning,
    /// Tuner is listening to a station
    Listening,
    /// Other activity
    Other,
}

/// Component tracking current status of a tuner
///
/// This component tracks what the tuner is currently doing and its
/// current tuning parameters.
#[derive(Debug, Clone)]
pub struct StatusComponent {
    /// What the tuner is currently doing
    pub activity: TunerActivity,

    /// Current center frequency in Hz (None if not tuned)
    pub current_frequency: Option<f64>,

    /// Current bandwidth in Hz (None if not tuned)
    pub current_bandwidth: Option<f64>,
}

impl StatusComponent {
    /// Create a new status component with Idle activity
    pub fn new() -> Self {
        Self {
            activity: TunerActivity::Idle,
            current_frequency: None,
            current_bandwidth: None,
        }
    }

    /// Mark tuner as scanning
    pub fn start_scanning(&mut self) {
        self.activity = TunerActivity::Scanning;
    }

    /// Mark tuner as listening
    pub fn start_listening(&mut self) {
        self.activity = TunerActivity::Listening;
    }

    /// Mark tuner as idle
    pub fn idle(&mut self) {
        self.activity = TunerActivity::Idle;
        self.current_frequency = None;
        self.current_bandwidth = None;
    }

    /// Update tuning parameters
    pub fn tune(&mut self, frequency: f64, bandwidth: f64) {
        self.current_frequency = Some(frequency);
        self.current_bandwidth = Some(bandwidth);
    }
}

impl Default for StatusComponent {
    fn default() -> Self {
        Self::new()
    }
}
