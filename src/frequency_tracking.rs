use rustradio::Complex;
use std::f64::consts::PI;
use tracing::debug;

/// Configuration for frequency tracking algorithms
#[derive(Debug, Clone)]
pub struct TrackingConfig {
    #[allow(unused)]
    pub method: TrackingMethod,
    pub convergence_threshold: f64, // Hz accuracy needed for convergence
    pub timeout_samples: usize,     // Maximum samples before giving up
    pub search_window: f64,         // Hz window around initial frequency
    pub min_samples_for_convergence: usize, // Minimum samples before declaring convergence
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self {
            method: TrackingMethod::Pll,
            convergence_threshold: 5_000.0,      // ±5 kHz accuracy
            timeout_samples: 96_000,             // ~50ms at 2 MHz sample rate
            search_window: 100_000.0,            // ±100 kHz search window
            min_samples_for_convergence: 19_200, // ~10ms minimum at 2 MHz
        }
    }
}

/// Available frequency tracking methods
#[derive(Debug, Clone, Copy)]
pub enum TrackingMethod {
    Pll,
    SpectralCentroid,
    CrossCorrelation,
}

impl std::str::FromStr for TrackingMethod {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "pll" => Ok(TrackingMethod::Pll),
            "spectral" | "centroid" => Ok(TrackingMethod::SpectralCentroid),
            "correlation" | "xcorr" => Ok(TrackingMethod::CrossCorrelation),
            _ => Err(format!("Unknown tracking method: {}", s)),
        }
    }
}

/// State of frequency tracking process
#[derive(Debug, Clone)]
#[allow(unused)]
pub enum TrackingState {
    Converging,
    Converged(f64), // refined frequency in Hz
    Failed,         // couldn't converge within parameters
    Timeout,        // exceeded maximum samples
}

/// Trait for frequency tracking algorithms
pub trait FrequencyTracker {
    fn new(initial_freq: f64, sample_rate: f64, config: &TrackingConfig) -> Self
    where
        Self: Sized;

    /// Process a single I/Q sample and return current tracking state
    fn process_sample(&mut self, sample: Complex) -> TrackingState;

    /// Get confidence in the current estimate (0.0 to 1.0)
    fn get_confidence(&self) -> f32;

    /// Reset the tracker to start over
    #[allow(unused)]
    fn reset(&mut self, new_initial_freq: f64);
}

/// Phase-Locked Loop frequency tracker
/// Uses a digital PLL to lock onto the carrier frequency
pub struct PllTracker {
    initial_freq: f64,
    current_freq: f64,
    sample_rate: f64,

    // PLL state
    phase_accumulator: f64,
    phase_error_integrator: f64,
    #[allow(unused)]
    frequency_error: f64,

    // PLL parameters
    loop_bandwidth: f64, // Hz, controls how fast PLL locks
    damping_factor: f64, // Critically damped = 0.707

    // Convergence tracking
    samples_processed: usize,
    frequency_history: Vec<f64>, // Recent frequency estimates
    history_size: usize,

    // Configuration
    config: TrackingConfig,
    converged: bool,
    last_state: TrackingState,
}

impl FrequencyTracker for PllTracker {
    fn new(initial_freq: f64, sample_rate: f64, config: &TrackingConfig) -> Self {
        // PLL design parameters for FM carrier tracking
        let loop_bandwidth = 1000.0; // 1 kHz loop bandwidth for good tracking
        let damping_factor = 0.707; // Critical damping

        debug!(
            initial_freq_mhz = initial_freq / 1e6,
            sample_rate_mhz = sample_rate / 1e6,
            loop_bandwidth_khz = loop_bandwidth / 1e3,
            convergence_threshold_khz = config.convergence_threshold / 1e3,
            "PLL frequency tracker initialized"
        );

        Self {
            initial_freq,
            current_freq: initial_freq,
            sample_rate,
            phase_accumulator: 0.0,
            phase_error_integrator: 0.0,
            frequency_error: 0.0,
            loop_bandwidth,
            damping_factor,
            samples_processed: 0,
            frequency_history: Vec::with_capacity(100),
            history_size: 50, // Keep last 50 estimates for stability check
            config: config.clone(),
            converged: false,
            last_state: TrackingState::Converging,
        }
    }

    fn process_sample(&mut self, sample: Complex) -> TrackingState {
        self.samples_processed += 1;

        // Check timeout
        if self.samples_processed >= self.config.timeout_samples {
            self.last_state = TrackingState::Timeout;
            return self.last_state.clone();
        }

        // If already converged, just return the converged frequency
        if self.converged {
            return TrackingState::Converged(self.current_freq);
        }

        // Generate reference signal at current frequency estimate
        let phase_increment = 2.0 * PI * self.current_freq / self.sample_rate;
        let reference = Complex::new(
            self.phase_accumulator.cos() as f32,
            -self.phase_accumulator.sin() as f32, // Negative for down-conversion
        );

        // Complex multiply to get phase error (mix down to baseband)
        let mixed = sample * reference;

        // Phase detector: extract phase error from mixed signal
        let phase_error = mixed.im.atan2(mixed.re) as f64;

        // Loop filter: proportional + integral controller
        let kp = 2.0 * self.damping_factor * self.loop_bandwidth;
        let ki = self.loop_bandwidth * self.loop_bandwidth;

        // Update integrator
        self.phase_error_integrator += phase_error / self.sample_rate;

        // Calculate frequency correction
        let frequency_correction = kp * phase_error + ki * self.phase_error_integrator;

        // Update current frequency estimate
        self.current_freq = self.initial_freq + frequency_correction;

        // Clamp frequency to search window
        let max_freq = self.initial_freq + self.config.search_window;
        let min_freq = self.initial_freq - self.config.search_window;
        self.current_freq = self.current_freq.clamp(min_freq, max_freq);

        // Update phase accumulator for next iteration
        self.phase_accumulator += phase_increment;
        self.phase_accumulator %= 2.0 * PI;

        // Store frequency estimate for convergence checking
        self.frequency_history.push(self.current_freq);
        if self.frequency_history.len() > self.history_size {
            self.frequency_history.remove(0);
        }

        // Check for convergence
        if self.samples_processed >= self.config.min_samples_for_convergence
            && self.frequency_history.len() >= 10
        {
            let recent_estimates = &self.frequency_history[self.frequency_history.len() - 10..];
            let mean_freq: f64 =
                recent_estimates.iter().sum::<f64>() / recent_estimates.len() as f64;
            let variance: f64 = recent_estimates
                .iter()
                .map(|f| (f - mean_freq).powi(2))
                .sum::<f64>()
                / recent_estimates.len() as f64;
            let std_dev = variance.sqrt();

            if std_dev < self.config.convergence_threshold {
                self.converged = true;
                self.current_freq = mean_freq; // Use mean as final estimate

                debug!(
                    original_freq_mhz = self.initial_freq / 1e6,
                    converged_freq_mhz = self.current_freq / 1e6,
                    error_khz = (self.current_freq - self.initial_freq) / 1e3,
                    samples_processed = self.samples_processed,
                    convergence_time_ms = self.samples_processed as f64 / self.sample_rate * 1000.0,
                    std_dev_hz = std_dev,
                    "PLL frequency tracking converged"
                );

                self.last_state = TrackingState::Converged(self.current_freq);
                return self.last_state.clone();
            }
        }

        // Still converging
        self.last_state = TrackingState::Converging;
        TrackingState::Converging
    }

    fn get_confidence(&self) -> f32 {
        if self.frequency_history.len() < 10 {
            return 0.0;
        }

        let recent_estimates =
            &self.frequency_history[self.frequency_history.len().saturating_sub(10)..];
        let mean_freq: f64 = recent_estimates.iter().sum::<f64>() / recent_estimates.len() as f64;
        let variance: f64 = recent_estimates
            .iter()
            .map(|f| (f - mean_freq).powi(2))
            .sum::<f64>()
            / recent_estimates.len() as f64;
        let std_dev = variance.sqrt();

        // Convert standard deviation to confidence (0.0 to 1.0)
        // Lower std_dev = higher confidence
        let normalized_error = std_dev / self.config.convergence_threshold;
        (1.0 - normalized_error.min(1.0)).max(0.0) as f32
    }

    fn reset(&mut self, new_initial_freq: f64) {
        debug!(
            old_freq_mhz = self.initial_freq / 1e6,
            new_freq_mhz = new_initial_freq / 1e6,
            "PLL tracker reset"
        );

        self.initial_freq = new_initial_freq;
        self.current_freq = new_initial_freq;
        self.phase_accumulator = 0.0;
        self.phase_error_integrator = 0.0;
        self.frequency_error = 0.0;
        self.samples_processed = 0;
        self.frequency_history.clear();
        self.converged = false;
        self.last_state = TrackingState::Converging;
    }
}

/// Factory function to create a frequency tracker based on the method
pub fn create_tracker(
    method: TrackingMethod,
    initial_freq: f64,
    sample_rate: f64,
    config: &TrackingConfig,
) -> Box<dyn FrequencyTracker> {
    match method {
        TrackingMethod::Pll => Box::new(PllTracker::new(initial_freq, sample_rate, config)),
        TrackingMethod::SpectralCentroid => {
            // TODO: Implement spectral centroid tracker
            unimplemented!("SpectralCentroid tracker not yet implemented")
        }
        TrackingMethod::CrossCorrelation => {
            // TODO: Implement cross-correlation tracker
            unimplemented!("CrossCorrelation tracker not yet implemented")
        }
    }
}
