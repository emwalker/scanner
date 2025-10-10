/// Peak detection configuration
#[derive(Clone)]
pub struct PeakDetectionConfig {
    pub fft_size: usize,
    pub threshold: f32,
    pub scan_duration: f64,
    pub spectral_threshold: f32,
    pub cfar: CfarConfig,
    pub noise_floor: NoiseFloorConfig,
    pub windowing: WindowingConfig,
    pub averaging: AveragingConfig,
    pub multi_frame: MultiFrameConfig,
}

impl Default for PeakDetectionConfig {
    fn default() -> Self {
        Self {
            fft_size: 1024,
            threshold: 1.0,
            scan_duration: 1.5,
            spectral_threshold: 0.2,
            cfar: CfarConfig::default(),
            noise_floor: NoiseFloorConfig::default(),
            windowing: WindowingConfig::default(),
            averaging: AveragingConfig::default(),
            multi_frame: MultiFrameConfig::default(),
        }
    }
}

/// CFAR (Constant False Alarm Rate) detection configuration
#[derive(Clone)]
pub struct CfarConfig {
    pub enabled: bool,
    pub threshold_factor: f32,
    pub guard_cells: usize,
    pub reference_cells: usize,
    pub false_alarm_rate: f32,
}

impl Default for CfarConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold_factor: 10.0,
            guard_cells: 10,
            reference_cells: 50,
            false_alarm_rate: 0.01,
        }
    }
}

/// Dynamic noise floor estimation configuration
#[derive(Clone)]
pub struct NoiseFloorConfig {
    pub enabled: bool,
    pub percentile: f32,
    pub history_frames: usize,
    pub threshold_multiplier: f32,
    pub adaptation_rate: f32,
}

impl Default for NoiseFloorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            percentile: 0.25,
            history_frames: 8,
            threshold_multiplier: 1.6,
            adaptation_rate: 0.35,
        }
    }
}

/// Window type for spectral analysis
#[derive(Debug, Clone)]
pub enum WindowType {
    Rectangular,
    Hamming,
    Hanning,
    BlackmanHarris,
}

/// Windowing configuration for spectral preprocessing
#[derive(Clone)]
pub struct WindowingConfig {
    pub enabled: bool,
    pub window_type: WindowType,
    pub zero_padding_factor: usize,
    pub overlap_percent: f32,
}

impl Default for WindowingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            window_type: WindowType::BlackmanHarris,
            zero_padding_factor: 2,
            overlap_percent: 0.0,
        }
    }
}

/// Signal averaging and smoothing configuration
#[derive(Clone)]
pub struct AveragingConfig {
    pub exponential_smoothing: ExponentialSmoothingConfig,
    pub multi_frame_averaging: MultiFrameAveragingConfig,
    pub coherent_integration_enabled: bool,
    pub moving_average: MovingAverageConfig,
}

impl Default for AveragingConfig {
    fn default() -> Self {
        Self {
            exponential_smoothing: ExponentialSmoothingConfig::default(),
            multi_frame_averaging: MultiFrameAveragingConfig::default(),
            coherent_integration_enabled: true,
            moving_average: MovingAverageConfig::default(),
        }
    }
}

/// Exponential smoothing configuration
#[derive(Clone)]
pub struct ExponentialSmoothingConfig {
    pub enabled: bool,
    pub alpha: f32,
}

impl Default for ExponentialSmoothingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            alpha: 0.3,
        }
    }
}

/// Multi-frame averaging configuration
#[derive(Clone)]
pub struct MultiFrameAveragingConfig {
    pub enabled: bool,
    pub frames: usize,
}

impl Default for MultiFrameAveragingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            frames: 3,
        }
    }
}

/// Moving average filter configuration
#[derive(Clone)]
pub struct MovingAverageConfig {
    pub enabled: bool,
    pub window_size: usize,
}

impl Default for MovingAverageConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            window_size: 5,
        }
    }
}

/// Multi-frame integration configuration
#[derive(Clone)]
pub struct MultiFrameConfig {
    pub enabled: bool,
    pub history_frames: usize,
    pub confirmation_threshold: usize,
    pub frequency_tolerance: f64,
    pub max_age: f64,
}

impl Default for MultiFrameConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            history_frames: 5,
            confirmation_threshold: 2,
            frequency_tolerance: 25_000.0,
            max_age: 10.0,
        }
    }
}
