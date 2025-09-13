//! Filter configuration for different FM processing stages
//!
//! This module provides optimized filter parameters for detection vs audio processing stages.

use tracing::debug;

#[derive(Debug, Clone, Copy)]
pub enum FilterPurpose {
    /// Optimize for signal detection - wider passband, fewer taps, less CPU
    Detection,
    /// Optimize for audio quality - narrower passband, more taps, higher CPU
    Audio,
}

#[derive(Debug, Clone)]
pub struct FmFilterConfig {
    /// Channel bandwidth in Hz (passband)
    pub channel_bandwidth: f32,
    /// Transition width in Hz
    pub transition_width: f32,
    /// Decimation factor
    pub decimation: usize,
    /// Expected number of filter taps (informational)
    pub estimated_taps: usize,
    /// Expected MFLOPS for this filter (informational)
    pub estimated_mflops: f32,
}

impl FmFilterConfig {
    /// Create optimal filter configuration for given purpose and sample rate
    pub fn for_purpose(purpose: FilterPurpose, sample_rate: f64) -> Self {
        match purpose {
            FilterPurpose::Detection => Self::for_detection(sample_rate),
            FilterPurpose::Audio => Self::for_audio(sample_rate),
        }
    }

    /// Optimized for signal detection - maximize usable bandwidth, minimize CPU
    fn for_detection(sample_rate: f64) -> Self {
        let usable_bandwidth = sample_rate * 0.9; // Increased from 80% to 90% usable bandwidth

        // For detection, we want to capture as much spectrum as possible
        // Use even wider transition width to minimize filter taps and reduce CPU load
        let channel_bandwidth = (usable_bandwidth * 0.95) as f32; // 95% of usable bandwidth
        let transition_width = (usable_bandwidth * 0.4) as f32; // Very wide transition for minimal taps

        // Higher decimation for detection since we don't need high sample rates for squelch analysis
        let decimation = match sample_rate {
            sr if sr <= 1_000_000.0 => 8, // 1 MHz → ~125 kHz (increased decimation)
            sr if sr <= 2_000_000.0 => 15, // 2 MHz → 133 kHz (increased decimation)
            sr if sr <= 5_000_000.0 => 25, // 5 MHz → 200 kHz (increased decimation)
            _ => 50,                      // 10+ MHz → 200+ kHz (increased decimation)
        };

        let estimated_taps = Self::estimate_taps(sample_rate as f32, transition_width);
        let estimated_mflops = estimated_taps as f32 * sample_rate as f32 / 1_000_000.0;

        Self {
            channel_bandwidth,
            transition_width,
            decimation,
            estimated_taps,
            estimated_mflops,
        }
    }

    /// Optimized for audio quality - standard FM broadcast specifications
    fn for_audio(sample_rate: f64) -> Self {
        let decimation = match sample_rate {
            sr if sr <= 1_000_000.0 => 8,  // 1 MHz → ~125 kHz
            sr if sr <= 2_000_000.0 => 13, // 2 MHz → 154 kHz
            sr if sr <= 5_000_000.0 => 25, // 5 MHz → 200 kHz
            _ => 50,                       // 10+ MHz → 200+ kHz
        };

        // Standard FM broadcast specifications - narrow bandwidth for audio quality
        let channel_bandwidth = 200_000.0; // 200 kHz - standard FM channel bandwidth
        let transition_width = 120_000.0; // 120 kHz transition - optimized for audio quality

        let estimated_taps = Self::estimate_taps(sample_rate as f32, transition_width);
        let estimated_mflops = estimated_taps as f32 * sample_rate as f32 / 1_000_000.0;

        Self {
            channel_bandwidth,
            transition_width,
            decimation,
            estimated_taps,
            estimated_mflops,
        }
    }

    /// Estimate number of filter taps based on sample rate and transition width
    ///
    /// Formula from DSP literature: taps ≈ (sample_rate / transition_width) * 0.9
    fn estimate_taps(sample_rate: f32, transition_width: f32) -> usize {
        let taps = (sample_rate / transition_width * 0.9) as usize;
        // Ensure odd number of taps for symmetric FIR filter
        if taps.is_multiple_of(2) {
            taps + 1
        } else {
            taps
        }
    }

    /// Get the cutoff frequency (half of channel bandwidth)
    pub fn cutoff_frequency(&self) -> f32 {
        self.channel_bandwidth / 2.0
    }

    /// Check if this configuration can handle the required frequency offset
    pub fn can_handle_offset(&self, frequency_offset: f64, _sample_rate: f64) -> bool {
        let max_offset = self.cutoff_frequency() as f64;
        frequency_offset.abs() <= max_offset
    }

    /// Get narrow interference rejection filter configuration for post-decimation filtering
    /// This provides sharp frequency isolation to reject adjacent channel interference
    pub fn narrow_interference_filter(decimated_sample_rate: f32) -> (f32, f32, usize) {
        // Target: 200 kHz passband to isolate FM channel from adjacent channels
        let channel_bandwidth = 200_000.0; // 200 kHz - captures target FM channel
        let transition_width = 50_000.0; // 50 kHz transition - sharper than main filter but reasonable

        let cutoff_freq = channel_bandwidth / 2.0; // 100 kHz cutoff
        let estimated_taps = Self::estimate_taps(decimated_sample_rate, transition_width);

        debug!(
            decimated_samp_rate_khz = decimated_sample_rate / 1000.0,
            narrow_cutoff_khz = cutoff_freq / 1000.0,
            narrow_transition_khz = transition_width / 1000.0,
            narrow_taps = estimated_taps,
            narrow_mflops = estimated_taps as f32 * decimated_sample_rate / 1_000_000.0,
            "Narrow interference rejection filter configuration"
        );

        (cutoff_freq, transition_width, estimated_taps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_filter_2mhz() {
        let config = FmFilterConfig::for_purpose(FilterPurpose::Detection, 2_000_000.0);

        // Detection should use wide bandwidth and high decimation
        assert!(config.channel_bandwidth > 1_000_000.0); // > 1 MHz passband
        assert!(config.transition_width > 100_000.0); // > 100 kHz transition
        assert!(config.decimation >= 10); // High decimation
        assert!(config.estimated_taps < 50); // Few taps for efficiency
    }

    #[test]
    fn test_audio_filter_2mhz() {
        let config = FmFilterConfig::for_purpose(FilterPurpose::Audio, 2_000_000.0);

        // Audio should use standard FM channel width
        assert_eq!(config.channel_bandwidth, 200_000.0); // Standard FM channel
        assert_eq!(config.transition_width, 120_000.0); // Optimized transition width for reduced CPU load
        assert!(config.decimation < 15); // Updated for optimized decimation

        assert!(config.estimated_taps >= 15); // Reduced from 20 due to wider transition for CPU optimization
    }

    #[test]
    fn test_offset_handling() {
        let config = FmFilterConfig::for_purpose(FilterPurpose::Detection, 2_000_000.0);

        // Should handle offsets within passband
        assert!(config.can_handle_offset(500_000.0, 2_000_000.0));

        // Should reject offsets beyond passband
        assert!(!config.can_handle_offset(1_500_000.0, 2_000_000.0));
    }

    #[test]
    fn test_performance_scaling() {
        let config_1mhz = FmFilterConfig::for_purpose(FilterPurpose::Detection, 1_000_000.0);
        let config_2mhz = FmFilterConfig::for_purpose(FilterPurpose::Detection, 2_000_000.0);

        // CPU load should scale roughly with sample rate
        let ratio = config_2mhz.estimated_mflops / config_1mhz.estimated_mflops;
        assert!(ratio > 1.5 && ratio < 3.0); // Should be ~2x for 2x sample rate
    }
}
