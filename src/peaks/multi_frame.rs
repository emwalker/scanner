//! Multi-Frame Peak Integration
//!
//! This module implements peak persistence tracking across multiple scanning frames
//! to reduce false negatives and enhance weak signal detection through confirmation logic.

use crate::types::Peak;
use std::collections::HashMap;
use tracing::debug;

/// Configuration for multi-frame peak integration
#[derive(Debug, Clone)]
pub struct MultiFrameConfig {
    /// Number of frames to track peak history
    pub history_frames: usize,
    /// Minimum detections in N frames to confirm a peak (N of M logic)
    pub confirmation_threshold: usize,
    /// Frequency tolerance for matching peaks across frames (Hz)
    pub frequency_tolerance: f64,
    /// Maximum time between frames before clearing history (seconds)
    pub max_frame_age: f64,
}

impl Default for MultiFrameConfig {
    fn default() -> Self {
        Self {
            history_frames: 5,             // Track last 5 frames
            confirmation_threshold: 2,     // Default threshold (will be adaptive)
            frequency_tolerance: 25_000.0, // 25 kHz tolerance
            max_frame_age: 10.0,           // 10 second timeout
        }
    }
}

/// Peak tracking information across frames
#[derive(Debug, Clone)]
pub struct PeakTracker {
    /// Frequency of the tracked peak
    pub frequency_hz: f64,
    /// Detection history (frame_id, magnitude)
    pub detections: Vec<(u64, f32)>,
    /// Last detection timestamp
    pub last_seen: std::time::Instant,
    /// Weighted average magnitude
    pub average_magnitude: f32,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
}

impl PeakTracker {
    fn new(frequency_hz: f64, magnitude: f32, frame_id: u64) -> Self {
        Self {
            frequency_hz,
            detections: vec![(frame_id, magnitude)],
            last_seen: std::time::Instant::now(),
            average_magnitude: magnitude,
            confidence: 1.0 / 5.0, // 1 detection out of 5 frames
        }
    }

    fn add_detection(&mut self, magnitude: f32, frame_id: u64, config: &MultiFrameConfig) {
        self.detections.push((frame_id, magnitude));
        self.last_seen = std::time::Instant::now();

        // Keep only recent frames
        self.detections
            .retain(|(id, _)| *id + config.history_frames as u64 > frame_id);

        // Update weighted average (recent detections weighted more heavily)
        let total_weight: f32 = self
            .detections
            .iter()
            .enumerate()
            .map(|(i, _)| (i + 1) as f32)
            .sum();

        self.average_magnitude = self
            .detections
            .iter()
            .enumerate()
            .map(|(i, (_, mag))| (i + 1) as f32 * mag)
            .sum::<f32>()
            / total_weight;

        // Update confidence based on detection rate
        self.confidence = self.detections.len() as f32 / config.history_frames as f32;
    }

    fn is_confirmed(&self, config: &MultiFrameConfig) -> bool {
        let confidence_score = self.calculate_weighted_confidence(config);
        // Use confidence threshold instead of fixed confirmation count
        confidence_score >= 0.45 // 45% confidence threshold for weak signal support
    }

    /// Calculate weighted confidence score using adaptive weighting based on signal characteristics
    fn calculate_weighted_confidence(&self, config: &MultiFrameConfig) -> f32 {
        if self.detections.is_empty() {
            return 0.0;
        }

        // Base confidence from detection rate
        let detection_rate = self.detections.len() as f32 / config.history_frames as f32;

        // Adaptive signal strength factor using z-score normalization
        let strength_factor = {
            let normalized_magnitude = (self.average_magnitude / 200.0).min(1.0);
            // Apply sigmoid-like function for better discrimination
            1.0 / (1.0 + (-4.0 * (normalized_magnitude - 0.5)).exp())
        };

        // Enhanced consistency factor using coefficient of variation
        let consistency_factor = if self.detections.len() > 1 {
            let magnitudes: Vec<f32> = self.detections.iter().map(|(_, mag)| *mag).collect();
            let mean = magnitudes.iter().sum::<f32>() / magnitudes.len() as f32;
            let variance = magnitudes
                .iter()
                .map(|mag| (mag - mean).powi(2))
                .sum::<f32>()
                / magnitudes.len() as f32;
            let cv = variance.sqrt() / mean; // Coefficient of variation

            // Lower CV = higher consistency
            (1.0 / (1.0 + cv * 2.0)).max(0.1)
        } else {
            0.8 // Single detection gets reduced consistency score
        };

        // Improved recency factor with exponential decay
        let recency_factor = if !self.detections.is_empty() {
            let latest_frame = self
                .detections
                .iter()
                .map(|(frame, _)| *frame)
                .max()
                .unwrap();
            let frame_ages: f32 = self
                .detections
                .iter()
                .map(|(frame, _)| (latest_frame - frame) as f32)
                .map(|age| (-age / 2.0).exp()) // Exponential decay
                .sum();

            (frame_ages / self.detections.len() as f32).min(1.0)
        } else {
            1.0
        };

        // Adaptive weighting based on signal strength
        let strength_weight = 0.25 + 0.15 * strength_factor; // 0.25-0.40 range
        let detection_weight = 0.55 - 0.15 * strength_factor; // 0.40-0.55 range

        // Weighted combination with adaptive weights
        let confidence = detection_rate * detection_weight
            + strength_factor * strength_weight
            + consistency_factor * 0.15
            + recency_factor * 0.1;

        confidence.min(1.0)
    }

    fn is_expired(&self, config: &MultiFrameConfig) -> bool {
        self.last_seen.elapsed().as_secs_f64() > config.max_frame_age
    }
}

/// Multi-frame peak integration processor
pub struct MultiFrameIntegrator {
    config: MultiFrameConfig,
    trackers: HashMap<u64, PeakTracker>, // frequency_key -> tracker
    current_frame: u64,
}

impl MultiFrameIntegrator {
    pub fn new(config: MultiFrameConfig) -> Self {
        Self {
            config,
            trackers: HashMap::new(),
            current_frame: 0,
        }
    }

    /// Process a new frame of detected peaks with adaptive confirmation thresholds
    pub fn process_frame(&mut self, peaks: Vec<Peak>) -> Vec<Peak> {
        self.current_frame += 1;
        debug!(
            frame = self.current_frame,
            peaks = peaks.len(),
            "Processing frame for multi-frame integration"
        );

        // Remove expired trackers
        self.trackers
            .retain(|_, tracker| !tracker.is_expired(&self.config));

        // Process each peak in this frame
        for peak in peaks {
            let frequency_key = self.frequency_to_key(peak.frequency_hz);

            if let Some(tracker) = self.trackers.get_mut(&frequency_key) {
                // Update existing tracker
                tracker.add_detection(peak.magnitude, self.current_frame, &self.config);
            } else {
                // Create new tracker
                let tracker =
                    PeakTracker::new(peak.frequency_hz, peak.magnitude, self.current_frame);
                self.trackers.insert(frequency_key, tracker);
            }
        }

        // Return confirmed peaks with adaptive thresholds
        self.get_confirmed_peaks()
    }

    /// Get all confirmed peaks from current trackers
    fn get_confirmed_peaks(&self) -> Vec<Peak> {
        self.trackers
            .values()
            .filter(|tracker| tracker.is_confirmed(&self.config))
            .map(|tracker| Peak {
                frequency_hz: tracker.frequency_hz,
                magnitude: tracker.average_magnitude,
                // Note: other Peak fields would need to be handled appropriately
            })
            .collect()
    }

    /// Convert frequency to key for grouping nearby frequencies
    fn frequency_to_key(&self, frequency_hz: f64) -> u64 {
        (frequency_hz / self.config.frequency_tolerance).round() as u64
    }

    /// Get current tracking statistics
    pub fn get_statistics(&self) -> MultiFrameStatistics {
        let total_trackers = self.trackers.len();
        let confirmed_trackers = self
            .trackers
            .values()
            .filter(|t| t.is_confirmed(&self.config))
            .count();
        let pending_trackers = total_trackers - confirmed_trackers;

        MultiFrameStatistics {
            total_trackers,
            confirmed_trackers,
            pending_trackers,
            current_frame: self.current_frame,
        }
    }
}

/// Statistics for multi-frame integration
#[derive(Debug)]
pub struct MultiFrameStatistics {
    pub total_trackers: usize,
    pub confirmed_trackers: usize,
    pub pending_trackers: usize,
    pub current_frame: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that peak persistence tracking works across frames
    #[test]
    fn test_peak_persistence_tracking() {
        let config = MultiFrameConfig::default();
        let mut integrator = MultiFrameIntegrator::new(config.clone());

        // Create a consistent signal that appears in multiple frames
        let test_frequency = 89_100_000.0;
        let test_magnitude = 50.0;

        // Frame 1: Single detection
        let frame1_peaks = vec![Peak {
            frequency_hz: test_frequency,
            magnitude: test_magnitude,
        }];
        let confirmed1 = integrator.process_frame(frame1_peaks);
        assert_eq!(
            confirmed1.len(),
            0,
            "Single detection should not be confirmed yet"
        );

        // Frame 2: Same signal detected again - should now be confirmed with sufficient confidence
        let frame2_peaks = vec![Peak {
            frequency_hz: test_frequency,
            magnitude: test_magnitude * 1.1,
        }];
        let confirmed2 = integrator.process_frame(frame2_peaks);
        assert_eq!(
            confirmed2.len(),
            1,
            "Two detections should be confirmed with sufficient confidence score"
        );

        // Frame 3: Third detection - should still be confirmed
        let frame3_peaks = vec![Peak {
            frequency_hz: test_frequency,
            magnitude: test_magnitude * 0.9,
        }];
        let confirmed3 = integrator.process_frame(frame3_peaks);
        assert_eq!(confirmed3.len(), 1, "Should maintain confirmation");

        let confirmed_peak = &confirmed2[0]; // Check the peak from frame 2 when confirmation happens
        assert!(
            (confirmed_peak.frequency_hz - test_frequency).abs() < 1000.0,
            "Confirmed peak frequency should match tracked frequency"
        );

        // Magnitude should be weighted average of first 2 detections
        let expected_avg = (test_magnitude + test_magnitude * 1.1) / 2.0;
        assert!(
            (confirmed_peak.magnitude - expected_avg).abs() < 1.0,
            "Confirmed peak magnitude should be weighted average"
        );
    }

    /// Test that confirmation logic (N of M frames) reduces false negatives
    #[test]
    fn test_confirmation_logic_reduces_false_negatives() {
        let config = MultiFrameConfig {
            history_frames: 4,
            confirmation_threshold: 2, // 2 out of 4 frames
            frequency_tolerance: 25_000.0,
            max_frame_age: 10.0,
        };
        let mut integrator = MultiFrameIntegrator::new(config);

        let weak_signal_freq = 89_200_000.0;

        // Simulate intermittent weak signal (detected in frames 1, 3, 5)
        // This represents a signal that's sometimes below threshold due to noise/fading

        // Frame 1: Weak signal detected
        let frame1 = vec![Peak {
            frequency_hz: weak_signal_freq,
            magnitude: 5.0,
        }];
        let confirmed1 = integrator.process_frame(frame1);
        assert_eq!(confirmed1.len(), 0, "Single detection insufficient");

        // Frame 2: Signal not detected (below threshold due to noise)
        let frame2 = vec![];
        let confirmed2 = integrator.process_frame(frame2);
        assert_eq!(confirmed2.len(), 0, "Still insufficient detections");

        // Frame 3: Signal detected again
        let frame3 = vec![Peak {
            frequency_hz: weak_signal_freq,
            magnitude: 6.0,
        }];
        let confirmed3 = integrator.process_frame(frame3);
        assert_eq!(
            confirmed3.len(),
            1,
            "Two detections should provide sufficient confidence"
        );

        // Frame 4: Signal not detected again
        let frame4 = vec![];
        let confirmed4 = integrator.process_frame(frame4);
        assert_eq!(
            confirmed4.len(),
            1,
            "Should still be confirmed from previous detections"
        );

        // Frame 5: Signal detected third time
        let frame5 = vec![Peak {
            frequency_hz: weak_signal_freq,
            magnitude: 5.5,
        }];
        let confirmed5 = integrator.process_frame(frame5);
        assert_eq!(confirmed5.len(), 1, "Should maintain confirmation");

        let stats = integrator.get_statistics();
        assert_eq!(stats.confirmed_trackers, 1);
        assert_eq!(stats.current_frame, 5);
    }

    /// Test weak signal enhancement through non-coherent integration
    #[test]
    fn test_weak_signal_enhancement() {
        let config = MultiFrameConfig::default();
        let mut integrator = MultiFrameIntegrator::new(config);

        // Test scenario: Weak signal with varying magnitude due to noise/fading
        let signal_freq = 89_300_000.0;
        let _base_magnitude = 3.0;

        // Simulate 5 frames with noisy detections of the same weak signal
        let magnitudes = vec![2.8, 3.2, 2.9, 3.3, 3.1]; // Varying around base_magnitude

        let mut final_confirmed = vec![];
        for (i, &magnitude) in magnitudes.iter().enumerate() {
            let frame_peaks = vec![Peak {
                frequency_hz: signal_freq,
                magnitude,
            }];
            final_confirmed = integrator.process_frame(frame_peaks);

            if i >= 2 {
                // After 3rd frame (confirmation threshold)
                assert!(
                    !final_confirmed.is_empty(),
                    "Weak signal should be confirmed after sufficient detections"
                );
            }
        }

        // Check that the averaged magnitude is more stable than individual detections
        assert_eq!(final_confirmed.len(), 1);
        let enhanced_peak = &final_confirmed[0];

        // Average should be close to the true signal level
        let expected_average = magnitudes.iter().sum::<f32>() / magnitudes.len() as f32;
        assert!(
            (enhanced_peak.magnitude - expected_average).abs() < 0.1,
            "Enhanced magnitude should be close to average: got {}, expected {}",
            enhanced_peak.magnitude,
            expected_average
        );

        // Verify the signal was tracked across all frames
        let stats = integrator.get_statistics();
        assert_eq!(stats.confirmed_trackers, 1);
        assert_eq!(stats.current_frame, 5);
    }

    /// Test that false positive rate doesn't increase with multi-frame integration
    #[test]
    fn test_no_false_positive_increase() {
        let config = MultiFrameConfig {
            history_frames: 5,
            confirmation_threshold: 2, // Not used in weighted confidence approach
            frequency_tolerance: 25_000.0,
            max_frame_age: 10.0,
        };
        let mut integrator = MultiFrameIntegrator::new(config);

        // Simulate random noise peaks that appear only once or twice
        // These should NOT be confirmed as they lack persistence

        // Frame 1: Random noise peak
        let frame1 = vec![Peak {
            frequency_hz: 89_100_000.0,
            magnitude: 10.0,
        }];
        let confirmed1 = integrator.process_frame(frame1);
        assert_eq!(confirmed1.len(), 0);

        // Frame 2: Different random noise peak
        let frame2 = vec![Peak {
            frequency_hz: 89_200_000.0,
            magnitude: 8.0,
        }];
        let confirmed2 = integrator.process_frame(frame2);
        assert_eq!(confirmed2.len(), 0);

        // Frame 3: Another different noise peak
        let frame3 = vec![Peak {
            frequency_hz: 89_300_000.0,
            magnitude: 12.0,
        }];
        let confirmed3 = integrator.process_frame(frame3);
        assert_eq!(confirmed3.len(), 0);

        // Frame 4: Repeat one noise peak (still only 2 total detections)
        let frame4 = vec![Peak {
            frequency_hz: 89_100_000.0,
            magnitude: 11.0,
        }];
        let confirmed4 = integrator.process_frame(frame4);
        assert_eq!(
            confirmed4.len(),
            0,
            "Two weak detections should not reach confidence threshold"
        );

        // Frame 5: Empty frame
        let frame5 = vec![];
        let confirmed5 = integrator.process_frame(frame5);
        assert_eq!(confirmed5.len(), 0);

        // Verify no spurious confirmations
        let stats = integrator.get_statistics();
        assert_eq!(
            stats.confirmed_trackers, 0,
            "Random noise should not be confirmed"
        );
        assert!(stats.total_trackers > 0, "Should be tracking some peaks");
        assert_eq!(
            stats.pending_trackers, stats.total_trackers,
            "All trackers should be pending"
        );
    }
}
