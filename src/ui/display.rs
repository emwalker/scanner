//! Progress display for real-time scanning feedback

use crate::ui::{ProgressEvent, ProgressEventType};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Displays progress information received from ChannelProgressReporter
pub struct ProgressDisplay {
    receiver: mpsc::Receiver<ProgressEvent>,
    last_update: Instant,
    update_interval: Duration,
    current_frequency: Option<f64>,
    window_id: Option<usize>,
    peak_count: usize,
}

impl ProgressDisplay {
    /// Create new progress display
    pub fn new(receiver: mpsc::Receiver<ProgressEvent>) -> Self {
        Self {
            receiver,
            last_update: Instant::now(),
            update_interval: Duration::from_millis(500), // Update every 500ms
            current_frequency: None,
            window_id: None,
            peak_count: 0,
        }
    }

    /// Run the display loop - processes events and updates display
    pub fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        while let Ok(event) = self.receiver.recv() {
            self.process_event(event);

            // Update display if enough time has passed
            if self.last_update.elapsed() >= self.update_interval {
                self.update_display();
                self.last_update = Instant::now();
            }
        }
        Ok(())
    }

    /// Run with timeout for testing
    pub fn run_with_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let start = Instant::now();

        while start.elapsed() < timeout {
            match self.receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(event) => {
                    self.process_event(event);

                    // Update display if enough time has passed
                    if self.last_update.elapsed() >= self.update_interval {
                        self.update_display();
                        self.last_update = Instant::now();
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // Continue checking until timeout
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // Channel closed, exit cleanly
                    break;
                }
            }
        }
        Ok(())
    }

    fn process_event(&mut self, event: ProgressEvent) {
        match event.event_type {
            ProgressEventType::PeakDetected => {
                self.current_frequency = Some(event.frequency_hz);
                self.window_id = Some(event.metadata.window_id);
                self.peak_count += 1;
            }
            ProgressEventType::CandidateCreated => {
                self.current_frequency = Some(event.frequency_hz);
            }
            ProgressEventType::AudioAnalysisStarted => {
                self.current_frequency = Some(event.frequency_hz);
            }
            ProgressEventType::AudioAnalysisCompleted => {
                // Keep current frequency
            }
            ProgressEventType::CandidateRejected => {
                // Keep current frequency
            }
            ProgressEventType::SignalGenerated => {
                // Keep current frequency
            }
            ProgressEventType::AudioPlaybackStarted => {
                // Keep current frequency
            }
            ProgressEventType::AudioPlaybackCompleted => {
                // Keep current frequency
            }
            ProgressEventType::ThreadCompleted => {
                // Keep current frequency
            }
        }
    }

    fn update_display(&self) {
        if let (Some(freq), Some(window)) = (self.current_frequency, self.window_id) {
            // Use tracing::info for progress display instead of eprintln!
            tracing::info!(
                "Scanning: {:.1} MHz (Window {}, {} peaks processed)",
                freq / 1e6,
                window,
                self.peak_count
            );
        }
    }

    /// Get current state for testing
    pub fn current_state(&self) -> (Option<f64>, Option<usize>, usize) {
        (self.current_frequency, self.window_id, self.peak_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::ProgressEvent;
    use std::sync::mpsc;

    #[test]
    fn test_progress_display_output() {
        let (sender, receiver) = mpsc::channel();
        let mut display = ProgressDisplay::new(receiver);

        // Send test events
        let events = vec![
            ProgressEvent {
                event_type: ProgressEventType::PeakDetected,
                frequency_hz: 88_900_000.0,
                metadata: crate::scanning::window::WindowMetadata {
                    center_frequency_hz: 88_900_000.0,
                    window_id: 1,
                },
                candidate_id: Some("88.9-1".to_string()),
                audio_quality: None,
                signal_strength: None,
                timestamp: Instant::now(),
                tuner_id: None,
            },
            ProgressEvent {
                event_type: ProgressEventType::CandidateCreated,
                frequency_hz: 89_100_000.0,
                metadata: crate::scanning::window::WindowMetadata {
                    center_frequency_hz: 89_100_000.0,
                    window_id: 1,
                },
                candidate_id: Some("89.1-1".to_string()),
                audio_quality: None,
                signal_strength: None,
                timestamp: Instant::now(),
                tuner_id: None,
            },
            ProgressEvent {
                event_type: ProgressEventType::PeakDetected,
                frequency_hz: 89_300_000.0,
                metadata: crate::scanning::window::WindowMetadata {
                    center_frequency_hz: 89_300_000.0,
                    window_id: 2,
                },
                candidate_id: Some("89.3-2".to_string()),
                audio_quality: None,
                signal_strength: None,
                timestamp: Instant::now(),
                tuner_id: None,
            },
        ];

        for event in events {
            sender.send(event).expect("Should send event");
        }

        // Close sender so display exits cleanly
        drop(sender);

        // Run display with timeout
        let result = display.run_with_timeout(Duration::from_millis(100));
        assert!(result.is_ok(), "Display should run without errors");

        // Verify final state
        let (freq, window, peaks) = display.current_state();
        assert_eq!(freq, Some(89_300_000.0), "Should track latest frequency");
        assert_eq!(window, Some(2), "Should track latest window");
        assert_eq!(peaks, 2, "Should count peak events");
    }

    #[test]
    fn test_progress_display_empty_events() {
        let (_sender, receiver) = mpsc::channel();
        let mut display = ProgressDisplay::new(receiver);

        // Run with no events
        let result = display.run_with_timeout(Duration::from_millis(50));
        assert!(result.is_ok(), "Display should handle no events gracefully");

        let (freq, window, peaks) = display.current_state();
        assert_eq!(freq, None, "Should have no frequency");
        assert_eq!(window, None, "Should have no window");
        assert_eq!(peaks, 0, "Should have no peaks");
    }

    #[test]
    fn test_progress_display_event_processing() {
        let (sender, receiver) = mpsc::channel();
        let mut display = ProgressDisplay::new(receiver);

        // Send single peak event
        sender
            .send(ProgressEvent {
                event_type: ProgressEventType::PeakDetected,
                frequency_hz: 88_900_000.0,
                metadata: crate::scanning::window::WindowMetadata {
                    center_frequency_hz: 88_900_000.0,
                    window_id: 3,
                },
                candidate_id: Some("88.9-3".to_string()),
                audio_quality: None,
                signal_strength: None,
                timestamp: Instant::now(),
                tuner_id: None,
            })
            .expect("Should send event");

        drop(sender);

        let result = display.run_with_timeout(Duration::from_millis(50));
        assert!(result.is_ok());

        let (freq, window, peaks) = display.current_state();
        assert_eq!(freq, Some(88_900_000.0));
        assert_eq!(window, Some(3));
        assert_eq!(peaks, 1);
    }
}
