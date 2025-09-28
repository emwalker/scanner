//! Progress reporting infrastructure for tracking scanning operations
//!
//! This module provides traits and implementations for reporting progress
//! during scanning operations, enabling real-time feedback to users.

pub mod display;
pub mod tracking;
pub mod tui;

use std::sync::{Arc, Mutex};

/// Trait for reporting progress events during scanning operations
pub trait ProgressReporter: Send + Sync {
    /// Report a progress event
    fn report(&self, event: ProgressEvent);
}

/// A progress event representing a milestone in the scanning process
#[derive(Debug, Clone)]
pub struct ProgressEvent {
    pub event_type: ProgressEventType,
    pub frequency_hz: f64,
    pub window_id: usize,
    pub candidate_id: Option<String>,
    pub audio_quality: Option<crate::audio_quality::AudioQuality>,
    pub timestamp: std::time::Instant,
}

/// Types of progress events that can be reported
#[derive(Debug, Clone)]
pub enum ProgressEventType {
    PeakDetected,
    CandidateCreated,
    AudioAnalysisStarted,
    AudioAnalysisCompleted,
    CandidateRejected,
    SignalGenerated,
    AudioPlaybackStarted,
    AudioPlaybackCompleted,
    ThreadCompleted,
}

/// No-operation progress reporter that does nothing (default behavior)
pub struct NoOpProgressReporter;

impl ProgressReporter for NoOpProgressReporter {
    fn report(&self, _event: ProgressEvent) {
        // Do nothing - maintains existing behavior
    }
}

/// Channel-based progress reporter that sends events via mpsc channel
pub struct ChannelProgressReporter {
    sender: std::sync::mpsc::Sender<ProgressEvent>,
}

impl ChannelProgressReporter {
    pub fn new(sender: std::sync::mpsc::Sender<ProgressEvent>) -> Self {
        Self { sender }
    }
}

impl ProgressReporter for ChannelProgressReporter {
    fn report(&self, event: ProgressEvent) {
        // Send event through channel - ignore errors if receiver is dropped
        let _ = self.sender.send(event);
    }
}

/// Mock progress reporter for testing that captures events
#[derive(Clone)]
pub struct MockProgressReporter {
    events: Arc<Mutex<Vec<ProgressEvent>>>,
}

impl Default for MockProgressReporter {
    fn default() -> Self {
        Self::new()
    }
}

impl MockProgressReporter {
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Get all captured events
    pub fn get_events(&self) -> Vec<ProgressEvent> {
        self.events.lock().unwrap().clone()
    }

    /// Get the count of captured events
    pub fn event_count(&self) -> usize {
        self.events.lock().unwrap().len()
    }

    /// Clear all captured events
    pub fn clear(&self) {
        self.events.lock().unwrap().clear();
    }
}

impl ProgressReporter for MockProgressReporter {
    fn report(&self, event: ProgressEvent) {
        self.events.lock().unwrap().push(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_reporter_interface() {
        let mock_reporter = MockProgressReporter::new();

        // Verify initial state
        assert_eq!(mock_reporter.event_count(), 0);
        assert!(mock_reporter.get_events().is_empty());

        // Report a test event
        let event = ProgressEvent {
            event_type: ProgressEventType::PeakDetected,
            frequency_hz: 88_900_000.0,
            window_id: 1,
            candidate_id: Some("88.9-1".to_string()),
            audio_quality: None,
            timestamp: std::time::Instant::now(),
        };

        mock_reporter.report(event.clone());

        // Verify event was captured
        assert_eq!(mock_reporter.event_count(), 1);
        let events = mock_reporter.get_events();
        assert_eq!(events.len(), 1);

        let captured_event = &events[0];
        assert_eq!(captured_event.frequency_hz, 88_900_000.0);
        assert_eq!(captured_event.window_id, 1);
        match captured_event.event_type {
            ProgressEventType::PeakDetected => {}
            _ => panic!("Expected PeakDetected event type"),
        }
    }

    #[test]
    fn test_mock_progress_reporter_multiple_events() {
        let mock_reporter = MockProgressReporter::new();

        // Report multiple events
        let events = vec![
            ProgressEvent {
                event_type: ProgressEventType::PeakDetected,
                frequency_hz: 88_900_000.0,
                window_id: 1,
                candidate_id: Some("88.9-1".to_string()),
                audio_quality: None,
                timestamp: std::time::Instant::now(),
            },
            ProgressEvent {
                event_type: ProgressEventType::CandidateCreated,
                frequency_hz: 88_900_000.0,
                window_id: 1,
                candidate_id: Some("88.9-1".to_string()),
                audio_quality: None,
                timestamp: std::time::Instant::now(),
            },
            ProgressEvent {
                event_type: ProgressEventType::ThreadCompleted,
                frequency_hz: 88_900_000.0,
                window_id: 1,
                candidate_id: None,
                audio_quality: None,
                timestamp: std::time::Instant::now(),
            },
        ];

        for event in events {
            mock_reporter.report(event);
        }

        // Verify all events were captured
        assert_eq!(mock_reporter.event_count(), 3);
        let captured_events = mock_reporter.get_events();
        assert_eq!(captured_events.len(), 3);

        // Verify event types in order
        match captured_events[0].event_type {
            ProgressEventType::PeakDetected => {}
            _ => panic!("Expected PeakDetected"),
        }
        match captured_events[1].event_type {
            ProgressEventType::CandidateCreated => {}
            _ => panic!("Expected CandidateCreated"),
        }
        match captured_events[2].event_type {
            ProgressEventType::ThreadCompleted => {}
            _ => panic!("Expected ThreadCompleted"),
        }
    }

    #[test]
    fn test_mock_progress_reporter_clear() {
        let mock_reporter = MockProgressReporter::new();

        // Add some events
        mock_reporter.report(ProgressEvent {
            event_type: ProgressEventType::PeakDetected,
            frequency_hz: 88_900_000.0,
            window_id: 1,
            candidate_id: Some("88.9-1".to_string()),
            audio_quality: None,
            timestamp: std::time::Instant::now(),
        });

        assert_eq!(mock_reporter.event_count(), 1);

        // Clear events
        mock_reporter.clear();

        // Verify events are cleared
        assert_eq!(mock_reporter.event_count(), 0);
        assert!(mock_reporter.get_events().is_empty());
    }

    #[test]
    fn test_channel_progress_reporter() {
        use std::sync::mpsc;

        let (sender, receiver) = mpsc::channel();
        let channel_reporter = ChannelProgressReporter::new(sender);

        // Report a test event
        let event = ProgressEvent {
            event_type: ProgressEventType::PeakDetected,
            frequency_hz: 88_900_000.0,
            window_id: 1,
            candidate_id: Some("88.9-1".to_string()),
            audio_quality: None,
            timestamp: std::time::Instant::now(),
        };

        channel_reporter.report(event.clone());

        // Verify event was sent through channel
        let received_event = receiver.recv().expect("Should receive event");
        assert_eq!(received_event.frequency_hz, 88_900_000.0);
        assert_eq!(received_event.window_id, 1);
        match received_event.event_type {
            ProgressEventType::PeakDetected => {}
            _ => panic!("Expected PeakDetected event type"),
        }

        // Report multiple events
        let events = vec![
            ProgressEvent {
                event_type: ProgressEventType::CandidateCreated,
                frequency_hz: 89_100_000.0,
                window_id: 2,
                candidate_id: Some("89.1-2".to_string()),
                audio_quality: None,
                timestamp: std::time::Instant::now(),
            },
            ProgressEvent {
                event_type: ProgressEventType::ThreadCompleted,
                frequency_hz: 89_100_000.0,
                window_id: 2,
                candidate_id: None,
                audio_quality: None,
                timestamp: std::time::Instant::now(),
            },
        ];

        for event in events {
            channel_reporter.report(event);
        }

        // Verify both events were received
        let event1 = receiver.recv().expect("Should receive first event");
        let event2 = receiver.recv().expect("Should receive second event");

        assert_eq!(event1.frequency_hz, 89_100_000.0);
        assert_eq!(event2.frequency_hz, 89_100_000.0);
        match event1.event_type {
            ProgressEventType::CandidateCreated => {}
            _ => panic!("Expected CandidateCreated"),
        }
        match event2.event_type {
            ProgressEventType::ThreadCompleted => {}
            _ => panic!("Expected ThreadCompleted"),
        }
    }

    #[test]
    fn test_channel_progress_reporter_dropped_receiver() {
        use std::sync::mpsc;

        let (sender, receiver) = mpsc::channel();
        let channel_reporter = ChannelProgressReporter::new(sender);

        // Drop receiver to simulate display thread exiting
        drop(receiver);

        // Reporting should not panic even if receiver is dropped
        let event = ProgressEvent {
            event_type: ProgressEventType::PeakDetected,
            frequency_hz: 88_900_000.0,
            window_id: 1,
            candidate_id: Some("88.9-1".to_string()),
            audio_quality: None,
            timestamp: std::time::Instant::now(),
        };

        channel_reporter.report(event); // Should not panic
    }

    #[test]
    fn test_no_op_progress_reporter() {
        let no_op_reporter = NoOpProgressReporter;

        // Should not panic or cause issues
        no_op_reporter.report(ProgressEvent {
            event_type: ProgressEventType::PeakDetected,
            frequency_hz: 88_900_000.0,
            window_id: 1,
            candidate_id: Some("88.9-1".to_string()),
            audio_quality: None,
            timestamp: std::time::Instant::now(),
        });

        // No way to verify it did nothing, but it shouldn't crash
    }
}
