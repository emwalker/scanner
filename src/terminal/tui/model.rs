//! TUI data model using The Elm Architecture pattern

use crate::terminal::{ProgressEvent, ProgressEventType};
use std::{
    collections::{BTreeMap, HashMap},
    time::Instant,
};

/// Information about a candidate's progress
#[derive(Debug, Clone)]
pub struct CandidateProgress {
    pub frequency_hz: f64,
    #[allow(dead_code)] // Kept for debugging and potential future use
    pub window_id: usize,
    pub completion: f64,
    pub status: CandidateStatus,
    pub audio_quality: Option<crate::audio_quality::AudioQuality>,
    pub last_update: Instant,
}

/// Information about a scanning window
#[derive(Debug, Clone)]
pub struct WindowProgress {
    #[allow(dead_code)] // Kept for debugging and potential future use
    pub window_id: usize,
    pub candidates: Vec<CandidateProgress>,
    pub is_complete: bool,
    pub candidate_lookup: HashMap<String, usize>, // candidate_id -> index in candidates vec
}

impl WindowProgress {
    /// Check if this window should be displayed in the UI
    /// Returns false if all candidates are rejected (noise) and window is complete
    pub fn should_display(&self) -> bool {
        // Always show incomplete windows
        if !self.is_complete {
            return true;
        }

        // For complete windows, only show if there's at least one non-rejected candidate
        self.candidates
            .iter()
            .any(|candidate| candidate.status != CandidateStatus::Rejected)
    }

    /// Get candidates that should be displayed for this window
    /// For complete windows with signals, hide rejected candidates
    /// For current window, show all candidates
    pub fn displayable_candidates(&self, is_current_window: bool) -> Vec<&CandidateProgress> {
        if !self.is_complete || is_current_window {
            // Show all candidates for incomplete windows or current window
            self.candidates.iter().collect()
        } else {
            // For completed non-current windows, only show non-rejected candidates
            self.candidates
                .iter()
                .filter(|candidate| candidate.status != CandidateStatus::Rejected)
                .collect()
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CandidateStatus {
    Detected,
    Analyzing,
    Rejected,
    Signal,
    Playing,
    Completed,
}

impl CandidateStatus {
    pub fn to_string(&self) -> &'static str {
        match self {
            CandidateStatus::Detected => "DETECTED",
            CandidateStatus::Analyzing => "ANALYZING",
            CandidateStatus::Rejected => "NOISE",
            CandidateStatus::Signal => "SIGNAL",
            CandidateStatus::Playing => "PLAYING",
            CandidateStatus::Completed => "DONE",
        }
    }
}

/// Main application model following The Elm Architecture
#[derive(Debug)]
pub struct Model {
    pub windows: BTreeMap<usize, WindowProgress>, // window_id -> WindowProgress (ordered by window_id)
    pub current_window: usize,
    pub should_quit: bool,
    pub theme_selector_open: bool,
    pub theme_selector_index: usize,
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

impl Model {
    pub fn new() -> Self {
        Self {
            windows: BTreeMap::new(),
            current_window: 0,
            should_quit: false,
            theme_selector_open: false,
            theme_selector_index: 0,
        }
    }

    /// Update the model based on a progress event
    pub fn update(&mut self, event: ProgressEvent) {
        // Only process events for actual candidates, not peaks
        if let ProgressEventType::PeakDetected = event.event_type {
            // Ignore peak detection events - we only care about candidates
            return;
        }

        // Track current window for determining when to freeze older windows
        if event.window_id > self.current_window {
            self.current_window = event.window_id;
            // Mark previous windows as complete to prevent further updates
            for (window_id, window) in self.windows.iter_mut() {
                if *window_id < self.current_window {
                    window.is_complete = true;
                }
            }
        }

        if let Some(candidate_id) = &event.candidate_id {
            // Don't process events for old windows at all
            if event.window_id < self.current_window {
                // Ignore events for completed windows
                return;
            }

            // Get or create window (only for current or future windows)
            let window = self
                .windows
                .entry(event.window_id)
                .or_insert_with(|| WindowProgress {
                    window_id: event.window_id,
                    candidates: Vec::new(),
                    is_complete: false,
                    candidate_lookup: HashMap::new(),
                });

            // Find or create candidate
            let candidate_index = if let Some(&index) = window.candidate_lookup.get(candidate_id) {
                index
            } else {
                // Create new candidate
                let new_candidate = CandidateProgress {
                    frequency_hz: event.frequency_hz,
                    window_id: event.window_id,
                    completion: 0.0,
                    status: CandidateStatus::Detected,
                    audio_quality: None,
                    last_update: Instant::now(),
                };
                let index = window.candidates.len();
                window.candidates.push(new_candidate);
                window.candidate_lookup.insert(candidate_id.clone(), index);
                index
            };

            let candidate = &mut window.candidates[candidate_index];

            // Update candidate based on event type
            match event.event_type {
                ProgressEventType::CandidateCreated => {
                    candidate.status = CandidateStatus::Detected;
                    candidate.completion = 0.3;
                }
                ProgressEventType::AudioAnalysisStarted => {
                    candidate.status = CandidateStatus::Analyzing;
                    candidate.completion = 0.5;
                }
                ProgressEventType::AudioAnalysisCompleted => {
                    // Don't override completion if we already have a Signal status with its own progress
                    if candidate.status == CandidateStatus::Signal {
                        // Keep the Signal completion percentage (60%)
                    } else if candidate.status != CandidateStatus::Rejected {
                        // For non-Signal, non-Rejected candidates, mark as complete
                        candidate.status = CandidateStatus::Signal;
                        candidate.completion = 0.6; // 60% - analysis complete, signal found
                    } else {
                        // For rejected candidates, set to 100%
                        candidate.completion = 1.0;
                    }
                }
                ProgressEventType::CandidateRejected => {
                    candidate.status = CandidateStatus::Rejected;
                    candidate.completion = 1.0;
                }
                ProgressEventType::SignalGenerated => {
                    candidate.status = CandidateStatus::Signal;
                    candidate.completion = 0.6; // 60% - signal found but not yet playing
                }
                ProgressEventType::AudioPlaybackStarted => {
                    candidate.status = CandidateStatus::Playing;
                    candidate.completion = 0.8; // 80% - now playing audio
                }
                ProgressEventType::AudioPlaybackCompleted => {
                    candidate.status = CandidateStatus::Completed;
                    candidate.completion = 1.0; // 100% - finished playing, completely done
                }
                ProgressEventType::ThreadCompleted => {
                    // This might not have a candidate ID, ignore for now
                }
                ProgressEventType::PeakDetected => {
                    // Already handled above
                }
            }

            // Update audio quality if provided in the event
            if let Some(quality) = event.audio_quality {
                candidate.audio_quality = Some(quality);
            }

            candidate.last_update = Instant::now();
        }
    }

    /// Check if all windows are empty
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty() || self.windows.values().all(|w| w.candidates.is_empty())
    }

    /// Check if all candidates are complete
    pub fn all_complete(&self) -> bool {
        !self.windows.is_empty()
            && self.windows.values().all(|window| {
                window.candidates.iter().all(|candidate| {
                    candidate.completion >= 1.0
                        && (candidate.status == CandidateStatus::Completed
                            || candidate.status == CandidateStatus::Rejected)
                })
            })
    }

    /// Get total candidate count across all windows
    pub fn candidate_count(&self) -> usize {
        self.windows.values().map(|w| w.candidates.len()).sum()
    }

    /// Request to quit the application
    pub fn quit(&mut self) {
        self.should_quit = true;
    }

    pub fn toggle_theme_selector(&mut self) {
        self.theme_selector_open = !self.theme_selector_open;
    }

    pub fn close_theme_selector(&mut self) {
        self.theme_selector_open = false;
    }

    pub fn theme_selector_next(&mut self, theme_count: usize) {
        if self.theme_selector_open {
            self.theme_selector_index = (self.theme_selector_index + 1) % theme_count;
        }
    }

    pub fn theme_selector_prev(&mut self, theme_count: usize) {
        if self.theme_selector_open {
            self.theme_selector_index = (self.theme_selector_index + theme_count - 1) % theme_count;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terminal::{ProgressEvent, ProgressEventType};
    use std::time::Instant;

    /// Test that candidates progress through all expected states
    #[test]
    fn test_complete_candidate_lifecycle() {
        let mut model = Model::new();
        let candidate_id = "88.9-1".to_string();
        let frequency = 88_900_000.0;
        let window_id = 1;

        // Step 1: Candidate created
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: frequency,
            window_id,
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Detected);
        assert_eq!(candidate.completion, 0.3); // 30%

        // Step 2: Audio analysis started
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioAnalysisStarted,
            frequency_hz: frequency,
            window_id,
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Analyzing);
        assert_eq!(candidate.completion, 0.5); // 50%

        // Step 3: Signal generated (good signal path)
        model.update(ProgressEvent {
            event_type: ProgressEventType::SignalGenerated,
            frequency_hz: frequency,
            window_id,
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Signal);
        assert_eq!(candidate.completion, 0.6); // 60%

        // Step 4: Audio playback started
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: frequency,
            window_id,
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Playing);
        assert_eq!(candidate.completion, 0.8); // 80%

        // Step 5: Audio playback completed
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackCompleted,
            frequency_hz: frequency,
            window_id,
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Completed);
        assert_eq!(candidate.completion, 1.0); // 100%
    }

    /// Test that rejected candidates reach terminal state correctly
    #[test]
    fn test_rejected_candidate_lifecycle() {
        let mut model = Model::new();
        let candidate_id = "88.9-1".to_string();
        let frequency = 88_900_000.0;
        let window_id = 1;

        // Step 1: Candidate created
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: frequency,
            window_id,
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        // Step 2: Audio analysis started
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioAnalysisStarted,
            frequency_hz: frequency,
            window_id,
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        // Step 3: Candidate rejected (noise)
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateRejected,
            frequency_hz: frequency,
            window_id,
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Rejected);
        assert_eq!(candidate.completion, 1.0); // 100% - terminal state
    }

    /// Test that no candidates remain stuck in intermediate states
    #[test]
    fn test_no_stuck_intermediate_states() {
        let mut model = Model::new();
        let window_id = 1;

        // Create multiple candidates in different states
        let candidates = vec![
            ("88.1-1", 88_100_000.0),
            ("88.3-1", 88_300_000.0),
            ("88.5-1", 88_500_000.0),
            ("88.7-1", 88_700_000.0),
            ("88.9-1", 88_900_000.0),
        ];

        // Create all candidates
        for (id, freq) in &candidates {
            model.update(ProgressEvent {
                event_type: ProgressEventType::CandidateCreated,
                frequency_hz: *freq,
                window_id,
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                timestamp: Instant::now(),
            });
        }

        // Start analysis for all
        for (id, freq) in &candidates {
            model.update(ProgressEvent {
                event_type: ProgressEventType::AudioAnalysisStarted,
                frequency_hz: *freq,
                window_id,
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                timestamp: Instant::now(),
            });
        }

        // Resolve all candidates to terminal states
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateRejected,
            frequency_hz: candidates[0].1,
            window_id,
            candidate_id: Some(candidates[0].0.to_string()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateRejected,
            frequency_hz: candidates[1].1,
            window_id,
            candidate_id: Some(candidates[1].0.to_string()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        // Complete signal paths for others
        for (id, freq) in &candidates[2..] {
            model.update(ProgressEvent {
                event_type: ProgressEventType::SignalGenerated,
                frequency_hz: *freq,
                window_id,
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                timestamp: Instant::now(),
            });

            model.update(ProgressEvent {
                event_type: ProgressEventType::AudioPlaybackStarted,
                frequency_hz: *freq,
                window_id,
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                timestamp: Instant::now(),
            });

            model.update(ProgressEvent {
                event_type: ProgressEventType::AudioPlaybackCompleted,
                frequency_hz: *freq,
                window_id,
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                timestamp: Instant::now(),
            });
        }

        // Verify no candidates are stuck in intermediate states
        let window = model.windows.get(&window_id).unwrap();
        for candidate in &window.candidates {
            match candidate.status {
                CandidateStatus::Detected | CandidateStatus::Analyzing => {
                    panic!(
                        "Candidate at {:.1} MHz stuck in intermediate state: {:?}",
                        candidate.frequency_hz / 1e6,
                        candidate.status
                    );
                }
                CandidateStatus::Rejected | CandidateStatus::Completed => {
                    // Terminal states are good
                    assert_eq!(candidate.completion, 1.0);
                }
                CandidateStatus::Signal | CandidateStatus::Playing => {
                    // These are valid but should have progressed to Completed
                    panic!(
                        "Candidate at {:.1} MHz should have completed: {:?}",
                        candidate.frequency_hz / 1e6,
                        candidate.status
                    );
                }
            }
        }
    }

    /// Test that windows complete sequentially, not overlapping
    #[test]
    fn test_sequential_window_completion() {
        let mut model = Model::new();

        // Create candidates in window 1
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 88_900_000.0,
            window_id: 1,
            candidate_id: Some("88.9-1".to_string()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        assert_eq!(model.current_window, 1);
        assert!(!model.windows.get(&1).unwrap().is_complete);

        // Start window 2 - should mark window 1 as complete
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 89_100_000.0,
            window_id: 2,
            candidate_id: Some("89.1-2".to_string()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        assert_eq!(model.current_window, 2);
        assert!(model.windows.get(&1).unwrap().is_complete);
        assert!(!model.windows.get(&2).unwrap().is_complete);

        // Start window 3 - should mark window 2 as complete
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 89_300_000.0,
            window_id: 3,
            candidate_id: Some("89.3-3".to_string()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        assert_eq!(model.current_window, 3);
        assert!(model.windows.get(&1).unwrap().is_complete);
        assert!(model.windows.get(&2).unwrap().is_complete);
        assert!(!model.windows.get(&3).unwrap().is_complete);
    }

    /// Test that old window events are ignored after window completion
    #[test]
    fn test_old_window_events_ignored() {
        let mut model = Model::new();

        // Create candidate in window 1
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 88_900_000.0,
            window_id: 1,
            candidate_id: Some("88.9-1".to_string()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        // Start window 2 (marks window 1 complete)
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 89_100_000.0,
            window_id: 2,
            candidate_id: Some("89.1-2".to_string()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        let window1_candidate_count = model.windows.get(&1).unwrap().candidates.len();

        // Try to add another candidate to completed window 1 - should be ignored
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 88_700_000.0,
            window_id: 1,
            candidate_id: Some("88.7-1".to_string()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        // Window 1 should still have the same number of candidates
        assert_eq!(
            model.windows.get(&1).unwrap().candidates.len(),
            window1_candidate_count
        );

        // Try to update existing candidate in window 1 - should be ignored
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioAnalysisStarted,
            frequency_hz: 88_900_000.0,
            window_id: 1,
            candidate_id: Some("88.9-1".to_string()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        // Candidate should still be in original state
        let candidate = &model.windows.get(&1).unwrap().candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Detected);
        assert_eq!(candidate.completion, 0.3);
    }

    /// Test window filtering behavior - only non-rejected candidates shown for complete windows
    #[test]
    fn test_window_candidate_filtering() {
        let mut model = Model::new();
        let window_id = 1;

        // Create multiple candidates
        let candidates = vec![
            ("88.1-1", 88_100_000.0),
            ("88.3-1", 88_300_000.0),
            ("88.5-1", 88_500_000.0),
        ];

        for (id, freq) in &candidates {
            model.update(ProgressEvent {
                event_type: ProgressEventType::CandidateCreated,
                frequency_hz: *freq,
                window_id,
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                timestamp: Instant::now(),
            });
        }

        // Reject first candidate, complete others
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateRejected,
            frequency_hz: candidates[0].1,
            window_id,
            candidate_id: Some(candidates[0].0.to_string()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        for (id, freq) in &candidates[1..] {
            model.update(ProgressEvent {
                event_type: ProgressEventType::SignalGenerated,
                frequency_hz: *freq,
                window_id,
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                timestamp: Instant::now(),
            });

            model.update(ProgressEvent {
                event_type: ProgressEventType::AudioPlaybackStarted,
                frequency_hz: *freq,
                window_id,
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                timestamp: Instant::now(),
            });

            model.update(ProgressEvent {
                event_type: ProgressEventType::AudioPlaybackCompleted,
                frequency_hz: *freq,
                window_id,
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                timestamp: Instant::now(),
            });
        }

        // Mark window complete by starting window 2
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 89_100_000.0,
            window_id: 2,
            candidate_id: Some("89.1-2".to_string()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        let window = model.windows.get(&window_id).unwrap();
        assert!(window.is_complete);

        // Test current window shows all candidates
        let current_displayable = window.displayable_candidates(true);
        assert_eq!(current_displayable.len(), 3); // All candidates

        // Test completed window only shows non-rejected candidates
        let completed_displayable = window.displayable_candidates(false);
        assert_eq!(completed_displayable.len(), 2); // Only non-rejected

        // Verify the rejected candidate is filtered out
        for candidate in completed_displayable {
            assert_ne!(candidate.status, CandidateStatus::Rejected);
        }
    }

    /// Test that window should_display logic works correctly
    #[test]
    fn test_window_display_logic() {
        let mut model = Model::new();
        let window_id = 1;

        // Create window with all rejected candidates
        let candidates = vec![("88.1-1", 88_100_000.0), ("88.3-1", 88_300_000.0)];

        for (id, freq) in &candidates {
            model.update(ProgressEvent {
                event_type: ProgressEventType::CandidateCreated,
                frequency_hz: *freq,
                window_id,
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                timestamp: Instant::now(),
            });

            model.update(ProgressEvent {
                event_type: ProgressEventType::CandidateRejected,
                frequency_hz: *freq,
                window_id,
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                timestamp: Instant::now(),
            });
        }

        let window = model.windows.get(&window_id).unwrap();

        // Incomplete window should always display
        assert!(window.should_display());

        // Mark window complete by starting window 2
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 89_100_000.0,
            window_id: 2,
            candidate_id: Some("89.1-2".to_string()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        let window = model.windows.get(&window_id).unwrap();
        assert!(window.is_complete);

        // Complete window with only rejected candidates should not display
        assert!(!window.should_display());
    }

    /// Test deterministic candidate ordering within windows
    #[test]
    fn test_deterministic_candidate_ordering() {
        let mut model = Model::new();
        let window_id = 1;

        // Create candidates in specific order
        let candidates = vec![
            ("89.1-1", 89_100_000.0),
            ("88.3-1", 88_300_000.0),
            ("90.5-1", 90_500_000.0),
            ("87.9-1", 87_900_000.0),
        ];

        for (id, freq) in &candidates {
            model.update(ProgressEvent {
                event_type: ProgressEventType::CandidateCreated,
                frequency_hz: *freq,
                window_id,
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                timestamp: Instant::now(),
            });
        }

        let window = model.windows.get(&window_id).unwrap();

        // Candidates should maintain insertion order
        assert_eq!(window.candidates.len(), 4);
        assert_eq!(window.candidates[0].frequency_hz, 89_100_000.0);
        assert_eq!(window.candidates[1].frequency_hz, 88_300_000.0);
        assert_eq!(window.candidates[2].frequency_hz, 90_500_000.0);
        assert_eq!(window.candidates[3].frequency_hz, 87_900_000.0);

        // displayable_candidates should also maintain this order
        let displayable = window.displayable_candidates(true);
        assert_eq!(displayable.len(), 4);
        assert_eq!(displayable[0].frequency_hz, 89_100_000.0);
        assert_eq!(displayable[1].frequency_hz, 88_300_000.0);
        assert_eq!(displayable[2].frequency_hz, 90_500_000.0);
        assert_eq!(displayable[3].frequency_hz, 87_900_000.0);
    }

    /// Test model utility functions
    #[test]
    fn test_model_utility_functions() {
        let mut model = Model::new();

        // Empty model - all_complete returns false for empty models
        assert!(model.is_empty());
        assert!(!model.all_complete()); // Empty model returns false for all_complete
        assert_eq!(model.candidate_count(), 0);

        // Add some candidates
        let window_id = 1;
        let candidates = vec![("88.1-1", 88_100_000.0), ("88.3-1", 88_300_000.0)];

        for (id, freq) in &candidates {
            model.update(ProgressEvent {
                event_type: ProgressEventType::CandidateCreated,
                frequency_hz: *freq,
                window_id,
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                timestamp: Instant::now(),
            });
        }

        // Model with incomplete candidates
        assert!(!model.is_empty());
        assert!(!model.all_complete());
        assert_eq!(model.candidate_count(), 2);

        // Complete all candidates
        for (id, freq) in &candidates {
            model.update(ProgressEvent {
                event_type: ProgressEventType::CandidateRejected,
                frequency_hz: *freq,
                window_id,
                candidate_id: Some(id.to_string()),
                audio_quality: None,
                timestamp: Instant::now(),
            });
        }

        // Model with complete candidates
        assert!(!model.is_empty());
        assert!(model.all_complete());
        assert_eq!(model.candidate_count(), 2);
    }

    /// Test quit functionality
    #[test]
    fn test_quit_functionality() {
        let mut model = Model::new();

        assert!(!model.should_quit);

        model.quit();

        assert!(model.should_quit);
    }

    /// Test AudioAnalysisCompleted event handling preserves Signal status
    #[test]
    fn test_audio_analysis_completed_preserves_signal() {
        let mut model = Model::new();
        let candidate_id = "88.9-1".to_string();
        let frequency = 88_900_000.0;
        let window_id = 1;

        // Create candidate and start analysis
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: frequency,
            window_id,
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioAnalysisStarted,
            frequency_hz: frequency,
            window_id,
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        // Generate signal first
        model.update(ProgressEvent {
            event_type: ProgressEventType::SignalGenerated,
            frequency_hz: frequency,
            window_id,
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Signal);
        assert_eq!(candidate.completion, 0.6);

        // AudioAnalysisCompleted should not override Signal status
        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioAnalysisCompleted,
            frequency_hz: frequency,
            window_id,
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.status, CandidateStatus::Signal);
        assert_eq!(candidate.completion, 0.6); // Should remain unchanged
    }

    /// Test that status text mapping remains exactly the same
    #[test]
    fn test_status_text_mapping_unchanged() {
        // These exact strings must be preserved across refactoring
        assert_eq!(CandidateStatus::Detected.to_string(), "DETECTED");
        assert_eq!(CandidateStatus::Analyzing.to_string(), "ANALYZING");
        assert_eq!(CandidateStatus::Rejected.to_string(), "NOISE");
        assert_eq!(CandidateStatus::Signal.to_string(), "SIGNAL");
        assert_eq!(CandidateStatus::Playing.to_string(), "PLAYING");
        assert_eq!(CandidateStatus::Completed.to_string(), "DONE");
    }

    /// Test that progress percentage calculations remain exact
    #[test]
    fn test_progress_percentages_unchanged() {
        let mut model = Model::new();
        let candidate_id = "88.9-1".to_string();
        let frequency = 88_900_000.0;
        let window_id = 1;

        // Test each state's exact completion percentage
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: frequency,
            window_id,
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.completion, 0.3); // DETECTED = 30%

        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioAnalysisStarted,
            frequency_hz: frequency,
            window_id,
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.completion, 0.5); // ANALYZING = 50%

        model.update(ProgressEvent {
            event_type: ProgressEventType::SignalGenerated,
            frequency_hz: frequency,
            window_id,
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.completion, 0.6); // SIGNAL = 60%

        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackStarted,
            frequency_hz: frequency,
            window_id,
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.completion, 0.8); // PLAYING = 80%

        model.update(ProgressEvent {
            event_type: ProgressEventType::AudioPlaybackCompleted,
            frequency_hz: frequency,
            window_id,
            candidate_id: Some(candidate_id.clone()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        let window = model.windows.get(&window_id).unwrap();
        let candidate = &window.candidates[0];
        assert_eq!(candidate.completion, 1.0); // DONE = 100%

        // Test rejected path
        let rejected_id = "89.1-1".to_string();
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: 89_100_000.0,
            window_id,
            candidate_id: Some(rejected_id.clone()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateRejected,
            frequency_hz: 89_100_000.0,
            window_id,
            candidate_id: Some(rejected_id.clone()),
            audio_quality: None,
            timestamp: Instant::now(),
        });

        let window = model.windows.get(&window_id).unwrap();
        let rejected_candidate = &window.candidates[1];
        assert_eq!(rejected_candidate.completion, 1.0); // NOISE = 100%
    }
}
