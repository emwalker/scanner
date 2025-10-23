use std::{thread::JoinHandle, time::Instant};

use crate::core::types::Result;

#[derive(Debug, Clone)]
pub struct Peak {
    pub frequency_hz: f64,
    pub magnitude: f64,
}

#[derive(Debug)]
pub enum PeakDetectionState {
    Pending,
    InProgress {
        thread_handle: JoinHandle<Result<Vec<Peak>>>,
        started_at: Instant,
    },
    Complete {
        peaks: Vec<Peak>,
    },
    Failed {
        error: String,
    },
}

#[derive(Debug)]
pub struct PeakDetectionComponent {
    state: PeakDetectionState,
}

impl Default for PeakDetectionComponent {
    fn default() -> Self {
        Self::new()
    }
}

impl PeakDetectionComponent {
    pub fn new() -> Self {
        Self {
            state: PeakDetectionState::Pending,
        }
    }

    pub fn state(&self) -> &PeakDetectionState {
        &self.state
    }

    pub fn is_pending(&self) -> bool {
        matches!(self.state, PeakDetectionState::Pending)
    }

    pub fn is_in_progress(&self) -> bool {
        matches!(self.state, PeakDetectionState::InProgress { .. })
    }

    pub fn is_complete(&self) -> bool {
        matches!(self.state, PeakDetectionState::Complete { .. })
    }

    pub fn is_failed(&self) -> bool {
        matches!(self.state, PeakDetectionState::Failed { .. })
    }

    pub fn start_detection(&mut self, handle: JoinHandle<Result<Vec<Peak>>>) {
        self.state = PeakDetectionState::InProgress {
            thread_handle: handle,
            started_at: Instant::now(),
        };
    }

    pub fn complete_detection(&mut self, peaks: Vec<Peak>) {
        self.state = PeakDetectionState::Complete { peaks };
    }

    pub fn fail_detection(&mut self, error: String) {
        self.state = PeakDetectionState::Failed { error };
    }

    pub fn take_handle(&mut self) -> Option<JoinHandle<Result<Vec<Peak>>>> {
        if let PeakDetectionState::InProgress { .. } = &self.state {
            let old_state = std::mem::replace(&mut self.state, PeakDetectionState::Pending);
            if let PeakDetectionState::InProgress { thread_handle, .. } = old_state {
                return Some(thread_handle);
            }
        }
        None
    }

    pub fn peaks(&self) -> Option<&Vec<Peak>> {
        if let PeakDetectionState::Complete { peaks } = &self.state {
            Some(peaks)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_default_state() {
        let component = PeakDetectionComponent::new();
        assert!(component.is_pending());
    }

    #[test]
    fn test_state_transition_to_in_progress() {
        let mut component = PeakDetectionComponent::new();

        let handle = std::thread::spawn(|| Ok(vec![]));
        component.start_detection(handle);

        assert!(component.is_in_progress());
    }

    #[test]
    fn test_state_transition_to_complete() {
        let mut component = PeakDetectionComponent::new();

        let peaks = vec![
            Peak {
                frequency_hz: 88.1e6,
                magnitude: 0.8,
            },
            Peak {
                frequency_hz: 88.9e6,
                magnitude: 0.9,
            },
        ];

        component.complete_detection(peaks.clone());

        assert!(component.is_complete());
        assert_eq!(component.peaks().unwrap().len(), 2);
    }

    #[test]
    fn test_state_transition_to_failed() {
        let mut component = PeakDetectionComponent::new();

        component.fail_detection("FFT error".to_string());

        assert!(component.is_failed());
    }

    #[test]
    fn test_take_handle() {
        let mut component = PeakDetectionComponent::new();

        let handle = std::thread::spawn(|| Ok(vec![]));
        component.start_detection(handle);

        let taken = component.take_handle();
        assert!(taken.is_some());

        // State should be reset after taking handle
        assert!(component.is_pending());
    }
}
