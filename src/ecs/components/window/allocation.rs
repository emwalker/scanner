//! Window allocation component - tracks tuner allocation for window

use std::time::Instant;

use crate::{
    ecs::SignalId,
    hardware::pool::{TaskRequirements, TunerActivity, TunerId},
};

/// Window tuner allocation state
#[derive(Debug, Clone)]
pub enum WindowAllocationComponent {
    /// No allocation requested yet
    None,
    /// Allocation has been requested
    Requested {
        requirements: TaskRequirements,
        activity: TunerActivity,
        requester_id: String,
    },
    /// Tuner has been allocated
    Allocated { tuner_id: TunerId },
    /// Window task is processing (spawned)
    Processing {
        tuner_id: TunerId,
        task_spawned_at: Instant,
    },
    /// Window is actively analyzing signals and playing audio
    Active {
        tuner_id: TunerId,
        signals_analyzing: usize,
        playback_queue: Vec<SignalId>,
        current_playing: Option<SignalId>,
        all_spawned: bool,
        started_at: Instant,
    },
    /// Window processing is complete, ready for deallocation
    Complete { tuner_id: TunerId },
}

impl WindowAllocationComponent {
    pub fn new() -> Self {
        Self::None
    }

    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    pub fn is_requested(&self) -> bool {
        matches!(self, Self::Requested { .. })
    }

    pub fn is_allocated(&self) -> bool {
        matches!(self, Self::Allocated { .. })
    }

    pub fn is_processing(&self) -> bool {
        matches!(self, Self::Processing { .. })
    }

    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active { .. })
    }

    pub fn is_complete(&self) -> bool {
        matches!(self, Self::Complete { .. })
    }

    pub fn request(
        &mut self,
        requirements: TaskRequirements,
        activity: TunerActivity,
        requester_id: String,
    ) {
        *self = Self::Requested {
            requirements,
            activity,
            requester_id,
        };
    }

    pub fn allocate(&mut self, tuner_id: TunerId) {
        *self = Self::Allocated { tuner_id };
    }

    pub fn clear(&mut self) {
        *self = Self::None;
    }

    pub fn tuner_id(&self) -> Option<&TunerId> {
        match self {
            Self::Allocated { tuner_id } => Some(tuner_id),
            Self::Processing { tuner_id, .. } => Some(tuner_id),
            Self::Active { tuner_id, .. } => Some(tuner_id),
            Self::Complete { tuner_id } => Some(tuner_id),
            _ => None,
        }
    }

    /// Get the currently playing signal, if any
    pub fn current_playing(&self) -> Option<&SignalId> {
        if let Self::Active {
            current_playing, ..
        } = self
        {
            current_playing.as_ref()
        } else {
            None
        }
    }

    /// Transition from Allocated to Processing when task spawned
    pub fn start_processing(&mut self, tuner_id: TunerId) {
        *self = Self::Processing {
            tuner_id,
            task_spawned_at: Instant::now(),
        };
    }

    /// Transition from Processing to Active when signals start being created
    pub fn start_active(&mut self, tuner_id: TunerId, initial_signal_count: usize) {
        tracing::debug!(
            tuner_id = ?tuner_id,
            initial_signal_count = initial_signal_count,
            "WindowAllocation: start_active - initialized signals_analyzing"
        );
        *self = Self::Active {
            tuner_id,
            signals_analyzing: initial_signal_count,
            playback_queue: Vec::new(),
            current_playing: None,
            all_spawned: false,
            started_at: Instant::now(),
        };
    }

    /// Mark all signals as spawned
    pub fn mark_all_spawned(&mut self) {
        if let Self::Active { all_spawned, .. } = self {
            *all_spawned = true;
        }
    }

    /// Add a signal to the playback queue and decrement analyzing count
    pub fn queue_for_playback(&mut self, signal_id: SignalId) {
        if let Self::Active {
            signals_analyzing,
            playback_queue,
            ..
        } = self
        {
            let before = *signals_analyzing;
            tracing::debug!(
                signal_id = ?signal_id,
                before = before,
                "WindowAllocation: queue_for_playback called"
            );
            playback_queue.push(signal_id);
            if *signals_analyzing > 0 {
                *signals_analyzing -= 1;
            }
            tracing::debug!(
                after = *signals_analyzing,
                "WindowAllocation: queue_for_playback decremented signals_analyzing"
            );
        }
    }

    /// Decrement analyzing count without queuing (for rejected signals)
    pub fn complete_analysis(&mut self) {
        if let Self::Active {
            signals_analyzing, ..
        } = self
            && *signals_analyzing > 0
        {
            let before = *signals_analyzing;
            *signals_analyzing -= 1;
            tracing::debug!(
                before = before,
                after = *signals_analyzing,
                "WindowAllocation: complete_analysis decremented signals_analyzing"
            );
        }
    }

    /// Get next signal from queue and mark as playing
    pub fn start_playing_next(&mut self) -> Option<SignalId> {
        if let Self::Active {
            playback_queue,
            current_playing,
            ..
        } = self
            && current_playing.is_none()
            && !playback_queue.is_empty()
        {
            let next = playback_queue.remove(0);
            *current_playing = Some(next.clone());
            return Some(next);
        }
        None
    }

    /// Clear current playing when audio completes
    pub fn stop_playing(&mut self) {
        if let Self::Active {
            current_playing, ..
        } = self
        {
            *current_playing = None;
        }
    }

    /// Return a signal to the playback queue (e.g., when spawning fails)
    pub fn return_playback_signal(&mut self, signal_id: SignalId) {
        if let Self::Active {
            playback_queue,
            current_playing,
            ..
        } = self
        {
            if current_playing
                .as_ref()
                .map(|current| current == &signal_id)
                .unwrap_or(false)
            {
                *current_playing = None;
            }

            if playback_queue.iter().all(|queued| queued != &signal_id) {
                playback_queue.insert(0, signal_id);
            }
        }
    }

    /// Check if all work for window is complete
    ///
    /// Returns true when:
    /// - All signals have been spawned
    /// - No signals are analyzing
    /// - Playback queue is empty
    /// - No signal is currently playing
    ///
    /// This represents completion of all analysis and playback work,
    /// regardless of whether the Segment resource has been cleared.
    pub fn all_work_complete(&self) -> bool {
        if let Self::Active {
            signals_analyzing,
            playback_queue,
            current_playing,
            all_spawned,
            ..
        } = self
        {
            *all_spawned
                && *signals_analyzing == 0
                && playback_queue.is_empty()
                && current_playing.is_none()
        } else {
            false
        }
    }

    /// Check if window is ready to complete
    ///
    /// A window is ready to complete when:
    /// - All signals have been spawned
    /// - No signals are analyzing
    /// - No signals are in playback queue
    /// - No signal is currently playing
    /// - Segment has been cleared (segment_exists = false)
    ///
    /// The segment_exists parameter ensures windows stay Active while
    /// their Segment resource is still alive, preventing the next window
    /// from starting and causing resource conflicts.
    pub fn is_ready_to_complete(&self, segment_exists: bool) -> bool {
        self.all_work_complete() && !segment_exists
    }

    /// Check if window has exceeded timeout (120 seconds)
    pub fn is_timed_out(&self) -> bool {
        if let Self::Active { started_at, .. } = self {
            started_at.elapsed().as_secs() > 120
        } else {
            false
        }
    }

    /// Transition to Complete state
    pub fn mark_complete(&mut self) {
        if let Some(tuner_id) = self.tuner_id().cloned() {
            *self = Self::Complete { tuner_id };
        }
    }
}

impl Default for WindowAllocationComponent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::signals::ModulationType;

    #[test]
    fn test_allocation_lifecycle() {
        let mut allocation = WindowAllocationComponent::new();
        assert!(allocation.is_none());

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 2.0e6,
            required_sample_rate: 2.0e6,
            priority: crate::hardware::pool::TaskPriority::Normal,
        };

        allocation.request(requirements, TunerActivity::Scanning, "test".to_string());
        assert!(allocation.is_requested());

        let device_id = crate::hardware::DeviceId::from_serial("mock", "dev1");
        let tuner_id = TunerId::new(device_id, 0);
        allocation.allocate(tuner_id.clone());
        assert!(allocation.is_allocated());
        assert_eq!(allocation.tuner_id(), Some(&tuner_id));

        allocation.clear();
        assert!(allocation.is_none());
    }

    #[test]
    fn test_window_stays_active_while_segment_exists() {
        let device_id = crate::hardware::DeviceId::from_serial("mock", "dev1");
        let tuner_id = TunerId::new(device_id, 0);

        let mut allocation = WindowAllocationComponent::new();
        allocation.start_active(tuner_id.clone(), 5);
        allocation.mark_all_spawned();

        // Complete all analysis
        for _ in 0..5 {
            allocation.complete_analysis();
        }

        // Window should NOT be ready to complete while segment exists
        assert!(
            !allocation.is_ready_to_complete(true),
            "Window should not be ready to complete while segment exists"
        );

        // Window SHOULD be ready to complete when segment is gone
        assert!(
            allocation.is_ready_to_complete(false),
            "Window should be ready to complete when segment is cleared"
        );
    }

    #[test]
    fn test_all_work_complete_requires_empty_playback_queue() {
        let device_id = crate::hardware::DeviceId::from_serial("mock", "dev1");
        let tuner_id = TunerId::new(device_id, 0);

        let mut allocation = WindowAllocationComponent::new();
        allocation.start_active(tuner_id.clone(), 2);
        allocation.mark_all_spawned();

        // Complete all analysis
        allocation.complete_analysis();
        allocation.complete_analysis();

        // Queue a signal for playback
        let signal_id = SignalId::new(96.9e6, ModulationType::WFM);
        allocation.queue_for_playback(signal_id.clone());

        // All analysis done, but playback pending - NOT complete
        assert!(
            !allocation.all_work_complete(),
            "Should not be complete with signal in playback queue"
        );

        // Remove from playback queue (simulate playback completion)
        if let WindowAllocationComponent::Active {
            playback_queue,
            current_playing,
            ..
        } = &mut allocation
        {
            playback_queue.clear();
            *current_playing = None;
        }

        // Now all work is complete
        assert!(
            allocation.all_work_complete(),
            "Should be complete when playback queue empty"
        );
    }
}
