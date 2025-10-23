//! Audio allocation component

use crate::hardware::pool::TunerId;

/// Component tracking resource allocation for audio playback
pub struct AudioAllocationComponent {
    /// ID of the tuner being used (if known)
    pub tuner_id: Option<TunerId>,

    /// Audio graph cancellation token
    pub graph_cancel: Option<rustradio::graph::CancellationToken>,

    /// Audio graph thread handle
    pub graph_thread: Option<std::thread::JoinHandle<()>>,
}

impl std::fmt::Debug for AudioAllocationComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioAllocationComponent")
            .field("tuner_id", &self.tuner_id)
            .field(
                "graph_cancel",
                &self.graph_cancel.as_ref().map(|_| "<token>"),
            )
            .field(
                "graph_thread",
                &self.graph_thread.as_ref().map(|_| "<thread>"),
            )
            .finish()
    }
}

impl AudioAllocationComponent {
    pub fn new(tuner_id: Option<TunerId>) -> Self {
        Self {
            tuner_id,
            graph_cancel: None,
            graph_thread: None,
        }
    }

    pub fn set_graph(
        &mut self,
        cancel: rustradio::graph::CancellationToken,
        thread: std::thread::JoinHandle<()>,
    ) {
        self.graph_cancel = Some(cancel);
        self.graph_thread = Some(thread);
    }

    pub fn cancel_graph(&mut self) {
        if let Some(cancel) = self.graph_cancel.take() {
            cancel.cancel();
        }
    }

    pub fn take_thread(&mut self) -> Option<std::thread::JoinHandle<()>> {
        self.graph_thread.take()
    }

    pub fn has_active_graph(&self) -> bool {
        self.graph_cancel.is_some() || self.graph_thread.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::DeviceId;

    #[test]
    fn test_create_allocation() {
        let device_id = DeviceId::from_serial("test", "device");
        let tuner_id = TunerId::new(device_id, 0);
        let allocation = AudioAllocationComponent::new(Some(tuner_id.clone()));

        assert_eq!(allocation.tuner_id, Some(tuner_id));
        assert!(!allocation.has_active_graph());
    }

    #[test]
    fn test_create_allocation_no_tuner() {
        let allocation = AudioAllocationComponent::new(None);

        assert_eq!(allocation.tuner_id, None);
        assert!(!allocation.has_active_graph());
    }
}
