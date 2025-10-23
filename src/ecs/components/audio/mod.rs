//! Audio components

use std::sync::atomic::{AtomicU64, Ordering};

mod allocation;
mod duration;
mod playback;
mod queue;
mod request;
mod request_component;
mod stop_listening_request;
mod tuning;

pub use allocation::AudioAllocationComponent;
pub use duration::PlaybackDuration;
pub use playback::AudioPlaybackComponent;
pub use queue::AudioQueueComponent;
pub use request::AudioRequest;
pub use request_component::AudioRequestComponent;
pub use stop_listening_request::StopListeningRequestComponent;
pub use tuning::AudioTuningComponent;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AudioId(u64);

static NEXT_AUDIO_ID: AtomicU64 = AtomicU64::new(1);

impl AudioId {
    pub fn new() -> Self {
        Self(NEXT_AUDIO_ID.fetch_add(1, Ordering::SeqCst))
    }
}

impl Default for AudioId {
    fn default() -> Self {
        Self::new()
    }
}
