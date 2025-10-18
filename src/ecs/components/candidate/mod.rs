//! Components for candidate entities

mod id;
mod info;
mod lifecycle;
mod progress;

pub use id::CandidateId;
pub use info::CandidateInfoComponent;
pub use lifecycle::{CandidateLifecycleComponent, CandidateState};
pub use progress::CandidateProgressComponent;
