//! Tuning coordination system - cleanup stale allocations

use std::time::Duration;

use tracing::debug;

use crate::{
    core::types::Result,
    ecs::{
        Entity,
        components::station::TuneState,
        system::{System, SystemContext},
    },
};

pub struct TuningCoordinationSystem {
    timeout_secs: u64,
}

impl TuningCoordinationSystem {
    pub fn new() -> Self {
        TuningCoordinationSystem { timeout_secs: 30 }
    }

    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }
}

impl System for TuningCoordinationSystem {
    fn name(&self) -> &'static str {
        "TuningCoordinationSystem"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        let signal_entities = match &context.signal_entities {
            Some(entities) => entities.clone(),
            None => return Ok(()),
        };

        let mut signals = match signal_entities.try_write() {
            Ok(s) => s,
            Err(_) => return Ok(()),
        };

        for signal in signals.iter_mut() {
            match &signal.tune_state {
                TuneState::RequestQueued { allocation, .. } => {
                    // Check if allocation has timed out
                    if allocation.state_changed_at().elapsed()
                        > Duration::from_secs(self.timeout_secs)
                    {
                        debug!(
                            signal_id = ?signal.id(),
                            timeout_secs = self.timeout_secs,
                            "TuningCoordinationSystem: Allocation timeout, clearing"
                        );
                        signal.tune_state = TuneState::Idle;
                    }
                }
                TuneState::Active { .. } => {
                    // Check if audio entity still exists
                    let audio_gone = if let Some(ref audio_entities) = context.audio_entities {
                        if let Ok(audios) = audio_entities.try_read() {
                            audios.iter().all(|a| a.playback.is_playing())
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                    if audio_gone {
                        debug!(
                            signal_id = ?signal.id(),
                            "TuningCoordinationSystem: Audio entity gone, clearing allocation"
                        );
                        signal.tune_state = TuneState::Idle;
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }
}

impl Default for TuningCoordinationSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_name() {
        let system = TuningCoordinationSystem::new();
        assert_eq!(system.name(), "TuningCoordinationSystem");
    }

    #[test]
    fn test_system_runs_with_empty_context() {
        let mut system = TuningCoordinationSystem::new();
        let mut context = SystemContext::new();
        let result = system.run(&mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_timeout_configuration() {
        let system = TuningCoordinationSystem::new().with_timeout(60);
        assert_eq!(system.timeout_secs, 60);
    }
}
