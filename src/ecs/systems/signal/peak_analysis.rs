use tracing::debug;

use crate::{
    core::types::Result,
    ecs::{
        Entity,
        system::{System, SystemContext},
    },
};

/// System that initiates signal analysis for detected signals
///
/// Flow:
/// 1. Query SignalEntity where analysis.is_not_started()
/// 2. Spawn thread to analyze signal quality
/// 3. Store thread handle in analysis component
/// 4. Transition to InProgress state
pub struct PeakAnalysisSystem;

impl Default for PeakAnalysisSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl PeakAnalysisSystem {
    pub fn new() -> Self {
        Self
    }
}

impl System for PeakAnalysisSystem {
    fn name(&self) -> &'static str {
        "PeakAnalysis"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        let signal_entities = match &context.signal_entities {
            Some(se) => se.clone(),
            None => return Ok(()),
        };

        // Read-only check first
        let signals_to_analyze = {
            let signals = match signal_entities.try_read() {
                Ok(s) => s,
                Err(_) => return Ok(()),
            };

            signals
                .iter()
                .filter(|s| s.analysis.is_not_started())
                .map(|s| (s.id().clone(), s.frequency()))
                .collect::<Vec<_>>()
        };

        if signals_to_analyze.is_empty() {
            return Ok(());
        }

        debug!(
            count = signals_to_analyze.len(),
            "PeakAnalysisSystem: Found signals (analysis not yet implemented, staying in Detected \
             state)"
        );

        // Phase 2: Implement actual signal analysis here
        // For now, signals remain in NotStarted/Detected state
        // This prevents showing placeholder "Good" quality before actual analysis

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use super::*;
    use crate::{
        core::signals::ModulationType,
        ecs::{EntityWorld, SignalEntity, TaskId, WindowId},
    };

    #[test]
    fn test_system_creation() {
        let system = PeakAnalysisSystem::new();
        assert_eq!(system.name(), "PeakAnalysis");
    }

    #[test]
    fn test_run_with_no_signals() {
        let mut system = PeakAnalysisSystem::new();
        let signal_entities = Arc::new(RwLock::new(EntityWorld::new()));

        let mut context = SystemContext::new().with_signal_entities(signal_entities);

        let result = system.run(&mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_with_not_started_signal() {
        let mut system = PeakAnalysisSystem::new();

        let task_id = TaskId::new("test-scan");
        let window_id = WindowId::new(task_id, 1);
        let signal = SignalEntity::new(88.5e6, window_id, ModulationType::WFM);

        let mut world = EntityWorld::new();
        world.insert(signal);

        let signal_entities = Arc::new(RwLock::new(world));
        let mut context = SystemContext::new().with_signal_entities(signal_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        // Verify signal was found (skeleton doesn't modify yet)
        let signals = signal_entities.read().unwrap();
        assert_eq!(signals.len(), 1);
    }

    #[test]
    fn test_leaves_signals_in_detected_state_until_phase2() {
        let mut system = PeakAnalysisSystem::new();

        let task_id = TaskId::new("test-scan");
        let window_id = WindowId::new(task_id, 1);
        let signal = SignalEntity::new(88.5e6, window_id, ModulationType::WFM);
        assert!(signal.analysis.is_not_started());

        let mut world = EntityWorld::new();
        world.insert(signal);

        let signal_entities = Arc::new(RwLock::new(world));
        let mut context = SystemContext::new().with_signal_entities(signal_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        // Verify signal remains in NotStarted/Detected state (no placeholder analysis)
        let signals = signal_entities.read().unwrap();
        let signal = signals.iter().next().unwrap();
        assert!(
            signal.analysis.is_not_started(),
            "Expected signal to remain in NotStarted state until Phase 2 implements real analysis"
        );
    }
}
