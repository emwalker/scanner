use tracing::debug;

use crate::{
    core::types::Result,
    ecs::system::{System, SystemContext},
};

/// System that monitors completion of window peak detection
///
/// In the current architecture, peak detection and signal creation happens
/// within Window::process(). This system serves as a checkpoint for when windows
/// have completed processing and signals are ready for analysis.
///
/// Future work: Integrate with decomposed peak detection to manage signal
/// creation as a separate ECS operation.
pub struct PeakCompletionSystem;

impl Default for PeakCompletionSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl PeakCompletionSystem {
    pub fn new() -> Self {
        Self
    }
}

impl System for PeakCompletionSystem {
    fn name(&self) -> &'static str {
        "PeakCompletion"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        let window_entities = match &context.window_entities {
            Some(we) => we.clone(),
            None => return Ok(()),
        };

        // Find windows that have completed processing
        let completed_windows = {
            let windows = match window_entities.try_read() {
                Ok(w) => w,
                Err(_) => return Ok(()),
            };

            windows.iter().filter(|w| w.is_completed()).count()
        };

        if completed_windows > 0 {
            debug!(
                count = completed_windows,
                "PeakCompletionSystem: Found windows with completed peak detection"
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use super::*;
    use crate::ecs::{EntityWorld, TaskId, WindowEntity, WindowId};

    #[test]
    fn test_system_creation() {
        let system = PeakCompletionSystem::new();
        assert_eq!(system.name(), "PeakCompletion");
    }

    #[test]
    fn test_run_with_no_windows() {
        let mut system = PeakCompletionSystem::new();
        let window_entities = Arc::new(RwLock::new(EntityWorld::new()));

        let mut context = SystemContext::new().with_window_entities(window_entities);

        let result = system.run(&mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_with_completed_window() {
        let mut system = PeakCompletionSystem::new();

        let task_id = TaskId::new("scan_1");
        let window_id = WindowId::new(task_id.clone(), 0);
        let mut window = WindowEntity::new(window_id, task_id, 100.0);

        window.progress.mark_completed();

        let mut world = EntityWorld::new();
        world.insert(window);

        let window_entities = Arc::new(RwLock::new(world));

        let mut context = SystemContext::new().with_window_entities(window_entities);

        let result = system.run(&mut context);
        assert!(result.is_ok());
    }
}
