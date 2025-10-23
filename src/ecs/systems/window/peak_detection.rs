use tracing::debug;

use crate::{
    core::types::Result,
    ecs::system::{System, SystemContext},
};

/// System that monitors windows ready for peak detection
///
/// In the current architecture, peak detection happens within Window::process().
/// This system serves as a checkpoint in the ECS pipeline for when peak detection
/// should be triggered or monitored.
///
/// Future work: Decompose Window::process() peak detection logic into this system.
pub struct PeakDetectionSystem;

impl Default for PeakDetectionSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl PeakDetectionSystem {
    pub fn new() -> Self {
        Self
    }
}

impl System for PeakDetectionSystem {
    fn name(&self) -> &'static str {
        "PeakDetection"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        let window_entities = match &context.window_entities {
            Some(we) => we.clone(),
            None => return Ok(()),
        };

        // Find windows with task handles (windows being processed)
        let windows_in_progress = {
            let windows = match window_entities.try_read() {
                Ok(w) => w,
                Err(_) => return Ok(()),
            };

            windows
                .iter()
                .filter(|w| w.has_task() && !w.is_completed() && !w.is_failed())
                .count()
        };

        if windows_in_progress > 0 {
            debug!(
                count = windows_in_progress,
                "PeakDetectionSystem: Monitoring windows with active peak detection"
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
        let system = PeakDetectionSystem::new();
        assert_eq!(system.name(), "PeakDetection");
    }

    #[test]
    fn test_run_with_no_windows() {
        let mut system = PeakDetectionSystem::new();
        let window_entities = Arc::new(RwLock::new(EntityWorld::new()));

        let mut context = SystemContext::new().with_window_entities(window_entities);

        let result = system.run(&mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_with_pending_window() {
        let mut system = PeakDetectionSystem::new();

        let task_id = TaskId::new("scan_1");
        let window_id = WindowId::new(task_id.clone(), 0);
        let window = WindowEntity::new(window_id, task_id, 100.0);

        let mut world = EntityWorld::new();
        world.insert(window);

        let window_entities = Arc::new(RwLock::new(world));

        let mut context = SystemContext::new().with_window_entities(window_entities);

        let result = system.run(&mut context);
        assert!(result.is_ok());
    }
}
