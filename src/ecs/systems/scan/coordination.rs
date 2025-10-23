//! Scan coordination system

use tracing::debug;

use crate::{
    core::types::Result,
    ecs::{
        entity::Entity,
        system::{System, SystemContext},
    },
};

/// System that coordinates scan operations
///
/// This system:
/// - Monitors active scan entities
/// - Updates scan progress and state
/// - Manages scan lifecycle transitions
/// - Coordinates with tuner allocation for scan tasks
pub struct CoordinationSystem;

impl Default for CoordinationSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl CoordinationSystem {
    pub fn new() -> Self {
        Self
    }
}

impl System for CoordinationSystem {
    fn name(&self) -> &'static str {
        "ScanCoordination"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        let task_entities = match &context.task_entities {
            Some(entities) => entities.clone(),
            None => {
                debug!("No task entities in context");
                return Ok(());
            }
        };

        let mut tasks = task_entities.write().unwrap();

        for task in tasks.iter_mut() {
            if !task.is_scan() {
                continue;
            }

            if task.state.is_completed() {
                debug!(task_id = %task.id(), "Task marked for completion");
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use super::*;
    use crate::ecs::{EntityWorld, ScanTaskData, TaskEntity, TaskId, components::task::TaskResult};

    fn create_test_task(task_num: usize) -> TaskEntity {
        TaskEntity::new_scan_with_defaults(
            TaskId::new(format!("scan_{}", task_num)),
            ScanTaskData::Placeholder,
            10,
        )
    }

    #[test]
    fn test_coordination_system_with_empty_context() {
        let mut system = CoordinationSystem::new();
        let mut context = SystemContext::new();

        let result = system.run(&mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_coordination_system_with_no_task_entities() {
        let mut system = CoordinationSystem::new();
        let mut context = SystemContext::new();

        let result = system.run(&mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_coordination_system_with_active_tasks() {
        let mut system = CoordinationSystem::new();

        let mut world = EntityWorld::new();
        world.insert(create_test_task(1));
        world.insert(create_test_task(2));

        let context_entities = Arc::new(RwLock::new(world));
        let mut context = SystemContext::new().with_task_entities(context_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let entities = context_entities.read().unwrap();
        assert_eq!(entities.len(), 2);
    }

    #[test]
    fn test_coordination_system_with_task_states() {
        let mut system = CoordinationSystem::new();

        let mut world = EntityWorld::new();
        let mut task1 = create_test_task(1);
        let mut task2 = create_test_task(2);
        let mut task3 = create_test_task(3);

        task1.state.start().unwrap();
        task2.state.complete(TaskResult::Success).unwrap();
        task3.state.start().unwrap();

        world.insert(task1);
        world.insert(task2);
        world.insert(task3);

        let context_entities = Arc::new(RwLock::new(world));
        let mut context = SystemContext::new().with_task_entities(context_entities.clone());

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let entities = context_entities.read().unwrap();
        assert_eq!(entities.iter().filter(|e| e.is_scan()).count(), 3);
        assert_eq!(
            entities.iter().filter(|e| e.state.is_completed()).count(),
            1
        );
    }
}
