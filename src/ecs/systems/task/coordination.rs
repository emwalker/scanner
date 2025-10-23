use crate::{
    core::types::Result,
    ecs::{
        entity::Entity,
        system::{System, SystemContext},
    },
};

pub struct TaskCoordinationSystem;

impl System for TaskCoordinationSystem {
    fn name(&self) -> &'static str {
        "TaskCoordinationSystem"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        let task_entities = match &context.task_entities {
            Some(entities) => entities.clone(),
            None => return Ok(()),
        };

        let window_entities = match &context.window_entities {
            Some(entities) => entities.clone(),
            None => return Ok(()),
        };

        let mut tasks = task_entities.write().unwrap();
        let windows = window_entities.read().unwrap();

        for task in tasks.iter_mut() {
            if !task.state.is_running() {
                continue;
            }

            let task_windows: Vec<_> = windows.iter().filter(|w| w.task_id == *task.id()).collect();

            let completed = task_windows
                .iter()
                .filter(|w| w.progress.is_completed())
                .count();

            task.progress.subtasks_completed = completed;
            task.progress.subtasks_total = task_windows.len();
            task.progress.update_progress();

            if !task_windows.is_empty() && completed == task_windows.len() {
                use crate::ecs::components::task::TaskResult;
                let _ = task.state.complete(TaskResult::Success);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::{
        components::window::WindowId,
        entities::{ScanTaskData, TaskEntity, TaskId, WindowEntity},
        world::EntityWorld,
    };

    #[test]
    fn test_task_coordination_updates_progress() {
        let mut system = TaskCoordinationSystem;
        let mut context = SystemContext::new();

        let task_id = TaskId::new("scan_1");
        let mut task =
            TaskEntity::new_scan_with_defaults(task_id.clone(), ScanTaskData::Placeholder, 2);
        task.state.start().unwrap();

        let mut window1 =
            WindowEntity::new(WindowId::new(task_id.clone(), 0), task_id.clone(), 88.0e6);
        window1.progress.mark_completed();

        let window2 = WindowEntity::new(WindowId::new(task_id.clone(), 1), task_id.clone(), 88.5e6);

        let mut task_world = EntityWorld::new();
        task_world.insert(task);

        let mut window_world = EntityWorld::new();
        window_world.insert(window1);
        window_world.insert(window2);

        let task_entities = std::sync::Arc::new(std::sync::RwLock::new(task_world));
        let window_entities = std::sync::Arc::new(std::sync::RwLock::new(window_world));

        context = context
            .with_task_entities(task_entities.clone())
            .with_window_entities(window_entities);

        system.run(&mut context).unwrap();

        let tasks = task_entities.read().unwrap();
        let task = tasks.iter().next().unwrap();
        assert_eq!(task.progress.percentage, 50);
    }
}
