use std::time::Instant;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskState {
    Queued,
    Running { started_at: Instant },
    Paused { paused_at: Instant },
    Completed { result: TaskResult },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskResult {
    Success,
    Failed(String),
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct TaskStateComponent {
    pub current: TaskState,
}

impl TaskStateComponent {
    pub fn new() -> Self {
        Self {
            current: TaskState::Queued,
        }
    }

    pub fn start(&mut self) -> Result<(), &'static str> {
        if !matches!(self.current, TaskState::Queued | TaskState::Paused { .. }) {
            return Err("Can only start queued or paused tasks");
        }
        self.current = TaskState::Running {
            started_at: Instant::now(),
        };
        Ok(())
    }

    pub fn pause(&mut self) -> Result<(), &'static str> {
        if !matches!(self.current, TaskState::Running { .. }) {
            return Err("Can only pause running tasks");
        }
        self.current = TaskState::Paused {
            paused_at: Instant::now(),
        };
        Ok(())
    }

    pub fn resume(&mut self) -> Result<(), &'static str> {
        if !matches!(self.current, TaskState::Paused { .. }) {
            return Err("Can only resume paused tasks");
        }
        self.current = TaskState::Running {
            started_at: Instant::now(),
        };
        Ok(())
    }

    pub fn complete(&mut self, result: TaskResult) -> Result<(), &'static str> {
        if matches!(self.current, TaskState::Completed { .. }) {
            return Err("Task already completed");
        }
        self.current = TaskState::Completed { result };
        Ok(())
    }

    pub fn is_running(&self) -> bool {
        matches!(self.current, TaskState::Running { .. })
    }

    pub fn is_completed(&self) -> bool {
        matches!(self.current, TaskState::Completed { .. })
    }
}

impl Default for TaskStateComponent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_state_lifecycle() {
        let mut component = TaskStateComponent::new();
        assert_eq!(component.current, TaskState::Queued);

        component.start().unwrap();
        assert!(component.is_running());

        component.pause().unwrap();
        assert!(!component.is_running());

        component.resume().unwrap();
        assert!(component.is_running());

        component.complete(TaskResult::Success).unwrap();
        assert!(component.is_completed());
    }

    #[test]
    fn test_cannot_start_completed_task() {
        let mut component = TaskStateComponent::new();
        component.start().unwrap();
        component.complete(TaskResult::Success).unwrap();
        assert!(component.start().is_err());
    }
}
