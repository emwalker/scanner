#[derive(Debug, Clone)]
pub struct TaskResultComponent {
    pub value: Option<TaskResultValue>,
}

#[derive(Debug, Clone)]
pub enum TaskResultValue {
    Success,
    Failed(String),
    Cancelled,
}

impl TaskResultComponent {
    pub fn new() -> Self {
        Self { value: None }
    }

    pub fn set_success(&mut self) {
        self.value = Some(TaskResultValue::Success);
    }

    pub fn set_failed(&mut self, reason: impl Into<String>) {
        self.value = Some(TaskResultValue::Failed(reason.into()));
    }

    pub fn set_cancelled(&mut self) {
        self.value = Some(TaskResultValue::Cancelled);
    }

    pub fn is_success(&self) -> bool {
        matches!(self.value, Some(TaskResultValue::Success))
    }

    pub fn is_failed(&self) -> bool {
        matches!(self.value, Some(TaskResultValue::Failed(_)))
    }
}

impl Default for TaskResultComponent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_component() {
        let mut component = TaskResultComponent::new();
        assert!(component.value.is_none());

        component.set_success();
        assert!(component.is_success());

        let mut component2 = TaskResultComponent::new();
        component2.set_failed("out of memory");
        assert!(component2.is_failed());
    }
}
