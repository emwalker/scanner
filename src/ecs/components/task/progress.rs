#[derive(Debug, Clone)]
pub struct TaskProgressComponent {
    pub percentage: u8,
    pub status_label: String,
    pub subtasks_total: usize,
    pub subtasks_completed: usize,
}

impl TaskProgressComponent {
    pub fn new(status_label: impl Into<String>, subtasks_total: usize) -> Self {
        let mut component = Self {
            percentage: 0,
            status_label: status_label.into(),
            subtasks_total,
            subtasks_completed: 0,
        };
        component.update_progress();
        component
    }

    pub fn update_progress(&mut self) {
        if self.subtasks_total == 0 {
            self.percentage = 100;
        } else {
            self.percentage =
                ((self.subtasks_completed as f32 / self.subtasks_total as f32) * 100.0) as u8;
        }
    }

    pub fn mark_subtask_complete(&mut self) {
        if self.subtasks_completed < self.subtasks_total {
            self.subtasks_completed += 1;
            self.update_progress();
        }
    }

    pub fn is_all_complete(&self) -> bool {
        self.subtasks_completed >= self.subtasks_total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_calculation() {
        let mut component = TaskProgressComponent::new("scanning", 4);
        assert_eq!(component.percentage, 0);

        component.mark_subtask_complete();
        assert_eq!(component.percentage, 25);

        component.mark_subtask_complete();
        assert_eq!(component.percentage, 50);

        component.mark_subtask_complete();
        assert_eq!(component.percentage, 75);

        component.mark_subtask_complete();
        assert_eq!(component.percentage, 100);
        assert!(component.is_all_complete());
    }

    #[test]
    fn test_zero_subtasks() {
        let component = TaskProgressComponent::new("empty", 0);
        assert_eq!(component.percentage, 100);
        assert!(component.is_all_complete());
    }
}
