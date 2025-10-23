use crate::ecs::components::scan::{
    PauseRequestComponent, ResumeRequestComponent, ScanConfigComponent, ScanLifecycleComponent,
    ScanProgressComponent, ScanResultsComponent, ScanTunerComponent,
};

#[derive(Debug)]
pub enum TaskComponents {
    Scan {
        config: ScanConfigComponent,
        progress: ScanProgressComponent,
        results: ScanResultsComponent,
        lifecycle: ScanLifecycleComponent,
        tuner: ScanTunerComponent,
        pause_request: Option<PauseRequestComponent>,
        resume_request: Option<ResumeRequestComponent>,
    },
}

impl TaskComponents {
    pub fn is_scan(&self) -> bool {
        matches!(self, TaskComponents::Scan { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::components::scan::ScanType;

    #[test]
    fn test_task_components_is_scan() {
        let components = TaskComponents::Scan {
            config: ScanConfigComponent::new(
                ScanType::Band,
                88.0e6,
                108.0e6,
                1.0e6,
                2.4e6,
                40.0,
                1.0,
                3,
            ),
            progress: ScanProgressComponent::new(1),
            results: ScanResultsComponent::new(),
            lifecycle: ScanLifecycleComponent::new(),
            tuner: ScanTunerComponent::new(),
            pause_request: None,
            resume_request: None,
        };

        assert!(components.is_scan());
    }
}
