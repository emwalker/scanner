use std::fmt;

use crate::{
    ecs::{
        ScanId,
        components::{
            scan::{
                PreviousPauseState, ScanConfigComponent, ScanLifecycleComponent, ScanPauseState,
                ScanProgressComponent, ScanResultsComponent, ScanTunerComponent, ScanType,
            },
            task::{TaskProgressComponent, TaskResultComponent, TaskStateComponent},
        },
        entities::task_components::TaskComponents,
        entity::Entity,
    },
    ui::{frequency_hz_label, frequency_hz_tabular},
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TaskId(pub String);

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl TaskId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn from_scan_id(scan_id: &ScanId) -> Self {
        Self(format!("scan_{}", scan_id.value()))
    }
}

/// Task-specific data for the "Window" column in Activities table
#[derive(Debug, Clone)]
pub enum TaskWindowCell {
    SpectrumBar {
        full_range_hz: (f64, f64),
        current_window_hz: Option<(f64, f64)>,
    },
}

#[derive(Debug, Clone)]
pub enum ScanTaskData {
    Placeholder,
}

#[derive(Debug, Clone)]
pub enum TaskKind {
    Scan(ScanTaskData),
}

pub struct TaskEntity {
    id: TaskId,
    pub kind: TaskKind,
    pub state: TaskStateComponent,
    pub progress: TaskProgressComponent,
    pub result: TaskResultComponent,
    pub components: TaskComponents,
}

impl Entity for TaskEntity {
    type Id = TaskId;

    fn id(&self) -> &Self::Id {
        &self.id
    }
}

impl TaskEntity {
    pub fn new_scan(
        id: TaskId,
        data: ScanTaskData,
        config: ScanConfigComponent,
        subtask_count: usize,
    ) -> Self {
        Self {
            id,
            kind: TaskKind::Scan(data),
            state: TaskStateComponent::new(),
            progress: TaskProgressComponent::new("initializing", subtask_count),
            result: TaskResultComponent::new(),
            components: TaskComponents::Scan {
                config,
                progress: ScanProgressComponent::new(subtask_count),
                results: ScanResultsComponent::new(),
                lifecycle: ScanLifecycleComponent::new(),
                tuner: ScanTunerComponent::new(),
                pause_request: None,
                resume_request: None,
            },
        }
    }

    pub fn new_scan_with_defaults(id: TaskId, data: ScanTaskData, subtask_count: usize) -> Self {
        let config =
            ScanConfigComponent::new(ScanType::Band, 88.0e6, 108.0e6, 1.0e6, 2.4e6, 40.0, 1.0, 3);
        Self::new_scan(id, data, config, subtask_count)
    }

    pub fn is_scan(&self) -> bool {
        matches!(self.kind, TaskKind::Scan(_))
    }

    pub fn as_scan(&self) -> Option<&TaskComponents> {
        if self.is_scan() {
            Some(&self.components)
        } else {
            None
        }
    }

    pub fn label(&self) -> String {
        match &self.kind {
            TaskKind::Scan(_) => {
                format!("Scan {}", self.id.0.split('_').nth(1).unwrap_or("0"))
            }
        }
    }

    pub fn summary(&self) -> String {
        match &self.components {
            TaskComponents::Scan { config, .. } => {
                let band = if config.freq_min >= 88e6 && config.freq_max <= 108e6 {
                    "FM"
                } else if config.freq_min >= 108e6 && config.freq_max <= 137e6 {
                    "Aircraft"
                } else if config.freq_min >= 144e6 && config.freq_max <= 148e6 {
                    "2M"
                } else if config.freq_min >= 162e6 && config.freq_max <= 163e6 {
                    "Weather"
                } else {
                    "SDR"
                };
                format!(
                    "{} • {}–{}",
                    band,
                    frequency_hz_label(config.freq_min),
                    frequency_hz_label(config.freq_max)
                )
            }
        }
    }

    pub fn assigned_tuner(&self) -> Option<String> {
        match &self.components {
            TaskComponents::Scan { tuner, .. } => tuner
                .assigned_tuner
                .as_ref()
                .map(|tuner_id| format!("Tuner {}", tuner_id.channel_index + 1)),
        }
    }

    pub fn window_cell_data(&self) -> TaskWindowCell {
        match &self.components {
            TaskComponents::Scan {
                config, progress, ..
            } => {
                let (start, end) = (config.freq_min, config.freq_max);
                let window_hz = config.step_size;

                let highest_completed = if progress.completed_windows.is_empty() {
                    0
                } else {
                    progress
                        .completed_windows
                        .iter()
                        .map(|w| w.window_index)
                        .max()
                        .unwrap_or(0)
                        + 1
                };

                let center = highest_completed as f64 * window_hz + start;
                let current_window = Some((center - window_hz / 2.0, center + window_hz / 2.0));

                TaskWindowCell::SpectrumBar {
                    full_range_hz: (start, end),
                    current_window_hz: current_window,
                }
            }
        }
    }

    pub fn current_activity(&self) -> String {
        match &self.components {
            TaskComponents::Scan {
                progress, results, ..
            } => match &progress.state {
                ScanPauseState::Pending => "Pending".to_string(),

                ScanPauseState::Scanning => {
                    let progress_pct = progress.progress_percentage();
                    format!(
                        "Scanning {}/{} ({:.0}%)",
                        progress.windows_completed,
                        progress.total_windows,
                        progress_pct * 100.0
                    )
                }

                ScanPauseState::PausedAtWindow { window_id } => {
                    format!(
                        "Paused at {}/{}",
                        window_id.window_index + 1,
                        progress.total_windows
                    )
                }

                ScanPauseState::PausedGlobally {
                    window_id,
                    previous_state,
                } => match previous_state {
                    PreviousPauseState::WasScanning => {
                        format!(
                            "Paused (scanning {}/{})",
                            window_id.window_index + 1,
                            progress.total_windows
                        )
                    }
                    PreviousPauseState::WasListening {
                        station_frequency_hz,
                        ..
                    } => {
                        format!("Paused ({})", frequency_hz_tabular(*station_frequency_hz))
                    }
                },

                ScanPauseState::Listening { .. } => "Listening".to_string(),

                ScanPauseState::Completed => {
                    let station_count = results.stations_discovered;
                    format!(
                        "Done ({} station{})",
                        station_count,
                        if station_count == 1 { "" } else { "s" }
                    )
                }

                ScanPauseState::WaitingForTuner => "Waiting for tuner".to_string(),

                ScanPauseState::TunerOffline => "Tuner offline".to_string(),
            },
        }
    }

    /// Request pause at current window
    pub fn request_pause(&mut self, window_num: usize) {
        let TaskComponents::Scan { pause_request, .. } = &mut self.components;
        *pause_request = Some(crate::ecs::components::scan::PauseRequestComponent::new(
            window_num,
        ));
    }

    /// Request pause and tune to a specific station
    pub fn request_pause_with_station(
        &mut self,
        window_num: usize,
        station_frequency_hz: f64,
        window_center_frequency_hz: f64,
    ) {
        let TaskComponents::Scan { pause_request, .. } = &mut self.components;
        *pause_request = Some(
            crate::ecs::components::scan::PauseRequestComponent::with_station(
                window_num,
                station_frequency_hz,
                window_center_frequency_hz,
            ),
        );
    }

    /// Clear pause request
    pub fn clear_pause_request(&mut self) {
        let TaskComponents::Scan { pause_request, .. } = &mut self.components;
        *pause_request = None;
    }

    /// Request resume from paused state
    pub fn request_resume(&mut self, window_num: usize) {
        let TaskComponents::Scan { resume_request, .. } = &mut self.components;
        *resume_request = Some(crate::ecs::components::scan::ResumeRequestComponent::new(
            window_num,
        ));
    }

    /// Clear resume request
    pub fn clear_resume_request(&mut self) {
        let TaskComponents::Scan { resume_request, .. } = &mut self.components;
        *resume_request = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_id_creation() {
        let id1 = TaskId::new("scan_1");
        let id2 = TaskId::new("scan_2");
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_task_entity_creation() {
        let task =
            TaskEntity::new_scan_with_defaults(TaskId::new("scan_1"), ScanTaskData::Placeholder, 5);

        assert_eq!(task.id().0, "scan_1");
        assert!(task.is_scan());
        assert!(!task.state.is_running());
        assert_eq!(task.progress.subtasks_total, 5);
    }

    #[test]
    fn test_task_state_transitions() {
        use crate::ecs::components::task::TaskResult;

        let mut task =
            TaskEntity::new_scan_with_defaults(TaskId::new("scan_1"), ScanTaskData::Placeholder, 1);

        task.state.start().unwrap();
        assert!(task.state.is_running());

        task.progress.mark_subtask_complete();
        assert!(task.progress.is_all_complete());

        task.state.complete(TaskResult::Success).unwrap();
        assert!(task.state.is_completed());
    }
}
