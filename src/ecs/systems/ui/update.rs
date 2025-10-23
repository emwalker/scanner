//! UI update system - syncs entity state to TUI Model

use std::sync::mpsc::Sender;

use indexmap::IndexMap;
use tracing::debug;

use crate::{
    core::types::Result,
    ecs::{
        Entity,
        components::{AnalysisStatus, signal::PlaybackState, window::WindowId},
        system::{System, SystemContext},
    },
    hardware::pool::{PoolStatus, TunerId},
    ui::{
        TuiEvent,
        tui::model::{state::TaskSummary, types::SpectrumStation},
    },
};

/// Signal data for TUI display
#[derive(Debug, Clone)]
pub struct SignalData {
    pub signal_id: String,
    pub frequency_hz: f64,
    pub status: AnalysisStatus,
    pub playback_state: PlaybackState,
    pub completion: f64,
    pub transition_status: Option<String>,
}

/// System that updates TUI model with entity state
///
/// This system:
/// - Queries StationEntity for discovered stations
/// - Queries AudioEntity for active playback
/// - Queries SignalEntity for scanning progress
/// - Queries TunerEntity for pool status
/// - Queries ScanEntity for active tasks
/// - Sends ActiveTunersUpdated events when pool status changes
/// - Populates Model spectrum fields and signal lists
/// - Maintains TEA pattern (Model is single source of UI truth)
pub struct UIUpdateSystem {
    stations: Vec<SpectrumStation>,
    active_frequency: Option<f64>,
    active_tuner_id: Option<TunerId>,
    signals_by_window: IndexMap<WindowId, Vec<SignalData>>,
    tasks: Vec<TaskSummary>,
    tui_event_sender: Option<Sender<TuiEvent>>,
    last_pool_status: Option<PoolStatus>,
}

impl Default for UIUpdateSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl UIUpdateSystem {
    pub fn new() -> Self {
        Self {
            stations: Vec::new(),
            active_frequency: None,
            active_tuner_id: None,
            signals_by_window: IndexMap::new(),
            tasks: Vec::new(),
            tui_event_sender: None,
            last_pool_status: None,
        }
    }

    pub fn with_tui_event_sender(mut self, sender: Sender<TuiEvent>) -> Self {
        self.tui_event_sender = Some(sender);
        self
    }

    /// Get discovered stations for spectrum display
    pub fn stations(&self) -> &[SpectrumStation] {
        &self.stations
    }

    /// Get active audio frequency if playing
    pub fn active_frequency(&self) -> Option<f64> {
        self.active_frequency
    }

    /// Get active tuner ID if playing
    pub fn active_tuner_id(&self) -> Option<&TunerId> {
        self.active_tuner_id.as_ref()
    }

    /// Get signals grouped by window ID (preserves insertion order)
    pub fn signals_by_window(&self) -> &IndexMap<WindowId, Vec<SignalData>> {
        &self.signals_by_window
    }

    /// Get task summaries for active scans
    pub fn tasks(&self) -> &[TaskSummary] {
        &self.tasks
    }

    pub fn current_window_for_task(
        &self,
        task_id: &crate::ecs::TaskId,
        context: &SystemContext,
    ) -> Option<usize> {
        if let Some(ref task_entities) = context.task_entities
            && let Ok(entities) = task_entities.try_read()
        {
            for task in entities.iter() {
                if task.id() == task_id {
                    let crate::ecs::TaskComponents::Scan { progress, .. } = &task.components;
                    return progress.current_window.as_ref().map(|w| w.window_index);
                }
            }
        }
        None
    }

    fn send_pool_status_update(&mut self, status: PoolStatus) {
        debug!(
            total_tuners = status.tuners.len(),
            available_count = status.available_tuner_count,
            allocated_count = status.allocated_tuner_count,
            has_sender = self.tui_event_sender.is_some(),
            "UIUpdateSystem: Pool status changed, attempting to send event"
        );

        for tuner in &status.tuners {
            debug!(
                tuner_id = ?tuner.id,
                state = ?tuner.state,
                activity = ?tuner.activity,
                "UIUpdateSystem: Tuner in status being sent"
            );
        }

        if let Some(ref sender) = self.tui_event_sender {
            match sender.send(TuiEvent::ActiveTunersUpdated {
                status: status.clone(),
            }) {
                Ok(_) => debug!("UIUpdateSystem: Event sent successfully"),
                Err(e) => debug!(error = ?e, "UIUpdateSystem: Failed to send event"),
            }
        } else {
            debug!("UIUpdateSystem: No TUI event sender configured");
        }

        self.last_pool_status = Some(status);
    }
}

impl System for UIUpdateSystem {
    fn name(&self) -> &'static str {
        "UIUpdate"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        self.stations.clear();
        self.active_frequency = None;
        self.active_tuner_id = None;
        self.signals_by_window.clear();
        self.tasks.clear();

        // Extract task summaries from task entities
        if let Some(ref task_entities) = context.task_entities
            && let Ok(tasks) = task_entities.try_read()
        {
            // Build tuner label lookup
            let tuner_labels: std::collections::HashMap<crate::hardware::pool::TunerId, String> =
                if let Some(ref tuner_entities) = context.tuner_entities
                    && let Ok(entities) = tuner_entities.try_lock()
                {
                    entities
                        .iter()
                        .map(|e| (e.id().clone(), e.display_name.name.clone()))
                        .collect()
                } else {
                    std::collections::HashMap::new()
                };

            self.tasks = tasks
                .iter()
                .filter_map(|task| {
                    let scan_components = task.as_scan()?;
                    let crate::ecs::TaskComponents::Scan { tuner, .. } = scan_components;

                    let assigned_tuner = tuner
                        .assigned_tuner
                        .as_ref()
                        .and_then(|tuner_id| tuner_labels.get(tuner_id).cloned());

                    Some(TaskSummary {
                        task_id: task.id().clone(),
                        label: task.label(),
                        summary: task.summary(),
                        activity: task.current_activity(),
                        assigned_tuner,
                        assigned_tuner_id: tuner.assigned_tuner.clone(),
                        window_cell_data: task.window_cell_data(),
                    })
                })
                .collect();
        }

        if let Some(ref _tuner_entities) = context.tuner_entities
            && let Some(ref pool) = context.pool
        {
            let status = pool.status();

            let status_changed = match &self.last_pool_status {
                None => true,
                Some(last) => {
                    last.available_tuner_count != status.available_tuner_count
                        || last.allocated_tuner_count != status.allocated_tuner_count
                        || last.tuners.len() != status.tuners.len()
                        || last
                            .tuners
                            .iter()
                            .zip(&status.tuners)
                            .any(|(a, b)| a.state != b.state || a.activity != b.activity)
                }
            };

            if status_changed {
                self.send_pool_status_update(status);
            }
        }

        // Read from SignalEntity (unified view of signals and stations)
        if let Some(ref signal_entities) = context.signal_entities {
            let entities = signal_entities.read().unwrap();

            // Build frequency-to-transition-status lookup
            let transition_statuses: std::collections::HashMap<u64, String> = entities
                .iter()
                .filter_map(|s| {
                    if s.tune_state.is_transitioning() {
                        Some(((s.frequency() * 1000.0) as u64, "Transitioning".to_string()))
                    } else if s.tune_state.is_request_queued() {
                        Some(((s.frequency() * 1000.0) as u64, "Queued".to_string()))
                    } else if s.tune_state.is_active() {
                        Some(((s.frequency() * 1000.0) as u64, "Active".to_string()))
                    } else {
                        None
                    }
                })
                .collect();

            for entity in entities.iter() {
                // Add to stations list if confirmed
                if entity.analysis.is_confirmed() {
                    self.stations.push(SpectrumStation {
                        frequency_hz: entity.frequency(),
                        signal_strength: entity.info.signal_strength().unwrap_or(0.0) as f32,
                        audio_quality: entity.info.audio_quality(),
                        is_active: entity.playback.is_playing(),
                    });
                }

                // Add to signals list (all signals)
                let window_id = entity.discovery.window_id().clone();
                let freq_key = (entity.frequency() * 1000.0) as u64;
                let transition_status = transition_statuses.get(&freq_key).cloned();

                let signal_data = SignalData {
                    signal_id: entity.id().as_str().to_string(),
                    frequency_hz: entity.frequency(),
                    status: entity.status(),
                    playback_state: entity.playback.state(),
                    completion: entity.completion(),
                    transition_status,
                };

                self.signals_by_window
                    .entry(window_id)
                    .or_default()
                    .push(signal_data);
            }
        }

        if let Some(ref audio_entities) = context.audio_entities {
            let entities = audio_entities.read().unwrap();

            if let Some(audio_entity) = entities.iter().find(|e| e.is_playing()) {
                let freq = audio_entity.frequency();
                self.active_frequency = Some(freq);
                self.active_tuner_id = audio_entity.tuner_id().cloned();

                let mut marked_count = 0;
                for station in &mut self.stations {
                    if (station.frequency_hz - freq).abs() < 1000.0 {
                        station.is_active = true;
                        marked_count += 1;
                    }
                }

                debug!(
                    frequency_mhz = freq / 1e6,
                    marked_count = marked_count,
                    "UIUpdateSystem found active audio and marked stations"
                );
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, RwLock},
        time::SystemTime,
    };

    use super::*;
    use crate::{
        audio::quality::AudioQuality,
        core::types::{ModulationType, Signal},
        ecs::{AudioEntity, EntityWorld},
    };

    fn create_test_signal(frequency: f64) -> Signal {
        Signal {
            frequency_hz: frequency,
            signal_strength: 0.8,
            bandwidth_hz: 200_000.0,
            modulation: ModulationType::WFM,
            audio_sample_rate: 48000,
            detected_at: SystemTime::now(),
            analysis_duration_ms: 100,
            detection_center_freq: frequency,
            audio_quality: AudioQuality::Good,
        }
    }

    #[test]
    fn test_ui_update_system_with_no_entities() {
        let mut system = UIUpdateSystem::new();
        let mut context = SystemContext::new();

        let result = system.run(&mut context);
        assert!(result.is_ok());
        assert_eq!(system.stations().len(), 0);
        assert_eq!(system.active_frequency(), None);
    }

    #[test]
    fn test_ui_update_system_with_stations() {
        let mut system = UIUpdateSystem::new();

        let mut signal_world = EntityWorld::new();

        let scan_id = crate::ecs::TaskId::new("test-scan".to_string());
        let window_id = WindowId::new(scan_id, 0);

        // Create confirmed SignalEntity (which shows up in stations list)
        let mut signal1 = crate::ecs::SignalEntity::new(88.9e6, window_id.clone());
        signal1
            .analysis
            .confirm_analysis(crate::audio::quality::AudioQuality::Good, 0.8);
        signal_world.insert(signal1);

        let mut signal2 = crate::ecs::SignalEntity::new(95.5e6, window_id);
        signal2
            .analysis
            .confirm_analysis(crate::audio::quality::AudioQuality::Good, 0.85);
        signal_world.insert(signal2);

        let signal_entities = Arc::new(RwLock::new(signal_world));
        let mut context = SystemContext::new().with_signal_entities(signal_entities);

        let result = system.run(&mut context);
        assert!(result.is_ok());
        assert_eq!(system.stations().len(), 2);
        assert!(system.stations().iter().any(|s| s.frequency_hz == 88.9e6));
        assert!(system.stations().iter().any(|s| s.frequency_hz == 95.5e6));
    }

    #[test]
    fn test_ui_update_system_with_active_audio() {
        let mut system = UIUpdateSystem::new();

        let scan_id = crate::ecs::TaskId::new("test-scan".to_string());
        let window_id = WindowId::new(scan_id, 0);

        let mut signal_world = EntityWorld::new();
        let mut signal = crate::ecs::SignalEntity::new(88.9e6, window_id);
        signal
            .analysis
            .confirm_analysis(crate::audio::quality::AudioQuality::Good, 0.8);
        signal_world.insert(signal);

        let mut audio_world = EntityWorld::new();
        let test_signal = create_test_signal(88.9e6);
        audio_world.insert(AudioEntity::new(test_signal, 88.9e6, None));

        let signal_entities = Arc::new(RwLock::new(signal_world));
        let audio_entities = Arc::new(RwLock::new(audio_world));

        let mut context = SystemContext::new()
            .with_signal_entities(signal_entities)
            .with_audio_entities(audio_entities);

        let result = system.run(&mut context);
        assert!(result.is_ok());
        assert_eq!(system.stations().len(), 1);
        assert_eq!(system.active_frequency(), Some(88.9e6));
        assert!(system.stations()[0].is_active);
    }

    #[test]
    fn test_ui_update_system_marks_correct_station_active() {
        let mut system = UIUpdateSystem::new();

        let scan_id = crate::ecs::TaskId::new("test-scan".to_string());
        let window_id = WindowId::new(scan_id, 0);

        let mut signal_world = EntityWorld::new();

        let mut signal1 = crate::ecs::SignalEntity::new(88.9e6, window_id.clone());
        signal1
            .analysis
            .confirm_analysis(crate::audio::quality::AudioQuality::Good, 0.8);
        signal_world.insert(signal1);

        let mut signal2 = crate::ecs::SignalEntity::new(95.5e6, window_id);
        signal2
            .analysis
            .confirm_analysis(crate::audio::quality::AudioQuality::Good, 0.85);
        signal_world.insert(signal2);

        let mut audio_world = EntityWorld::new();
        audio_world.insert(AudioEntity::new(create_test_signal(95.5e6), 95.5e6, None));

        let signal_entities = Arc::new(RwLock::new(signal_world));
        let audio_entities = Arc::new(RwLock::new(audio_world));

        let mut context = SystemContext::new()
            .with_signal_entities(signal_entities)
            .with_audio_entities(audio_entities);

        let result = system.run(&mut context);
        assert!(result.is_ok());

        let active_station = system.stations().iter().find(|s| s.is_active).unwrap();
        assert_eq!(active_station.frequency_hz, 95.5e6);

        let inactive_station = system.stations().iter().find(|s| !s.is_active).unwrap();
        assert_eq!(inactive_station.frequency_hz, 88.9e6);
    }
}
