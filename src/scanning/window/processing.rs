//! DEPRECATED: Window processing functions
//!
//! This module contains logic that is being migrated to the ECS system.
//! Phase 3 decomposed peak detection into PeakDetectionSystem and
//! PeakCompletionSystem. The functions in this module will be gradually
//! moved to pure functions in the signal processing module or removed
//! entirely as the ECS migration completes.
//!
//! TODO Phase 3+: Delete this module once WindowProcessingSystem is fully
//! migrated to ECS-based systems.

#[allow(unused_imports)]
use tracing::debug;

use crate::{
    core::types::{Result, ScanningConfig},
    hardware::pool::SegmentTrait,
    pause_signal::PauseSignal,
};

pub fn peaks(
    station_mode: bool,
    center_freq: f64,
    config: &ScanningConfig,
    device: &dyn SegmentTrait,
    pause_signal: Option<PauseSignal>,
) -> Result<Vec<crate::core::types::Peak>> {
    if station_mode {
        debug!(
            "Station mode: Creating direct peak for {:.1} MHz",
            center_freq / 1e6
        );
        Ok(vec![crate::core::types::Peak {
            frequency_hz: center_freq,
            magnitude: 1.0,
        }])
    } else {
        let sdr_rx_for_peaks = device.audio_subscriber();
        crate::signal::collect_peaks(config, sdr_rx_for_peaks, center_freq, pause_signal)
    }
}

pub fn signals_from_peaks(
    station_mode: bool,
    _window_num: usize,
    center_freq: f64,
    config: &ScanningConfig,
    peaks: &[crate::core::types::Peak],
) -> Vec<crate::core::types::Candidate> {
    let mut signals = Vec::new();

    if station_mode {
        debug!(
            "Station mode: Creating direct signal for {:.1} MHz",
            center_freq / 1e6
        );
        signals.push(crate::core::types::Candidate::Fm(
            crate::signal::Candidate {
                frequency_hz: center_freq,
                signal_strength: "Strong".to_string(),
                peak_count: 1,
                max_magnitude: 1.0,
                avg_magnitude: 1.0,
            },
        ));
        return signals;
    }

    for signal in crate::signal::find_signals(peaks, config, center_freq) {
        let signal_freq = signal.frequency_hz();

        let rounded_freq = (signal_freq / 100000.0).round() * 100000.0;
        let frequency_khz = (rounded_freq / 1000.0) as u64;

        let already_processed = {
            match crate::signal::PROCESSED_FREQUENCIES.read() {
                Ok(processed) => processed.contains(&frequency_khz),
                Err(e) => {
                    debug!(
                        error = %e,
                        "Failed to read PROCESSED_FREQUENCIES, assuming not processed"
                    );
                    false
                }
            }
        };

        if already_processed {
            debug!(
                signal_frequency_mhz = signal_freq / 1e6,
                "Skipping signal creation for already processed frequency"
            );
            continue;
        }

        if config.debug.pipeline {
            let frequency_offset = signal_freq - center_freq;
            debug!(
                message = "Signal created",
                signal_frequency_mhz = signal_freq / 1e6,
                window_center_mhz = center_freq / 1e6,
                frequency_offset_khz = frequency_offset / 1e3,
                signal_strength = match &signal {
                    crate::core::types::Candidate::Fm(signal) => &signal.signal_strength,
                }
            );
        }
        signals.push(signal);
    }

    signals
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analysis_thread_returns_results() {
        use crate::{audio::quality::AudioQuality, ecs::components::AnalysisResults};

        let handle = std::thread::spawn(|| -> Result<AnalysisResults> {
            Ok(AnalysisResults {
                quality: AudioQuality::Good,
                strength: 0.8,
            })
        });

        let result = handle.join().unwrap();
        assert!(result.is_ok());
        let analysis = result.unwrap();
        assert_eq!(analysis.quality, AudioQuality::Good);
        assert_eq!(analysis.strength, 0.8);
    }

    #[test]
    fn test_stores_handle_in_signal_entity() {
        use std::sync::{Arc, RwLock};

        use crate::{
            audio::quality::AudioQuality,
            ecs::{
                Entity, EntityWorld, SignalEntity, TaskId, WindowId, components::AnalysisResults,
            },
        };

        let signal_entities = Arc::new(RwLock::new(EntityWorld::new()));

        let task_id = TaskId::new("test-scan");
        let window_id = WindowId::new(task_id, 0);
        let entity = SignalEntity::new(88.9e6, window_id);
        let signal_id = entity.id().clone();
        signal_entities.write().unwrap().insert(entity);

        let (result_tx, result_rx) = std::sync::mpsc::channel();

        let handle = std::thread::spawn(move || -> Result<AnalysisResults> {
            let results = AnalysisResults {
                quality: AudioQuality::Good,
                strength: 0.8,
            };
            let _ = result_tx.send(results.clone());
            Ok(results)
        });

        {
            let mut entities = signal_entities.write().unwrap();
            if let Some(signal) = entities.get_mut(&signal_id) {
                signal.analysis.start_analysis(handle, result_rx);
            }
        }

        let entities = signal_entities.read().unwrap();
        let signal = entities.get(&signal_id).unwrap();
        assert!(signal.analysis.is_in_progress());
    }
}
