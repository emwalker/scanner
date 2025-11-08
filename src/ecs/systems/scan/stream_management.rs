//! Audio stream management system - maintains broadcast channel during analysis
//!
//! This system monitors WindowEntity lifecycle states and keeps the broadcast
//! channel open while analysis is happening. When all signals for a window
//! are analyzed, the window entity (and its segment) can be cleaned up.

use tracing::debug;

use crate::{
    core::types::Result,
    ecs::{
        Entity,
        system::{System, SystemContext},
    },
};

pub struct AudioStreamManagementSystem;

impl AudioStreamManagementSystem {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AudioStreamManagementSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl System for AudioStreamManagementSystem {
    fn name(&self) -> &'static str {
        "AudioStreamManagement"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        // Don't process windows during global pause
        // This prevents deallocating windows while paused, which would prevent resume
        if context.is_globally_paused() {
            return Ok(());
        }

        let window_entities = match &context.window_entities {
            Some(entities) => entities.clone(),
            None => return Ok(()),
        };

        let signal_entities = match &context.signal_entities {
            Some(entities) => entities.clone(),
            None => return Ok(()),
        };

        // Check each window's lifecycle and update based on signal analysis progress
        if let Ok(mut windows) = window_entities.try_write() {
            for window in windows.iter_mut() {
                // Skip windows that aren't active anymore
                if !window.allocation.is_active() {
                    continue;
                }

                let lifecycle_is_analyzing = window.lifecycle.is_analyzing();
                let lifecycle_is_complete = window.lifecycle.is_complete();

                // Only proceed if we're still analyzing or the lifecycle believes we're complete
                // but the allocation hasn't been released yet.
                if !lifecycle_is_analyzing && !lifecycle_is_complete {
                    continue;
                }

                // Count how many signals for this window are still being analyzed or played
                if let Ok(signals) = signal_entities.try_read() {
                    let signals_for_window: Vec<_> = signals
                        .iter()
                        .filter(|s| s.discovery.window_id() == window.id())
                        .collect();

                    let is_analysis_done = |s: &&crate::ecs::SignalEntity| {
                        s.analysis.is_confirmed()
                            || s.analysis.is_rejected()
                            || s.analysis.is_error()
                    };

                    debug!(
                        window_index = window.window_index(),
                        total_signals = signals_for_window.len(),
                        in_progress = signals_for_window
                            .iter()
                            .filter(|s| !is_analysis_done(s))
                            .count(),
                        playing = signals_for_window
                            .iter()
                            .filter(|s| s.playback.is_playing())
                            .count(),
                        "AudioStreamManagementSystem: Checking window completion"
                    );

                    let pending_count = signals_for_window
                        .iter()
                        .filter(|s| !is_analysis_done(s) || s.playback.is_playing())
                        .count();

                    // If any signals are still pending, keep analyzing
                    if pending_count > 0 {
                        continue;
                    }

                    // All signals complete - proceed with two-phase cleanup:
                    // 1. Clear Segment when all work done (releases SDR broadcast hub)
                    // 2. Mark window Complete on next iteration (state transitions)
                    //
                    // This ensures Segment stays alive while signals are playing,
                    // preventing "Window segment not available" errors.
                    debug!(
                        window_index = window.window_index(),
                        "All signals for window analyzed, checking completion"
                    );
                    if lifecycle_is_analyzing {
                        window.lifecycle.complete_signal();
                    }

                    // Clear Segment when all work complete (analysis + playback)
                    // This releases the SDR graph and tuner resources
                    if window.allocation.all_work_complete() && window.segment.is_some() {
                        debug!(
                            window_index = window.window_index(),
                            "All work complete (analysis and playback), clearing Segment"
                        );
                        window.segment = None; // Drop Segment, triggers RAII cleanup
                    }

                    // Check if window is ready to complete (all analysis done, no playback pending)
                    let segment_exists = window.segment.is_some();
                    if window.allocation.is_ready_to_complete(segment_exists) {
                        debug!(
                            window_index = window.window_index(),
                            segment_exists = segment_exists,
                            "Window ready to complete - all analysis and playback done"
                        );

                        // Use completion coordinator to ensure all side effects happen atomically
                        let window_id = window.id().clone();
                        crate::ecs::systems::window::completion::complete_window(
                            &window_id, window, context,
                        );

                        // Clear the allocation after completion
                        window.allocation.clear();
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use super::*;
    use crate::{
        audio::quality::AudioQuality,
        core::signals::ModulationType,
        ecs::{EntityWorld, SignalEntity, TaskId, WindowEntity, WindowId},
        hardware::{DeviceId, pool::TunerId},
    };

    #[test]
    fn test_system_name() {
        let system = AudioStreamManagementSystem::new();
        assert_eq!(system.name(), "AudioStreamManagement");
    }

    #[test]
    fn test_system_with_no_windows() {
        let mut system = AudioStreamManagementSystem::new();
        let mut context = SystemContext::new();

        let result = system.run(&mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_system_default() {
        let _system = AudioStreamManagementSystem;
    }

    #[test]
    fn test_system_respects_global_pause() {
        use std::sync::Mutex;

        use crate::ecs::{GlobalPauseState, system::System};

        let mut system = AudioStreamManagementSystem::new();

        // Create a window in Active state with completed signal
        // This window is ready to be marked Complete and deallocated
        let task_id = TaskId::new("scan_1");
        let window_id = WindowId::new(task_id.clone(), 0);
        let mut window = WindowEntity::new(window_id.clone(), task_id.clone(), 88.9e6);

        let device_id = DeviceId::from_serial("sdrplay", "test123");
        let tuner_id = TunerId::new(device_id, 0);
        window.allocation.allocate(tuner_id.clone());
        window.allocation.start_processing(tuner_id.clone());
        window.lifecycle.start_analyzing(0); // Start with 0 signals (will mark as complete)
        window.allocation.start_active(tuner_id.clone(), 0);
        window.allocation.mark_all_spawned();

        // Mark lifecycle as complete (no signals to wait for)
        window.lifecycle.complete_signal();
        // No segment (already cleared, ready to deallocate)
        window.segment = None;

        let window_entities = Arc::new(RwLock::new(EntityWorld::<WindowEntity>::new()));
        {
            let mut windows = window_entities.write().unwrap();
            windows.insert(window);
        }

        let signal_entities = Arc::new(RwLock::new(EntityWorld::<SignalEntity>::new()));

        // Create global pause resource in Paused state
        let global_pause = Arc::new(Mutex::new(GlobalPauseState::Paused {
            had_active_scans: true,
            playing_stations: vec![],
        }));

        let mut context = SystemContext::new()
            .with_window_entities(window_entities.clone())
            .with_signal_entities(signal_entities)
            .with_global_pause_resource(global_pause);

        // Run the system - should return early due to global pause
        // WITHOUT the fix, this would mark the window Complete and deallocate the tuner
        system.run(&mut context).unwrap();

        // Verify window was NOT modified (still Active, not Complete)
        let windows = window_entities.read().unwrap();
        let window = windows.get(&window_id).expect("Window should exist");

        assert!(
            window.allocation.is_active(),
            "BUG: Window should remain Active during global pause (not deallocated). This is the \
             root cause of the pause/resume bug - AudioStreamManagementSystem keeps running \
             during pause and deallocates the window, preventing resume."
        );
        assert!(
            !window.progress.is_completed(),
            "Window should NOT be marked complete during global pause"
        );
        assert!(
            !window.allocation.is_complete(),
            "Window allocation should NOT be marked complete during global pause"
        );
    }

    /// BUG REPRODUCTION: Window won't complete when signals are in Failed state
    /// This reproduces the issue from the screenshot where the scan is stuck at 1/40 (2%)
    /// with 5 "Skipped" signals preventing the window from completing
    /// RED TEST: Window should complete after playback finishes
    /// Reproduces bug where window stays at "All signals analyzed" but never becomes ready
    #[test]
    fn test_window_completes_after_playback_finishes() {
        use crate::ecs::system::System;

        let mut system = AudioStreamManagementSystem::new();

        let task_id = TaskId::new("scan_1");
        let window_id = WindowId::new(task_id.clone(), 0);
        let mut window = WindowEntity::new(window_id.clone(), task_id.clone(), 88.9e6);

        // Allocate tuner and transition to Active
        let device_id = DeviceId::from_serial("sdrplay", "test123");
        let tuner_id = TunerId::new(device_id, 0);
        window.allocation.allocate(tuner_id.clone());
        window.allocation.start_processing(tuner_id.clone());
        window.lifecycle.start_analyzing(1);
        window.allocation.start_active(tuner_id.clone(), 1);
        window.allocation.mark_all_spawned();

        // Create signal that completed with Good quality and was queued for playback
        let mut signal = SignalEntity::new(88.9e6, window_id.clone(), ModulationType::WFM);
        signal.analysis.confirm_analysis(AudioQuality::Good, 0.8);
        signal.info.set_audio_quality(Some(AudioQuality::Good));
        signal.info.set_signal_strength(Some(0.8));

        // Simulate SignalAnalysisSystem having queued it for playback
        window.lifecycle.complete_signal();
        window.allocation.queue_for_playback(signal.id().clone());

        // Simulate AudioPlaybackSystem having started playback
        let signal_id = window.allocation.start_playing_next();
        assert_eq!(signal_id, Some(signal.id().clone()));

        // Simulate playback finishing - AudioPlaybackSystem should call stop_playing()
        // but currently doesn't, which is the bug
        window.allocation.stop_playing();

        let window_entities = Arc::new(RwLock::new(EntityWorld::<WindowEntity>::new()));
        {
            let mut windows = window_entities.write().unwrap();
            windows.insert(window);
        }

        let signal_entities = Arc::new(RwLock::new(EntityWorld::<SignalEntity>::new()));
        {
            let mut signals = signal_entities.write().unwrap();
            signals.insert(signal);
        }

        let mut context = SystemContext::new()
            .with_window_entities(window_entities.clone())
            .with_signal_entities(signal_entities);

        // Run the system - should mark window as complete since all signals done and no playback
        system.run(&mut context).unwrap();

        // TDD RED PHASE: Window should be marked complete
        let windows = window_entities.read().unwrap();
        let window = windows.get(&window_id).expect("Window should exist");

        assert!(
            window.progress.is_completed(),
            "BUG: Window should complete after playback finishes! All signals done \
             (in_progress=0, playing=0) but window stuck. This is why window 1 never starts - \
             tuner never gets deallocated."
        );
    }

    #[test]
    fn test_window_completes_with_failed_signals() {
        use crate::ecs::system::System;

        let mut system = AudioStreamManagementSystem::new();

        // Create a window in Active state with signals
        let task_id = TaskId::new("scan_1");
        let window_id = WindowId::new(task_id.clone(), 0);
        let mut window = WindowEntity::new(window_id.clone(), task_id.clone(), 88.9e6);

        // Allocate tuner and transition to Active state
        let device_id = DeviceId::from_serial("sdrplay", "test123");
        let tuner_id = TunerId::new(device_id, 0);
        window.allocation.allocate(tuner_id.clone());
        window.allocation.start_processing(tuner_id.clone());
        window.lifecycle.start_analyzing(6); // 6 signals total
        window.allocation.start_active(tuner_id, 6);
        window.allocation.mark_all_spawned(); // All signals spawned

        let window_entities = Arc::new(RwLock::new(EntityWorld::<WindowEntity>::new()));
        {
            let mut windows = window_entities.write().unwrap();
            windows.insert(window);
        }

        // Create 6 signals: 5 Failed (Poor quality/Skipped), 1 Complete (Good quality)
        let signal_entities = Arc::new(RwLock::new(EntityWorld::<SignalEntity>::new()));
        {
            let mut signals = signal_entities.write().unwrap();

            // 5 failed signals (Skipped due to Poor quality)
            for i in 0..5 {
                let mut signal = SignalEntity::new(
                    88.0e6 + (i as f64 * 0.1e6),
                    window_id.clone(),
                    ModulationType::WFM,
                );
                signal.analysis.reject_analysis(AudioQuality::Poor, 0.2);
                signals.insert(signal);
            }

            // 1 completed signal (good quality, finished playing)
            let mut signal = SignalEntity::new(88.9e6, window_id.clone(), ModulationType::WFM);
            signal.analysis.confirm_analysis(AudioQuality::Good, 0.8);
            signals.insert(signal);
        }

        // Simulate SignalAnalysisSystem having processed all signals
        {
            let mut windows = window_entities.write().unwrap();
            let window = windows.get_mut(&window_id).unwrap();
            // For each signal, SignalAnalysisSystem calls both:
            // - window.lifecycle.complete_signal() (decrements lifecycle count)
            // - window.allocation.complete_analysis() or queue_for_playback() (decrements analyzing
            //   count)
            for _ in 0..6 {
                window.lifecycle.complete_signal();
                window.allocation.complete_analysis();
            }
        }

        let mut context = SystemContext::new()
            .with_window_entities(window_entities.clone())
            .with_signal_entities(signal_entities);

        // Run the system
        system.run(&mut context).unwrap();

        // TDD GREEN PHASE: With our fix, this should now pass!
        // The window should complete because all signals are done
        // (5 Failed + 1 Complete, none playing)
        let windows = window_entities.read().unwrap();
        let window = windows.get(&window_id).expect("Window should exist");

        assert!(
            window.progress.is_completed(),
            "Window should be marked completed! All signals finished (5 Failed + 1 Complete), and \
             with our fix using is_done() instead of is_complete(), the system now correctly \
             recognizes that Failed signals are terminal states."
        );
    }

    #[test]
    fn test_segment_cleanup_waits_for_playback_completion() {
        use crate::ecs::components::signal::SignalId;

        // This test verifies the fix for the bug where Segment was cleared
        // while signals were still in the playback queue, causing
        // "Window segment not available" errors.

        let task_id = TaskId::new("scan_1");
        let window_id = WindowId::new(task_id.clone(), 8);

        // Create window in Active state with analysis complete
        let mut window = WindowEntity::new(window_id.clone(), task_id.clone(), 96.9e6);

        let device_id = DeviceId::from_serial("sdrplay", "test123");
        let tuner_id = TunerId::new(device_id.clone(), 0);

        window.allocation.allocate(tuner_id.clone());
        window.allocation.start_processing(tuner_id.clone());
        window.lifecycle.start_analyzing(2);
        window.allocation.start_active(tuner_id.clone(), 2);
        window.allocation.mark_all_spawned();

        // Complete all analysis
        window.lifecycle.complete_signal();
        window.lifecycle.complete_signal();
        window.allocation.complete_analysis();
        window.allocation.complete_analysis();

        // Queue signal for playback (simulating signal 96.9 from the bug report)
        let signal_id = SignalId::new(96.9e6, ModulationType::WFM);
        window.allocation.queue_for_playback(signal_id);

        // Verify pre-conditions
        assert!(
            !window.allocation.all_work_complete(),
            "Work should not be complete with playback pending"
        );

        // The key assertion: With the fix, all_work_complete() correctly
        // checks playback_queue.is_empty(), preventing premature Segment cleanup
        assert!(
            !window.allocation.all_work_complete(),
            "BUG FIX: all_work_complete() must check playback queue, not just signals_analyzing"
        );

        // Now clear playback queue (simulate audio playback completion)
        if let crate::ecs::components::window::WindowAllocationComponent::Active {
            playback_queue,
            current_playing,
            ..
        } = &mut window.allocation
        {
            playback_queue.clear();
            *current_playing = None;
        }

        // After playback completes, all work is complete
        assert!(
            window.allocation.all_work_complete(),
            "All work should be complete after playback queue is empty"
        );

        // Verify the window can now be marked ready to complete
        let segment_exists = false; // Simulating Segment cleared after all_work_complete()
        assert!(
            window.allocation.is_ready_to_complete(segment_exists),
            "Window should be ready to complete after all work done and Segment cleared"
        );
    }
}
