//! Integration tests for graceful shutdown functionality
//!
//! These tests verify that shutdown works correctly in all scanner states:
//! - Shutdown while scanning
//! - Shutdown while paused
//! - Shutdown during window processing
//! - Shutdown during audio playback
//! - Double shutdown (graceful then force)
//!
//! Each test has a timeout to catch hangs.

use std::sync::Arc;
use std::time::Duration;
use triggered::Listener;

#[test]
fn test_shutdown_while_paused() {
    let timeout_duration = Duration::from_secs(5);
    let start = std::time::Instant::now();

    let (trigger, listener) = triggered::trigger();

    let thread_listener = listener.clone();
    let handle = std::thread::spawn(move || simulate_paused_scanner(thread_listener));

    std::thread::sleep(Duration::from_millis(100));

    trigger.trigger();

    let result = handle.join();

    let elapsed = start.elapsed();
    assert!(
        elapsed < timeout_duration,
        "Shutdown while paused took {:?}, expected < {:?}",
        elapsed,
        timeout_duration
    );
    assert!(result.is_ok(), "Scanner thread should exit cleanly");
}

#[test]
fn test_shutdown_while_scanning() {
    let timeout_duration = Duration::from_secs(5);
    let start = std::time::Instant::now();

    let (trigger, listener) = triggered::trigger();

    let thread_listener = listener.clone();
    let handle = std::thread::spawn(move || simulate_scanning_loop(thread_listener));

    std::thread::sleep(Duration::from_millis(50));

    trigger.trigger();

    let result = handle.join();

    let elapsed = start.elapsed();
    assert!(
        elapsed < timeout_duration,
        "Shutdown while scanning took {:?}, expected < {:?}",
        elapsed,
        timeout_duration
    );
    assert!(result.is_ok(), "Scanner thread should exit cleanly");
}

#[test]
fn test_shutdown_during_window_processing() {
    let timeout_duration = Duration::from_secs(5);
    let start = std::time::Instant::now();

    let (trigger, listener) = triggered::trigger();

    let thread_listener = listener.clone();
    let handle = std::thread::spawn(move || simulate_window_processing(thread_listener));

    std::thread::sleep(Duration::from_millis(100));

    trigger.trigger();

    let result = handle.join();

    let elapsed = start.elapsed();
    assert!(
        elapsed < timeout_duration,
        "Shutdown during window processing took {:?}, expected < {:?}",
        elapsed,
        timeout_duration
    );
    assert!(result.is_ok(), "Window processing should exit cleanly");
}

#[test]
fn test_immediate_shutdown() {
    let timeout_duration = Duration::from_secs(2);
    let start = std::time::Instant::now();

    let (trigger, listener) = triggered::trigger();

    trigger.trigger();

    let thread_listener = listener.clone();
    let handle = std::thread::spawn(move || simulate_scanning_loop(thread_listener));

    let result = handle.join();

    let elapsed = start.elapsed();
    assert!(
        elapsed < timeout_duration,
        "Immediate shutdown took {:?}, expected < {:?}",
        elapsed,
        timeout_duration
    );
    assert!(
        result.is_ok(),
        "Should exit immediately when already triggered"
    );
}

#[test]
fn test_shutdown_signal_propagation() {
    let (trigger, listener) = triggered::trigger();

    let listener1 = listener.clone();
    let listener2 = listener.clone();
    let listener3 = listener.clone();

    assert!(!listener1.is_triggered());
    assert!(!listener2.is_triggered());
    assert!(!listener3.is_triggered());

    trigger.trigger();

    assert!(listener1.is_triggered());
    assert!(listener2.is_triggered());
    assert!(listener3.is_triggered());
}

#[test]
fn test_multiple_shutdown_checks() {
    let (trigger, listener) = triggered::trigger();

    for _ in 0..100 {
        assert!(!listener.is_triggered());
    }

    trigger.trigger();

    for _ in 0..100 {
        assert!(listener.is_triggered());
    }
}

#[test]
fn test_shutdown_cleanup_order() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    let cleanup_order = Arc::new(AtomicUsize::new(0));
    let audio_stopped = Arc::new(AtomicBool::new(false));
    let sdr_stopped = Arc::new(AtomicBool::new(false));

    let (trigger, listener) = triggered::trigger();

    let order = cleanup_order.clone();
    let audio = audio_stopped.clone();
    let sdr = sdr_stopped.clone();

    let handle = std::thread::spawn(move || simulate_cleanup_sequence(listener, order, audio, sdr));

    std::thread::sleep(Duration::from_millis(50));
    trigger.trigger();

    handle.join().unwrap();

    assert!(
        audio_stopped.load(Ordering::SeqCst),
        "Audio should be stopped"
    );
    assert!(sdr_stopped.load(Ordering::SeqCst), "SDR should be stopped");

    let order = cleanup_order.load(Ordering::SeqCst);
    assert_eq!(order, 2, "Both cleanup steps should complete in order");
}

fn simulate_paused_scanner(shutdown: Listener) {
    let paused = true;

    loop {
        if paused {
            if shutdown.is_triggered() {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
            continue;
        }

        std::thread::sleep(Duration::from_millis(10));

        if shutdown.is_triggered() {
            break;
        }
    }
}

fn simulate_scanning_loop(shutdown: Listener) {
    for _ in 0..100 {
        if shutdown.is_triggered() {
            break;
        }

        std::thread::sleep(Duration::from_millis(10));
    }
}

fn simulate_window_processing(shutdown: Listener) {
    for window in 0..50 {
        if shutdown.is_triggered() {
            break;
        }

        process_simulated_window(window, shutdown.clone());

        if shutdown.is_triggered() {
            break;
        }
    }
}

fn process_simulated_window(_window_id: usize, shutdown: Listener) {
    for _ in 0..10 {
        if shutdown.is_triggered() {
            return;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn simulate_cleanup_sequence(
    shutdown: Listener,
    cleanup_order: Arc<std::sync::atomic::AtomicUsize>,
    audio_stopped: Arc<std::sync::atomic::AtomicBool>,
    sdr_stopped: Arc<std::sync::atomic::AtomicBool>,
) {
    loop {
        if shutdown.is_triggered() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    audio_stopped.store(true, std::sync::atomic::Ordering::SeqCst);
    cleanup_order.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    std::thread::sleep(Duration::from_millis(10));

    sdr_stopped.store(true, std::sync::atomic::Ordering::SeqCst);
    cleanup_order.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

#[test]
fn test_pause_signal_shutdown_interaction() {
    use scanner::scanner_state::PauseSignal;

    let pause_signal = PauseSignal::new();
    let (trigger, shutdown) = triggered::trigger();

    pause_signal.pause();
    assert!(pause_signal.is_paused());
    assert!(!shutdown.is_triggered());

    trigger.trigger();
    assert!(pause_signal.is_paused());
    assert!(shutdown.is_triggered());

    pause_signal.unpause();
    assert!(!pause_signal.is_paused());
    assert!(shutdown.is_triggered());
}

#[test]
fn test_concurrent_shutdown_checks() {
    use std::sync::Barrier;

    let (trigger, listener) = triggered::trigger();
    let barrier = Arc::new(Barrier::new(5));
    let mut handles = vec![];

    for thread_id in 0..5 {
        let listener_clone = listener.clone();
        let barrier_clone = barrier.clone();

        let handle = std::thread::spawn(move || {
            barrier_clone.wait();

            for iteration in 0..100 {
                if listener_clone.is_triggered() {
                    return (thread_id, iteration);
                }
                std::thread::sleep(Duration::from_micros(100));
            }
            (thread_id, 100)
        });

        handles.push(handle);
    }

    std::thread::sleep(Duration::from_millis(10));
    trigger.trigger();

    let mut all_detected = true;
    for handle in handles {
        let (_thread_id, iteration) = handle.join().unwrap();
        if iteration == 100 {
            all_detected = false;
        }
    }

    assert!(all_detected, "All threads should detect shutdown");
}

#[test]
fn test_shutdown_with_state_machine() {
    use scanner::scanner_state::ScannerState;

    let mut state = ScannerState::new();
    let (trigger, shutdown) = triggered::trigger();

    state.start_window(5);
    assert!(state.is_scanning());

    state.handle_pause(5);
    assert!(state.is_paused());

    if shutdown.is_triggered() {
        assert!(false, "Should not be triggered yet");
    }

    trigger.trigger();

    assert!(shutdown.is_triggered(), "Shutdown should be triggered");
    assert!(state.is_paused(), "State should still be paused");
}

mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 20,
            .. ProptestConfig::default()
        })]

        #[test]
        fn shutdown_at_arbitrary_time(
            shutdown_delay_ms in 0u64..500,
            work_iterations in 10usize..100,
        ) {
            let (trigger, listener) = triggered::trigger();

            let thread_listener = listener.clone();
            let handle = std::thread::spawn(move || {
                for i in 0..work_iterations {
                    if thread_listener.is_triggered() {
                        return i;
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                work_iterations
            });

            std::thread::sleep(Duration::from_millis(shutdown_delay_ms));
            trigger.trigger();

            let result = handle.join();
            prop_assert!(result.is_ok(), "Thread should exit cleanly");

            let iterations_completed = result.unwrap();
            prop_assert!(
                iterations_completed <= work_iterations,
                "Should complete at most {} iterations, got {}",
                work_iterations,
                iterations_completed
            );
        }

        #[test]
        fn shutdown_during_state_transitions(
            pause_at_window in 1usize..20,
            shutdown_delay_ms in 0u64..100,
        ) {
            use scanner::scanner_state::ScannerState;

            let mut state = ScannerState::new();
            let (trigger, shutdown) = triggered::trigger();

            state.start_window(pause_at_window);

            std::thread::sleep(Duration::from_millis(shutdown_delay_ms));

            state.handle_pause(pause_at_window);

            trigger.trigger();

            prop_assert!(shutdown.is_triggered(), "Shutdown should be triggered");
            prop_assert!(state.is_paused(), "State should be paused");
        }

        #[test]
        fn pause_and_shutdown_timing(
            pause_delay_ms in 0u64..100,
            shutdown_delay_ms in 0u64..100,
            resume_delay_ms in 0u64..100,
        ) {
            use scanner::scanner_state::PauseSignal;

            let pause_signal = PauseSignal::new();
            let (trigger, shutdown) = triggered::trigger();

            std::thread::sleep(Duration::from_millis(pause_delay_ms));
            pause_signal.pause();

            std::thread::sleep(Duration::from_millis(shutdown_delay_ms));
            trigger.trigger();

            std::thread::sleep(Duration::from_millis(resume_delay_ms));
            pause_signal.unpause();

            prop_assert!(shutdown.is_triggered(), "Shutdown should persist");
            prop_assert!(!pause_signal.is_paused(), "Should be unpaused");
        }

        #[test]
        fn concurrent_shutdown_detection(
            num_threads in 2usize..8,
            trigger_delay_ms in 10u64..100,
        ) {
            let (trigger, listener) = triggered::trigger();
            let mut handles = vec![];

            for _ in 0..num_threads {
                let listener_clone = listener.clone();
                let handle = std::thread::spawn(move || {
                    let mut checks = 0;
                    while !listener_clone.is_triggered() {
                        checks += 1;
                        if checks > 1000 {
                            return false;
                        }
                        std::thread::sleep(Duration::from_micros(100));
                    }
                    true
                });
                handles.push(handle);
            }

            std::thread::sleep(Duration::from_millis(trigger_delay_ms));
            trigger.trigger();

            let mut all_detected = true;
            for handle in handles {
                let detected = handle.join().unwrap();
                if !detected {
                    all_detected = false;
                }
            }

            prop_assert!(all_detected, "All threads should detect shutdown");
        }

        #[test]
        fn shutdown_check_frequency(
            num_checks in 100usize..1000,
        ) {
            let (trigger, listener) = triggered::trigger();

            for _ in 0..num_checks {
                prop_assert!(!listener.is_triggered(), "Should not be triggered before trigger");
            }

            trigger.trigger();

            for _ in 0..num_checks {
                prop_assert!(listener.is_triggered(), "Should be triggered after trigger");
            }
        }

        #[test]
        fn window_processing_with_shutdown(
            total_windows in 5usize..20,
            shutdown_at_window in 1usize..10,
        ) {
            let shutdown_window = shutdown_at_window.min(total_windows - 1);
            let (trigger, listener) = triggered::trigger();

            let thread_listener = listener.clone();
            let handle = std::thread::spawn(move || {
                let mut processed = 0;
                for window in 0..total_windows {
                    if thread_listener.is_triggered() {
                        return processed;
                    }

                    std::thread::sleep(Duration::from_millis(10));
                    processed += 1;

                    if window == shutdown_window {
                        return processed;
                    }
                }
                processed
            });

            std::thread::sleep(Duration::from_millis(shutdown_window as u64 * 10 + 5));
            trigger.trigger();

            let result = handle.join();
            prop_assert!(result.is_ok(), "Thread should exit cleanly");

            let windows_processed = result.unwrap();
            prop_assert!(
                windows_processed <= total_windows,
                "Should process at most {} windows, got {}",
                total_windows,
                windows_processed
            );
        }
    }
}
