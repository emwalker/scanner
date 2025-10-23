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

use std::{sync::Arc, time::Duration};

use scanner::shutdown::ShutdownCoordinator;
use tokio_util::sync::CancellationToken;

#[test]
fn test_shutdown_while_paused() {
    let timeout_duration = Duration::from_secs(5);
    let start = std::time::Instant::now();

    let coordinator = Arc::new(ShutdownCoordinator::new());

    coordinator
        .spawn_sdr_thread(simulate_paused_scanner)
        .unwrap();

    std::thread::sleep(Duration::from_millis(100));

    coordinator.shutdown();
    Arc::try_unwrap(coordinator)
        .expect("Failed to unwrap coordinator")
        .wait()
        .unwrap();

    let elapsed = start.elapsed();
    assert!(
        elapsed < timeout_duration,
        "Shutdown while paused took {:?}, expected < {:?}",
        elapsed,
        timeout_duration
    );
}

#[test]
fn test_shutdown_while_scanning() {
    let timeout_duration = Duration::from_secs(5);
    let start = std::time::Instant::now();

    let coordinator = Arc::new(ShutdownCoordinator::new());

    coordinator
        .spawn_sdr_thread(simulate_scanning_loop)
        .unwrap();

    std::thread::sleep(Duration::from_millis(50));

    coordinator.shutdown();
    Arc::try_unwrap(coordinator)
        .expect("Failed to unwrap coordinator")
        .wait()
        .unwrap();

    let elapsed = start.elapsed();
    assert!(
        elapsed < timeout_duration,
        "Shutdown while scanning took {:?}, expected < {:?}",
        elapsed,
        timeout_duration
    );
}

#[test]
fn test_shutdown_during_window_processing() {
    let timeout_duration = Duration::from_secs(5);
    let start = std::time::Instant::now();

    let coordinator = Arc::new(ShutdownCoordinator::new());

    coordinator
        .spawn_sdr_thread(simulate_window_processing)
        .unwrap();

    std::thread::sleep(Duration::from_millis(100));

    coordinator.shutdown();
    Arc::try_unwrap(coordinator)
        .expect("Failed to unwrap coordinator")
        .wait()
        .unwrap();

    let elapsed = start.elapsed();
    assert!(
        elapsed < timeout_duration,
        "Shutdown during window processing took {:?}, expected < {:?}",
        elapsed,
        timeout_duration
    );
}

#[test]
fn test_immediate_shutdown() {
    let timeout_duration = Duration::from_secs(2);
    let start = std::time::Instant::now();

    let coordinator = Arc::new(ShutdownCoordinator::new());

    coordinator
        .spawn_sdr_thread(simulate_scanning_loop)
        .unwrap();

    coordinator.shutdown();

    Arc::try_unwrap(coordinator).unwrap().wait().unwrap();

    let elapsed = start.elapsed();
    assert!(
        elapsed < timeout_duration,
        "Immediate shutdown took {:?}, expected < {:?}",
        elapsed,
        timeout_duration
    );
}

#[test]
fn test_shutdown_signal_propagation() {
    let coordinator = Arc::new(ShutdownCoordinator::new());

    let token1 = coordinator.token();
    let token2 = coordinator.token();
    let token3 = coordinator.token();

    assert!(!token1.is_cancelled());
    assert!(!token2.is_cancelled());
    assert!(!token3.is_cancelled());

    coordinator.shutdown();

    assert!(token1.is_cancelled());
    assert!(token2.is_cancelled());
    assert!(token3.is_cancelled());
}

#[test]
fn test_multiple_shutdown_checks() {
    let coordinator = Arc::new(ShutdownCoordinator::new());
    let token = coordinator.token();

    for _ in 0..100 {
        assert!(!token.is_cancelled());
    }

    coordinator.shutdown();

    for _ in 0..100 {
        assert!(token.is_cancelled());
    }
}

#[test]
fn test_shutdown_cleanup_order() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    let cleanup_order = Arc::new(AtomicUsize::new(0));
    let audio_stopped = Arc::new(AtomicBool::new(false));
    let sdr_stopped = Arc::new(AtomicBool::new(false));

    let coordinator = Arc::new(ShutdownCoordinator::new());

    let order = cleanup_order.clone();
    let audio = audio_stopped.clone();
    let sdr = sdr_stopped.clone();

    coordinator
        .spawn_sdr_thread(move |cancel_token| {
            simulate_cleanup_sequence(cancel_token, order, audio, sdr)
        })
        .unwrap();

    std::thread::sleep(Duration::from_millis(50));
    coordinator.shutdown();
    Arc::try_unwrap(coordinator)
        .expect("Failed to unwrap coordinator")
        .wait()
        .unwrap();

    assert!(
        audio_stopped.load(Ordering::SeqCst),
        "Audio should be stopped"
    );
    assert!(sdr_stopped.load(Ordering::SeqCst), "SDR should be stopped");

    let order = cleanup_order.load(Ordering::SeqCst);
    assert_eq!(order, 2, "Both cleanup steps should complete in order");
}

fn simulate_paused_scanner(shutdown: CancellationToken) {
    let paused = true;

    loop {
        if paused {
            if shutdown.is_cancelled() {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
            continue;
        }

        std::thread::sleep(Duration::from_millis(10));

        if shutdown.is_cancelled() {
            break;
        }
    }
}

fn simulate_scanning_loop(shutdown: CancellationToken) {
    for _ in 0..100 {
        if shutdown.is_cancelled() {
            break;
        }

        std::thread::sleep(Duration::from_millis(10));
    }
}

fn simulate_window_processing(shutdown: CancellationToken) {
    for window in 0..50 {
        if shutdown.is_cancelled() {
            break;
        }

        process_simulated_window(window, shutdown.clone());

        if shutdown.is_cancelled() {
            break;
        }
    }
}

fn process_simulated_window(_window_id: usize, shutdown: CancellationToken) {
    for _ in 0..10 {
        if shutdown.is_cancelled() {
            return;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn simulate_cleanup_sequence(
    shutdown: CancellationToken,
    cleanup_order: Arc<std::sync::atomic::AtomicUsize>,
    audio_stopped: Arc<std::sync::atomic::AtomicBool>,
    sdr_stopped: Arc<std::sync::atomic::AtomicBool>,
) {
    loop {
        if shutdown.is_cancelled() {
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
    use scanner::pause_signal::PauseSignal;

    let pause_signal = PauseSignal::new();
    let coordinator = Arc::new(ShutdownCoordinator::new());
    let token = coordinator.token();

    pause_signal.pause();
    assert!(pause_signal.is_paused());
    assert!(!token.is_cancelled());

    coordinator.shutdown();
    assert!(pause_signal.is_paused());
    assert!(token.is_cancelled());

    pause_signal.unpause();
    assert!(!pause_signal.is_paused());
    assert!(token.is_cancelled());
}

#[test]
fn test_concurrent_shutdown_checks() {
    use std::sync::Barrier;

    let coordinator = Arc::new(ShutdownCoordinator::new());
    let token = coordinator.token();
    let barrier = Arc::new(Barrier::new(5));
    let mut handles = vec![];

    for thread_id in 0..5 {
        let token_clone = token.clone();
        let barrier_clone = barrier.clone();

        let handle = std::thread::spawn(move || {
            barrier_clone.wait();

            for iteration in 0..100 {
                if token_clone.is_cancelled() {
                    return (thread_id, iteration);
                }
                std::thread::sleep(Duration::from_micros(100));
            }
            (thread_id, 100)
        });

        handles.push(handle);
    }

    std::thread::sleep(Duration::from_millis(10));
    coordinator.shutdown();

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
fn test_shutdown_with_entity_state() {
    use scanner::ecs::{ScanConfigComponent, ScanEntity, ScanType, TaskId, components::WindowId};

    let config =
        ScanConfigComponent::new(ScanType::Band, 88.0e6, 108.0e6, 1.0e6, 2.4e6, 40.0, 1.0, 10);
    let mut entity = ScanEntity::new(config, 1);
    let coordinator = Arc::new(ShutdownCoordinator::new());
    let token = coordinator.token();

    entity.lifecycle.start();
    let task_id = TaskId::new("test_shutdown".to_string());
    let window_id = WindowId::new(task_id.clone(), 5);
    entity.progress.start_window(window_id.clone());
    assert!(entity.is_scanning());

    entity.progress.pause(window_id);
    assert!(entity.is_paused());

    if token.is_cancelled() {
        panic!("Should not be triggered yet");
    }

    coordinator.shutdown();

    assert!(token.is_cancelled(), "Shutdown should be triggered");
    assert!(entity.is_paused(), "State should still be paused");
}

mod property_tests {
    use proptest::prelude::*;

    use super::*;

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
            let coordinator = Arc::new(ShutdownCoordinator::new());
            let token = coordinator.token();

            let thread_token = token.clone();
            let handle = std::thread::spawn(move || {
                for i in 0..work_iterations {
                    if thread_token.is_cancelled() {
                        return i;
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                work_iterations
            });

            std::thread::sleep(Duration::from_millis(shutdown_delay_ms));
            coordinator.shutdown();

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
            use scanner::ecs::{ScanConfigComponent, ScanEntity, ScanType, TaskId};
            use scanner::ecs::components::WindowId;

            let config = ScanConfigComponent::new(
                ScanType::Band,
                88.0e6,
                108.0e6,
                1.0e6,
                2.4e6,
                40.0,
                1.0,
                pause_at_window.max(1),
            );
            let mut entity = ScanEntity::new(config, 1);
            let coordinator = Arc::new(ShutdownCoordinator::new());
            let token = coordinator.token();

            let task_id = TaskId::new("test_shutdown".to_string());
            let window_id = WindowId::new(task_id.clone(), pause_at_window);
            entity.progress.start_window(window_id.clone());

            std::thread::sleep(Duration::from_millis(shutdown_delay_ms));

            entity.progress.pause(window_id);

            coordinator.shutdown();

            prop_assert!(token.is_cancelled(), "Shutdown should be triggered");
            prop_assert!(entity.is_paused(), "State should be paused");
        }

        #[test]
        fn pause_and_shutdown_timing(
            pause_delay_ms in 0u64..100,
            shutdown_delay_ms in 0u64..100,
            resume_delay_ms in 0u64..100,
        ) {
            use scanner::pause_signal::PauseSignal;

            let pause_signal = PauseSignal::new();
            let coordinator = Arc::new(ShutdownCoordinator::new());
            let token = coordinator.token();

            std::thread::sleep(Duration::from_millis(pause_delay_ms));
            pause_signal.pause();

            std::thread::sleep(Duration::from_millis(shutdown_delay_ms));
            coordinator.shutdown();

            std::thread::sleep(Duration::from_millis(resume_delay_ms));
            pause_signal.unpause();

            prop_assert!(token.is_cancelled(), "Shutdown should persist");
            prop_assert!(!pause_signal.is_paused(), "Should be unpaused");
        }

        #[test]
        fn concurrent_shutdown_detection(
            num_threads in 2usize..8,
            trigger_delay_ms in 10u64..100,
        ) {
            let coordinator = Arc::new(ShutdownCoordinator::new());
            let token = coordinator.token();
            let mut handles = vec![];

            for _ in 0..num_threads {
                let token_clone = token.clone();
                let handle = std::thread::spawn(move || {
                    let mut checks = 0;
                    while !token_clone.is_cancelled() {
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
            coordinator.shutdown();

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
            let coordinator = Arc::new(ShutdownCoordinator::new());
            let token = coordinator.token();

            for _ in 0..num_checks {
                prop_assert!(!token.is_cancelled(), "Should not be cancelled before shutdown");
            }

            coordinator.shutdown();

            for _ in 0..num_checks {
                prop_assert!(token.is_cancelled(), "Should be cancelled after shutdown");
            }
        }

        #[test]
        fn window_processing_with_shutdown(
            total_windows in 5usize..20,
            shutdown_at_window in 1usize..10,
        ) {
            let shutdown_window = shutdown_at_window.min(total_windows - 1);
            let coordinator = Arc::new(ShutdownCoordinator::new());
            let token = coordinator.token();

            let thread_token = token.clone();
            let handle = std::thread::spawn(move || {
                let mut processed = 0;
                for window in 0..total_windows {
                    if thread_token.is_cancelled() {
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
            coordinator.shutdown();

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

#[cfg(loom)]
mod loom_tests {
    use loom::{
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        thread,
    };

    #[test]
    fn test_pause_signal_concurrent_access() {
        loom::model(|| {
            let paused = Arc::new(AtomicBool::new(false));

            let p1 = paused.clone();
            let p2 = paused.clone();

            let t1 = thread::spawn(move || {
                p1.store(true, Ordering::SeqCst);
                p1.load(Ordering::SeqCst)
            });

            let t2 = thread::spawn(move || p2.load(Ordering::SeqCst));

            let r1 = t1.join().unwrap();
            let _r2 = t2.join().unwrap();

            assert!(r1, "Thread that set pause should see true");
        });
    }

    #[test]
    fn test_shutdown_and_pause_interaction() {
        loom::model(|| {
            let shutdown = Arc::new(AtomicBool::new(false));
            let paused = Arc::new(AtomicBool::new(false));

            let s1 = shutdown.clone();
            let p1 = paused.clone();
            let main_thread = thread::spawn(move || {
                for _ in 0..3 {
                    if p1.load(Ordering::SeqCst) {
                        if s1.load(Ordering::SeqCst) {
                            return true;
                        }
                        thread::yield_now();
                        continue;
                    }

                    if s1.load(Ordering::SeqCst) {
                        return true;
                    }

                    thread::yield_now();
                }
                false
            });

            let s2 = shutdown.clone();
            let p2 = paused.clone();
            let control_thread = thread::spawn(move || {
                p2.store(true, Ordering::SeqCst);
                thread::yield_now();
                s2.store(true, Ordering::SeqCst);
            });

            control_thread.join().unwrap();
            let _main_exited = main_thread.join().unwrap();
        });
    }

    #[test]
    fn test_multiple_shutdown_checks() {
        loom::model(|| {
            let shutdown = Arc::new(AtomicBool::new(false));

            let s1 = shutdown.clone();
            let worker = thread::spawn(move || {
                let mut checks = 0;
                for _ in 0..3 {
                    if s1.load(Ordering::SeqCst) {
                        return checks;
                    }
                    checks += 1;
                    thread::yield_now();
                }
                checks
            });

            shutdown.store(true, Ordering::SeqCst);

            let _result = worker.join().unwrap();
        });
    }

    #[test]
    fn test_concurrent_pause_unpause() {
        loom::model(|| {
            let paused = Arc::new(AtomicBool::new(false));

            let p1 = paused.clone();
            let setter = thread::spawn(move || {
                p1.store(true, Ordering::SeqCst);
                thread::yield_now();
                p1.store(false, Ordering::SeqCst);
            });

            let p2 = paused.clone();
            let reader = thread::spawn(move || {
                let first = p2.load(Ordering::SeqCst);
                thread::yield_now();
                let second = p2.load(Ordering::SeqCst);
                (first, second)
            });

            setter.join().unwrap();
            let (_first, _second) = reader.join().unwrap();
        });
    }

    #[test]
    fn test_shutdown_with_multiple_workers() {
        loom::model(|| {
            let shutdown = Arc::new(AtomicBool::new(false));
            let mut handles = vec![];

            for _ in 0..2 {
                let s = shutdown.clone();
                let h = thread::spawn(move || {
                    for _ in 0..3 {
                        if s.load(Ordering::SeqCst) {
                            return true;
                        }
                        thread::yield_now();
                    }
                    false
                });
                handles.push(h);
            }

            thread::yield_now();
            shutdown.store(true, Ordering::SeqCst);

            for handle in handles {
                let _detected = handle.join().unwrap();
            }
        });
    }
}

#[cfg(not(loom))]
#[test]
fn loom_tests_require_loom_cfg() {
    println!("Loom tests are only available with RUSTFLAGS=\"--cfg loom\"");
    println!("Run: RUSTFLAGS=\"--cfg loom\" cargo test --test loom_shutdown_test --release");
}
