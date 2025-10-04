//! Loom concurrency tests for shutdown logic
//!
//! These tests use Loom to deterministically explore all possible thread interleavings
//! to find race conditions and other concurrency bugs.
//!
//! To run these tests:
//! ```
//! RUSTFLAGS="--cfg loom" cargo test --test loom_shutdown_test --release
//! ```
//!
//! Note: These tests are only compiled when the `loom` cfg is set.

#[cfg(loom)]
mod loom_tests {
    use loom::sync::Arc;
    use loom::sync::atomic::{AtomicBool, Ordering};
    use loom::thread;

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
