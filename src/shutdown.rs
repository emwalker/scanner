//! Centralized shutdown coordination using structured concurrency patterns
//!
//! This module provides the `ShutdownCoordinator` which ensures that:
//! - All spawned threads are automatically tracked
//! - Shutdown propagates to all components via a single cancellation token
//! - All threads are properly joined during cleanup
//! - It's architecturally impossible to forget shutdown checks
//!
//! ## Lifecycle State Machine
//!
//! The coordinator enforces a strict lifecycle using typestate:
//! - **Active**: Can spawn threads, can transition to ShuttingDown
//! - **ShuttingDown**: Cannot spawn threads, can transition to Terminated via `wait()`
//! - **Terminated**: All threads joined, no further operations allowed
//!
//! State transitions are enforced at runtime, preventing operations in invalid states.

use std::sync::Mutex;

use tokio_util::sync::CancellationToken;
use tracing::debug;

use crate::core::types::{Result, ScannerError};

/// State: Coordinator is active and can spawn threads
#[derive(Debug, Clone, PartialEq)]
pub struct Active;

/// State: Shutdown initiated, no more threads can be spawned
#[derive(Debug, Clone, PartialEq)]
pub struct ShuttingDown;

/// State: All threads have been joined, coordinator is terminated
#[derive(Debug, Clone, PartialEq)]
pub struct Terminated;

/// Coordinator lifecycle state
///
/// This enum wraps typestate structs, allowing compile-time type safety
/// for state transitions while maintaining runtime flexibility for
/// dynamic event handling.
#[derive(Debug, Clone, PartialEq)]
pub enum CoordinatorState {
    /// Coordinator is active and can spawn threads
    Active(Active),
    /// Shutdown initiated, no more threads can be spawned
    ShuttingDown(ShuttingDown),
    /// All threads have been joined, coordinator is terminated
    Terminated(Terminated),
}

/// Centralized shutdown coordination for multi-SDR scanner
///
/// This coordinator ensures structured concurrency by:
/// - Tracking all spawned threads automatically
/// - Providing a single cancellation token that propagates to all components
/// - Guaranteeing all threads are joined on shutdown
///
/// # Example
///
/// ```
/// use scanner::shutdown::ShutdownCoordinator;
///
/// # fn main() -> scanner::core::types::Result<()> {
/// let coordinator = ShutdownCoordinator::new();
///
/// // Spawn tracked threads
/// coordinator.spawn_sdr_thread(|cancel_token| {
///     while !cancel_token.is_cancelled() {
///         // Do work
///     }
/// })?;
///
/// // Later: shutdown everything
/// coordinator.shutdown();
/// coordinator.wait()?; // All threads guaranteed to be joined
/// //
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct ShutdownCoordinator {
    /// Current lifecycle state
    state: Mutex<CoordinatorState>,

    /// Root cancellation token - cancelling this cancels all child tokens
    token: CancellationToken,

    /// All spawned thread handles (for dedicated blocking I/O threads)
    thread_handles: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl ShutdownCoordinator {
    /// Create a new shutdown coordinator
    pub fn new() -> Self {
        Self {
            state: Mutex::new(CoordinatorState::Active(Active)),
            token: CancellationToken::new(),
            thread_handles: Mutex::new(Vec::new()),
        }
    }

    /// Get a child cancellation token
    ///
    /// When the coordinator's root token is cancelled, all child tokens
    /// are automatically cancelled as well.
    pub fn token(&self) -> CancellationToken {
        self.token.child_token()
    }

    /// Spawn a tracked SDR I/O thread
    ///
    /// The spawned thread receives a cancellation token and should check
    /// it periodically to respond to shutdown requests.
    ///
    /// # Arguments
    ///
    /// * `f` - Function to run in the thread, receives a CancellationToken
    ///
    /// # Example
    ///
    /// ```
    /// use scanner::shutdown::ShutdownCoordinator;
    ///
    /// # fn main() -> scanner::core::types::Result<()> {
    /// # let coordinator = ShutdownCoordinator::new();
    /// coordinator.spawn_sdr_thread(|cancel_token| {
    ///     while !cancel_token.is_cancelled() {
    ///         // Process SDR samples
    ///     }
    /// })?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn spawn_sdr_thread<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce(CancellationToken) + Send + 'static,
    {
        match self.state.try_lock() {
            Ok(state_guard) => {
                if !matches!(*state_guard, CoordinatorState::Active(_)) {
                    return Err(ScannerError::PoolShutdown);
                }
                drop(state_guard);
            }
            Err(std::sync::TryLockError::Poisoned(e)) => {
                return Err(ScannerError::MutexPoisoned {
                    context: format!("Failed to lock coordinator state: {}", e),
                });
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                return Err(ScannerError::PoolShutdown);
            }
        }

        let cancel_token = self.token();
        let handle = std::thread::spawn(move || {
            f(cancel_token);
        });

        match self.thread_handles.try_lock() {
            Ok(mut guard) => {
                guard.push(handle);
                Ok(())
            }
            Err(std::sync::TryLockError::Poisoned(e)) => Err(ScannerError::MutexPoisoned {
                context: format!("Failed to lock thread_handles: {}", e),
            }),
            Err(std::sync::TryLockError::WouldBlock) => Err(ScannerError::PoolShutdown),
        }
    }

    /// Initiate graceful shutdown
    ///
    /// This cancels the root token, which automatically propagates to all
    /// child tokens. All spawned threads should detect the cancellation
    /// and begin cleanup.
    ///
    /// Transitions from Active → ShuttingDown. Idempotent if already shutting down.
    pub fn shutdown(&self) {
        debug!("ShutdownCoordinator: Initiating shutdown");

        if let Ok(mut state) = self.state.lock()
            && matches!(*state, CoordinatorState::Active(_))
        {
            *state = CoordinatorState::ShuttingDown(ShuttingDown);
            debug!("ShutdownCoordinator: Transitioned to ShuttingDown state");
        }

        self.token.cancel();
    }

    /// Check if shutdown has been initiated
    pub fn is_shutdown(&self) -> bool {
        self.token.is_cancelled()
    }

    /// Check if coordinator is in Active state
    pub fn is_active(&self) -> bool {
        self.state
            .lock()
            .map(|state| matches!(*state, CoordinatorState::Active(_)))
            .unwrap_or(false)
    }

    /// Check if coordinator is in ShuttingDown state
    pub fn is_shutting_down(&self) -> bool {
        self.state
            .lock()
            .map(|state| matches!(*state, CoordinatorState::ShuttingDown(_)))
            .unwrap_or(false)
    }

    /// Check if coordinator is in Terminated state
    pub fn is_terminated(&self) -> bool {
        self.state
            .lock()
            .map(|state| matches!(*state, CoordinatorState::Terminated(_)))
            .unwrap_or(false)
    }

    /// Wait for all spawned threads to complete
    ///
    /// This should be called after `shutdown()` to ensure all threads
    /// have properly joined before the process exits.
    ///
    /// Transitions from ShuttingDown → Terminated. Returns an error if called
    /// when not in ShuttingDown state.
    ///
    /// # Returns
    ///
    /// Returns an error if any thread panicked during execution or if called
    /// in the wrong state.
    pub fn wait(&self) -> Result<()> {
        debug!("ShutdownCoordinator: Waiting for all threads to complete");

        match self.state.try_lock() {
            Ok(state_guard) => {
                if !matches!(*state_guard, CoordinatorState::ShuttingDown(_)) {
                    return Err(ScannerError::InvalidState {
                        expected: "ShuttingDown".to_string(),
                        actual: format!("{:?}", *state_guard),
                    });
                }
                drop(state_guard);
            }
            Err(std::sync::TryLockError::Poisoned(e)) => {
                return Err(ScannerError::MutexPoisoned {
                    context: format!("Failed to lock coordinator state: {}", e),
                });
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                return Err(ScannerError::PoolShutdown);
            }
        }

        let mut handles_guard =
            self.thread_handles
                .lock()
                .map_err(|e| ScannerError::MutexPoisoned {
                    context: format!("Failed to lock thread_handles: {}", e),
                })?;
        let handles: Vec<_> = handles_guard.drain(..).collect();
        drop(handles_guard);

        let total_threads = handles.len();
        debug!(
            thread_count = total_threads,
            "ShutdownCoordinator: Joining threads"
        );

        for (idx, handle) in handles.into_iter().enumerate() {
            match handle.join() {
                Ok(_) => {
                    debug!(
                        thread_idx = idx + 1,
                        total = total_threads,
                        "ShutdownCoordinator: Thread joined successfully"
                    );
                }
                Err(e) => {
                    debug!(
                        thread_idx = idx + 1,
                        total = total_threads,
                        error = ?e,
                        "ShutdownCoordinator: Thread panicked"
                    );
                    return Err(ScannerError::ThreadPanic(
                        "Thread panicked during shutdown".to_string(),
                    ));
                }
            }
        }

        if let Ok(mut state) = self.state.lock() {
            *state = CoordinatorState::Terminated(Terminated);
            debug!("ShutdownCoordinator: Transitioned to Terminated state");
        }

        debug!("ShutdownCoordinator: All threads joined successfully");
        Ok(())
    }
}

impl Default for ShutdownCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        time::Duration,
    };

    use super::*;

    #[test]
    fn test_coordinator_creation() {
        let coordinator = ShutdownCoordinator::new();
        assert!(!coordinator.is_shutdown());
    }

    #[test]
    fn test_shutdown_cancels_token() {
        let coordinator = ShutdownCoordinator::new();
        let token = coordinator.token();

        assert!(!token.is_cancelled());
        coordinator.shutdown();
        assert!(token.is_cancelled());
    }

    #[test]
    fn test_spawn_and_join_thread() {
        let coordinator = ShutdownCoordinator::new();
        let executed = Arc::new(AtomicBool::new(false));
        let executed_clone = executed.clone();

        coordinator
            .spawn_sdr_thread(move |_cancel| {
                executed_clone.store(true, Ordering::SeqCst);
            })
            .unwrap();

        std::thread::sleep(Duration::from_millis(50));
        coordinator.shutdown();
        coordinator.wait().unwrap();

        assert!(executed.load(Ordering::SeqCst));
    }

    #[test]
    fn test_thread_responds_to_cancellation() {
        let coordinator = ShutdownCoordinator::new();
        let iterations = Arc::new(AtomicBool::new(false));
        let iterations_clone = iterations.clone();

        coordinator
            .spawn_sdr_thread(move |cancel_token| {
                while !cancel_token.is_cancelled() {
                    std::thread::sleep(Duration::from_millis(10));
                }
                iterations_clone.store(true, Ordering::SeqCst);
            })
            .unwrap();

        std::thread::sleep(Duration::from_millis(50));
        coordinator.shutdown();
        coordinator.wait().unwrap();

        assert!(iterations.load(Ordering::SeqCst));
    }

    #[test]
    fn test_multiple_threads() {
        let coordinator = ShutdownCoordinator::new();
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        for _ in 0..5 {
            let count_clone = count.clone();
            coordinator
                .spawn_sdr_thread(move |cancel_token| {
                    while !cancel_token.is_cancelled() {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    count_clone.fetch_add(1, Ordering::SeqCst);
                })
                .unwrap();
        }

        std::thread::sleep(Duration::from_millis(50));
        coordinator.shutdown();
        coordinator.wait().unwrap();

        assert_eq!(count.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn test_child_tokens_cancelled_together() {
        let coordinator = ShutdownCoordinator::new();
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
    fn test_spawn_after_shutdown_returns_error() {
        let coordinator = ShutdownCoordinator::new();
        coordinator.shutdown();

        let result = coordinator.spawn_sdr_thread(|_cancel| {});

        assert!(result.is_err());
        match result {
            Err(ScannerError::PoolShutdown) => {}
            _ => panic!("Expected PoolShutdown error"),
        }
    }

    #[test]
    fn test_spawn_during_concurrent_shutdown() {
        let coordinator = Arc::new(ShutdownCoordinator::new());
        let coordinator_clone = coordinator.clone();

        let spawn_handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(10));
            coordinator_clone.spawn_sdr_thread(|_cancel| {
                std::thread::sleep(Duration::from_millis(50));
            })
        });

        coordinator.shutdown();

        let spawn_result = spawn_handle.join().unwrap();
        assert!(
            spawn_result.is_err(),
            "Spawning during shutdown should fail"
        );
    }

    #[test]
    fn test_initial_state_is_active() {
        let coordinator = ShutdownCoordinator::new();
        assert!(coordinator.is_active());
        assert!(!coordinator.is_shutting_down());
        assert!(!coordinator.is_terminated());
    }

    #[test]
    fn test_state_transitions_through_lifecycle() {
        let coordinator = ShutdownCoordinator::new();

        assert!(coordinator.is_active());

        coordinator.shutdown();
        assert!(!coordinator.is_active());
        assert!(coordinator.is_shutting_down());
        assert!(!coordinator.is_terminated());

        coordinator.wait().unwrap();
        assert!(!coordinator.is_active());
        assert!(!coordinator.is_shutting_down());
        assert!(coordinator.is_terminated());
    }

    #[test]
    fn test_spawn_only_allowed_in_active_state() {
        let coordinator = ShutdownCoordinator::new();

        let result1 = coordinator.spawn_sdr_thread(|_cancel| {});
        assert!(result1.is_ok(), "Should spawn in Active state");

        coordinator.shutdown();
        let result2 = coordinator.spawn_sdr_thread(|_cancel| {});
        assert!(result2.is_err(), "Should not spawn in ShuttingDown state");

        coordinator.wait().unwrap();
        let result3 = coordinator.spawn_sdr_thread(|_cancel| {});
        assert!(result3.is_err(), "Should not spawn in Terminated state");
    }

    #[test]
    fn test_wait_requires_shutting_down_state() {
        let coordinator = ShutdownCoordinator::new();

        let result = coordinator.wait();
        assert!(
            result.is_err(),
            "Wait should fail when not in ShuttingDown state"
        );
        assert!(matches!(result, Err(ScannerError::InvalidState { .. })));
    }

    #[test]
    fn test_shutdown_idempotent() {
        let coordinator = ShutdownCoordinator::new();

        coordinator.shutdown();
        assert!(coordinator.is_shutting_down());

        coordinator.shutdown();
        assert!(coordinator.is_shutting_down());
    }

    #[test]
    fn test_state_transitions_with_threads() {
        let coordinator = ShutdownCoordinator::new();
        let executed = Arc::new(AtomicBool::new(false));
        let executed_clone = executed.clone();

        assert!(coordinator.is_active());

        coordinator
            .spawn_sdr_thread(move |cancel_token| {
                while !cancel_token.is_cancelled() {
                    std::thread::sleep(Duration::from_millis(10));
                }
                executed_clone.store(true, Ordering::SeqCst);
            })
            .unwrap();

        coordinator.shutdown();
        assert!(coordinator.is_shutting_down());

        coordinator.wait().unwrap();
        assert!(coordinator.is_terminated());
        assert!(executed.load(Ordering::SeqCst));
    }
}
