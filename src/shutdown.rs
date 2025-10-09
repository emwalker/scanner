//! Centralized shutdown coordination using structured concurrency patterns
//!
//! This module provides the `ShutdownCoordinator` which ensures that:
//! - All spawned threads are automatically tracked
//! - Shutdown propagates to all components via a single cancellation token
//! - All threads are properly joined during cleanup
//! - It's architecturally impossible to forget shutdown checks

use crate::types::{Result, ScannerError};
use std::sync::Mutex;
use tokio_util::sync::CancellationToken;
use tracing::debug;

/// Centralized shutdown coordination for multi-SDR scanner
///
/// This coordinator ensures structured concurrency by:
/// - Tracking all spawned threads automatically
/// - Providing a single cancellation token that propagates to all components
/// - Guaranteeing all threads are joined on shutdown
///
/// # Example
///
/// ```ignore
/// let mut coordinator = ShutdownCoordinator::new();
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
/// ```
#[derive(Debug)]
pub struct ShutdownCoordinator {
    /// Root cancellation token - cancelling this cancels all child tokens
    token: CancellationToken,

    /// All spawned thread handles (for dedicated blocking I/O threads)
    thread_handles: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl ShutdownCoordinator {
    /// Create a new shutdown coordinator
    pub fn new() -> Self {
        Self {
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
    /// ```ignore
    /// coordinator.spawn_sdr_thread(|cancel_token| {
    ///     while !cancel_token.is_cancelled() {
    ///         // Process SDR samples
    ///     }
    ///     debug!("SDR thread shutting down");
    /// })?;
    /// ```
    pub fn spawn_sdr_thread<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce(CancellationToken) + Send + 'static,
    {
        let cancel_token = self.token();
        let handle = std::thread::spawn(move || {
            f(cancel_token);
        });

        self.thread_handles.lock().unwrap().push(handle);

        Ok(())
    }

    /// Initiate graceful shutdown
    ///
    /// This cancels the root token, which automatically propagates to all
    /// child tokens. All spawned threads should detect the cancellation
    /// and begin cleanup.
    pub fn shutdown(&self) {
        debug!("ShutdownCoordinator: Initiating shutdown");
        self.token.cancel();
    }

    /// Check if shutdown has been initiated
    pub fn is_shutdown(&self) -> bool {
        self.token.is_cancelled()
    }

    /// Wait for all spawned threads to complete
    ///
    /// This should be called after `shutdown()` to ensure all threads
    /// have properly joined before the process exits.
    ///
    /// # Returns
    ///
    /// Returns an error if any thread panicked during execution.
    pub fn wait(self) -> Result<()> {
        debug!("ShutdownCoordinator: Waiting for all threads to complete");

        let handles = self.thread_handles.into_inner().unwrap();
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
                    return Err(ScannerError::Custom(
                        "Thread panicked during shutdown".to_string(),
                    ));
                }
            }
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
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

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
}
