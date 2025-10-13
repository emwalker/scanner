//! RAII wrapper for tuners acquired from the pool

use crate::core::types::{Result, ScannerError};
use crate::hardware::pool::SubprocessHandle;
use crate::hardware::pool::state::PoolInner;
use crate::hardware::pool::types::TunerId;
use rustradio::Complex;
use rustradio::graph::GraphRunner;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tracing::error;

/// Backend implementation for tuner operations
pub enum TunerBackend {
    /// Subprocess-based IPC device access
    Subprocess { subprocess: Arc<SubprocessHandle> },
}

/// Smart pointer that auto-returns tuner on drop (RAII)
///
/// # Design: Explicit Methods vs. Deref
///
/// Unlike `object-pool` crate's `Reusable<T>` which implements `Deref`/`DerefMut`,
/// we use explicit methods (`add_source_to_graph()`, `tune()`, `set_gain()`).
///
/// **Why not Deref?**
/// - Our methods automatically pass the correct `channel_index`
/// - Deref would expose raw `Device` trait, bypassing channel logic
/// - Explicit methods make multi-channel handling clear
/// - Prevents accidental misuse (e.g., calling device.tune() with wrong channel)
///
/// # Lock Ordering
///
/// To prevent deadlocks, always acquire locks in this order:
/// 1. Device lock (`self.device`)
/// 2. Pool lock (`self.pool`)
///
/// All methods follow this ordering. The Drop implementation only locks the pool,
/// ensuring safe cleanup even if device lock is held elsewhere.
pub struct Tuner {
    /// Tuner identifier
    pub tuner_id: TunerId,

    /// Backend for device operations
    pub backend: TunerBackend,

    /// Pool inner reference for auto-return
    pub(crate) pool_inner: Arc<Mutex<PoolInner>>,

    /// Notification closure (captures status computation and callbacks)
    pub(crate) on_return: Box<dyn Fn() + Send + Sync>,

    /// Shutdown mode flag (shared with Pool)
    pub shutdown_mode: Arc<AtomicBool>,
}

impl Tuner {
    /// Get the tuner ID
    pub fn id(&self) -> &TunerId {
        &self.tuner_id
    }

    /// Get the channel index for this tuner
    pub fn channel_index(&self) -> usize {
        self.tuner_id.channel_index
    }

    /// Stop streaming on this tuner
    ///
    /// Sends StopStream message and waits for acknowledgment to ensure
    /// the worker is ready for new commands.
    ///
    /// During shutdown mode, uses fire-and-forget to avoid blocking.
    pub fn stop_stream(&self) -> Result<()> {
        match &self.backend {
            TunerBackend::Subprocess { subprocess } => subprocess.stop_stream(self.channel_index()),
        }
    }

    /// Add source to rustradio graph for this tuner
    ///
    /// This is a convenience method that automatically uses the correct channel index.
    /// Returns error if pool is in shutdown mode.
    pub fn add_source_to_graph(
        &self,
        graph: &mut rustradio::graph::Graph,
        freq: f64,
        samp_rate: f64,
        gain_db: f64,
    ) -> Result<rustradio::stream::ReadStream<Complex>> {
        if self.shutdown_mode.load(Ordering::SeqCst) {
            return Err(ScannerError::PoolShutdown);
        }

        match &self.backend {
            TunerBackend::Subprocess { subprocess } => {
                match subprocess.configure_and_start(self.channel_index(), freq, gain_db, samp_rate)
                {
                    Ok(_config) => {
                        let (source, stream) = crate::hardware::pool::SubprocessSource::new(
                            Arc::clone(&subprocess.data_receiver),
                            self.channel_index(),
                        );

                        graph.add(Box::new(source));
                        Ok(stream)
                    }
                    Err(e) => {
                        error!(
                            tuner_id = ?self.tuner_id,
                            error = ?e,
                            "Failed to configure and start subprocess streaming"
                        );
                        Err(e)
                    }
                }
            }
        }
    }
}

impl Drop for Tuner {
    /// Return tuner to pool automatically when dropped
    ///
    /// # Lock ordering
    /// Acquires pool lock only (safe - device lock already released by caller)
    ///
    /// # Shutdown safety
    /// Uses try_lock() to avoid blocking during shutdown. If the pool is locked
    /// (e.g., another thread is querying status), we skip returning the tuner
    /// since the pool is likely being destroyed anyway.
    ///
    /// # Stream cleanup
    /// Caller should call stop_stream() explicitly before dropping to ensure
    /// proper acknowledgment. Drop does not stop streams to avoid blocking.
    fn drop(&mut self) {
        let shutdown_mode = self.shutdown_mode.load(Ordering::SeqCst);

        let returned = match self.pool_inner.try_lock() {
            Ok(mut pool_inner) => {
                let returned = pool_inner.return_tuner(self.tuner_id.clone(), shutdown_mode);
                tracing::debug!(tuner_id = ?self.tuner_id, "Tuner returned to pool");
                returned
            }
            Err(_) => {
                tracing::warn!(
                    tuner_id = ?self.tuner_id,
                    "Could not return tuner to pool (locked) - likely shutting down"
                );
                false
            }
        };

        if returned {
            (self.on_return)();
        }
    }
}
