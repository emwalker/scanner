//! RAII wrapper for tuners acquired from the pool

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use rustradio::{Complex, graph::GraphRunner};
use tracing::error;

use crate::{
    core::types::{Result, ScannerError},
    hardware::pool::{SubprocessHandle, types::TunerId},
};

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
/// All methods follow this ordering. The Drop implementation only locks entities,
/// ensuring safe cleanup even if other locks are held elsewhere.
pub struct Tuner {
    /// Tuner identifier
    pub tuner_id: TunerId,

    /// Backend for device operations
    pub backend: TunerBackend,

    /// Notification closure (captures status computation and callbacks)
    pub(crate) on_return: Box<dyn Fn() + Send + Sync>,

    /// Shutdown mode flag (shared with Pool)
    pub shutdown_mode: Arc<AtomicBool>,

    /// ECS tuner entities (for deallocation on drop)
    pub(crate) tuner_entities: Arc<Mutex<crate::ecs::EntityWorld<crate::ecs::TunerEntity>>>,
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
    /// # Shutdown safety
    /// Uses try_lock() to avoid blocking during shutdown. If entities are locked
    /// (e.g., another thread is querying status), we skip returning the tuner
    /// since the pool is likely being destroyed anyway.
    ///
    /// # Stream cleanup
    /// Caller should call stop_stream() explicitly before dropping to ensure
    /// proper acknowledgment. Drop does not stop streams to avoid blocking.
    fn drop(&mut self) {
        let shutdown_mode = self.shutdown_mode.load(Ordering::SeqCst);

        if shutdown_mode {
            tracing::debug!(tuner_id = ?self.tuner_id, "Tuner drop ignored (shutdown mode)");
            return;
        }

        let returned = match self.tuner_entities.try_lock() {
            Ok(mut entities) => {
                if let Some(entity) = entities.get_mut(&self.tuner_id) {
                    entity.allocation.deallocate();
                    entity.status.idle();
                    tracing::debug!(tuner_id = ?self.tuner_id, "Tuner returned to pool");
                    true
                } else {
                    tracing::warn!(tuner_id = ?self.tuner_id, "TunerEntity not found on drop");
                    false
                }
            }
            Err(_) => {
                tracing::warn!(
                    tuner_id = ?self.tuner_id,
                    "Could not return tuner to pool (entities locked) - likely shutting down"
                );
                false
            }
        };

        if returned {
            (self.on_return)();
        }
    }
}
