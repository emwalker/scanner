//! RAII wrapper for pooled tuners

use super::PoolInner;
use super::TunerId;
use crate::sdr;
use crate::types::Result;
use rustradio::Complex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

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
pub struct PooledTuner {
    /// Tuner identifier
    pub tuner_id: TunerId,

    /// Shared reference to the underlying device
    pub device: Arc<Mutex<Box<dyn sdr::DeviceTrait>>>,

    /// Pool reference for auto-return
    pub pool: Arc<Mutex<PoolInner>>,

    /// Shutdown mode flag (shared with Pool)
    pub shutdown_mode: Arc<AtomicBool>,
}

impl PooledTuner {
    /// Get the tuner ID
    pub fn id(&self) -> &TunerId {
        &self.tuner_id
    }

    /// Get the channel index for this tuner
    pub fn channel_index(&self) -> usize {
        self.tuner_id.channel_index
    }

    /// Add source to rustradio graph for this tuner
    ///
    /// This is a convenience method that automatically uses the correct channel index.
    /// Returns error if pool is in shutdown mode.
    ///
    /// # Lock ordering
    /// Acquires device lock only (safe - no pool lock needed)
    pub fn add_source_to_graph(
        &self,
        graph: &mut rustradio::graph::Graph,
        freq: f64,
        samp_rate: f64,
        gain_db: f64,
    ) -> Result<rustradio::stream::ReadStream<Complex>> {
        // Check shutdown mode first (lock-free)
        if self.shutdown_mode.load(Ordering::SeqCst) {
            return Err(crate::types::ScannerError::Custom(
                "Operation rejected - pool in shutdown mode".to_string(),
            ));
        }

        let device = self.device.lock().unwrap();
        device.add_source_to_graph(graph, freq, samp_rate, gain_db)
    }

    /// Tune this tuner to a new frequency
    ///
    /// Returns error if pool is in shutdown mode.
    ///
    /// # Lock ordering
    /// Acquires device lock only (safe - no pool lock needed)
    pub fn tune(&mut self, freq: f64) -> Result<()> {
        // Check shutdown mode first (lock-free)
        if self.shutdown_mode.load(Ordering::SeqCst) {
            return Err(crate::types::ScannerError::Custom(
                "Operation rejected - pool in shutdown mode".to_string(),
            ));
        }

        let mut device = self.device.lock().unwrap();
        device.tune(freq)
    }

    /// Set gain for this tuner
    ///
    /// Returns error if pool is in shutdown mode.
    ///
    /// # Lock ordering
    /// Acquires device lock only (safe - no pool lock needed)
    pub fn set_gain(&mut self, gain: f64) -> Result<()> {
        // Check shutdown mode first (lock-free)
        if self.shutdown_mode.load(Ordering::SeqCst) {
            return Err(crate::types::ScannerError::Custom(
                "Operation rejected - pool in shutdown mode".to_string(),
            ));
        }

        let mut device = self.device.lock().unwrap();
        device.set_gain(gain)
    }
}

impl Drop for PooledTuner {
    /// Return tuner to pool automatically when dropped
    ///
    /// # Lock ordering
    /// Acquires pool lock only (safe - device lock already released by caller)
    ///
    /// # Shutdown safety
    /// Uses try_lock() to avoid blocking during shutdown. If the pool is locked
    /// (e.g., another thread is querying status), we skip returning the tuner
    /// since the pool is likely being destroyed anyway.
    fn drop(&mut self) {
        let shutdown_mode = self.shutdown_mode.load(Ordering::SeqCst);

        match self.pool.try_lock() {
            Ok(mut pool) => {
                pool.return_tuner(self.tuner_id.clone(), shutdown_mode);
                tracing::debug!(tuner_id = ?self.tuner_id, "Tuner returned to pool");
            }
            Err(_) => {
                tracing::warn!(
                    tuner_id = ?self.tuner_id,
                    "Could not return tuner to pool (locked) - likely shutting down"
                );
            }
        }
    }
}
