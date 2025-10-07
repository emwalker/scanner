//! Pool-based Segment implementation
//!
//! Provides a Segment implementation that acquires a tuner from the pool,
//! manages the SDR graph, and automatically returns the tuner on drop (RAII).

use crate::pool::PooledTuner;
use crate::sdr::Segment;
use crate::shutdown::ShutdownCoordinator;
use crate::types::{Result, ScanningConfig};
use rustradio::graph::{Graph, GraphRunner};
use std::sync::Arc;
use std::thread;
use tokio_util::sync::CancellationToken;
use tracing::debug;

/// Pool-based Segment implementation
///
/// Wraps a PooledTuner and provides the Segment interface. When dropped,
/// automatically returns the tuner to the pool (RAII).
///
/// # Shutdown Safety
/// - Uses CancellationToken for graceful shutdown
/// - Graph runner stops cleanly when token is cancelled
/// - Tuner is returned to pool even if shutdown occurs mid-processing
pub struct PoolSegment {
    _pooled_tuner: PooledTuner,
    audio_sender: tokio::sync::broadcast::Sender<crate::broadcast::SamplePacket>,
    graph_handle: Option<thread::JoinHandle<()>>,
    graph_cancel: rustradio::graph::CancellationToken,
}

impl PoolSegment {
    /// Create a new PoolSegment by acquiring a tuner from the pool
    ///
    /// # Arguments
    /// * `pool` - The tuner pool to acquire from
    /// * `center_freq` - Center frequency in Hz
    /// * `config` - Scanning configuration
    /// * `shutdown_coordinator` - Coordinates graceful shutdown
    pub fn new(
        pool: &Arc<crate::pool::Pool>,
        center_freq: f64,
        config: &ScanningConfig,
        shutdown_coordinator: &Arc<ShutdownCoordinator>,
    ) -> Result<Self> {
        let shutdown_token = shutdown_coordinator.token();

        // Check shutdown before acquiring from pool
        if shutdown_token.is_cancelled() {
            debug!("Shutdown requested before pool acquisition, aborting");
            return Err(crate::types::ScannerError::Custom(
                "Shutdown in progress".to_string(),
            ));
        }

        // Acquire tuner from pool
        let requirements = crate::pool::TaskRequirements {
            frequency_hz: center_freq,
            bandwidth_hz: config.samp_rate,
            required_sample_rate: config.samp_rate,
            priority: crate::pool::TaskPriority::Normal,
        };

        let pooled_tuner = pool.acquire(&requirements, crate::pool::TunerActivity::Listening)?;
        debug!(tuner_id = ?pooled_tuner.id(), "Acquired tuner from pool for listening");

        Self::from_pooled_tuner(pooled_tuner, center_freq, config, shutdown_token)
    }

    /// Create PoolSegment from an already-acquired PooledTuner
    ///
    /// Used internally by Window which manages its own pool acquisition.
    pub(crate) fn from_pooled_tuner(
        pooled_tuner: PooledTuner,
        center_freq: f64,
        config: &ScanningConfig,
        shutdown_token: CancellationToken,
    ) -> Result<Self> {
        // Create broadcast channel for samples (same pattern as SoapySdrManager)
        let buffer_size_packets = 524288 / config.packet_size;
        let (audio_sender, _) = tokio::sync::broadcast::channel(buffer_size_packets);

        // Create rustradio graph
        let mut graph = Graph::new();
        let stream = pooled_tuner.add_source_to_graph(
            &mut graph,
            center_freq,
            config.samp_rate,
            config.sdr_gain,
        )?;

        // Add BroadcastSink to send samples to broadcast channel
        let broadcast_sink =
            crate::broadcast::BroadcastSink::new(stream, audio_sender.clone(), config.packet_size);
        graph.add(Box::new(broadcast_sink));

        // Get cancellation token from graph
        let graph_cancel = graph.cancel_token();

        // Spawn thread to cancel graph when shutdown is requested
        let graph_cancel_clone = graph_cancel.clone();
        let shutdown_token_clone = shutdown_token.clone();
        thread::spawn(move || {
            while !shutdown_token_clone.is_cancelled() {
                thread::sleep(std::time::Duration::from_millis(100));
            }
            debug!("Shutdown requested, cancelling pool-based SDR graph");
            graph_cancel_clone.cancel();
        });

        // Use channel-based synchronization to ensure graph is ready (same pattern as SoapySdrManager)
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();

        // Spawn graph thread
        let graph_handle = Some(thread::spawn(move || {
            debug!("Pool-based SDR graph thread started");

            // Signal ready before starting graph
            let _ = ready_tx.send(());
            debug!("Pool-based SDR graph ready, signaling main thread");

            if let Err(e) = graph.run() {
                debug!("Pool-based SDR graph error: {}", e);
            }
            debug!("Pool-based SDR graph thread exited");
        }));

        // Wait for graph to be ready
        debug!("Waiting for pool-based SDR graph to initialize...");
        match ready_rx.recv_timeout(std::time::Duration::from_secs(5)) {
            Ok(_) => {
                debug!("Pool-based SDR graph ready");
            }
            Err(_) => {
                return Err(crate::types::ScannerError::Custom(
                    "Pool-based SDR graph failed to initialize within 5 seconds".to_string(),
                ));
            }
        }

        Ok(Self {
            _pooled_tuner: pooled_tuner,
            audio_sender,
            graph_handle,
            graph_cancel,
        })
    }
}

impl Segment for PoolSegment {
    fn audio_subscriber(&self) -> tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket> {
        let receiver = self.audio_sender.subscribe();
        debug!(
            receiver_len = receiver.len(),
            sender_receiver_count = self.audio_sender.receiver_count(),
            broadcast_capacity = self.audio_sender.len(),
            "Pool-based segment: New audio subscriber created"
        );
        receiver
    }
}

impl Drop for PoolSegment {
    fn drop(&mut self) {
        debug!(
            receiver_count = self.audio_sender.receiver_count(),
            "Pool-based segment dropping: Stopping SDR graph"
        );

        // Cancel the graph
        self.graph_cancel.cancel();

        // Wait for graph thread to finish
        if let Some(handle) = self.graph_handle.take() {
            debug!("Waiting for pool-based SDR graph thread to finish");
            let _ = handle.join();
            debug!("Pool-based SDR graph thread finished");
        }

        // PooledTuner will be dropped here, automatically returning to pool (RAII)
        debug!("PoolSegment dropped, tuner will be returned to pool");
    }
}
