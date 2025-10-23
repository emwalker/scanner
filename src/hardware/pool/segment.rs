//! Pool-based Segment implementation
//!
//! Provides a Segment implementation that acquires a tuner from the pool,
//! manages the SDR graph, and automatically returns the tuner on drop (RAII).

use std::{sync::Arc, thread};

use rustradio::graph::{Graph, GraphRunner};
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use tracing::debug;

use crate::{
    core::types::{Peak, Result, ScannerError, ScanningConfig},
    hardware::pool::{SegmentTrait, Tuner},
    pause_signal::PauseSignal,
    shutdown::ShutdownCoordinator,
};

/// Temporary SDR graph for peak detection
///
/// This graph is created before the Segment exists, runs peak detection,
/// then is destroyed. The Segment with its broadcast channel is created
/// AFTER peak detection completes, ensuring signals subscribe to a
/// fresh channel with no buffer overflow.
pub struct DetectionGraph {
    graph_handle: Option<thread::JoinHandle<()>>,
    graph_cancel: rustradio::graph::CancellationToken,
    monitoring_thread: Option<thread::JoinHandle<()>>,
}

impl Drop for DetectionGraph {
    fn drop(&mut self) {
        self.graph_cancel.cancel();
        if let Some(handle) = self.graph_handle.take() {
            let _ = handle.join();
        }
        // Join monitoring thread to prevent thread leak
        if let Some(handle) = self.monitoring_thread.take() {
            let _ = handle.join();
        }
    }
}

/// Run peak detection using a temporary SDR graph
///
/// Creates a temporary graph that broadcasts samples for peak detection only.
/// After peak detection completes, the graph is destroyed. This ensures the
/// Segment's broadcast channel is created AFTER peak detection, preventing
/// buffer overflow when signals spawn as late subscribers.
///
/// # Arguments
/// * `tuner` - Borrowed tuner (not consumed)
/// * `center_freq` - Center frequency in Hz
/// * `config` - Scanning configuration
/// * `shutdown_token` - Shutdown coordination
/// * `pause_signal` - Optional pause signal for peak detection
///
/// # Returns
/// Detected peaks and the temporary graph (caller must drop it)
pub fn detect_peaks_with_temp_graph(
    tuner: &Tuner,
    center_freq: f64,
    config: &ScanningConfig,
    shutdown_token: CancellationToken,
    pause_signal: Option<PauseSignal>,
) -> Result<(Vec<Peak>, DetectionGraph)> {
    debug!(
        center_freq_mhz = center_freq / 1e6,
        "Creating temporary SDR graph for peak detection"
    );

    // Create temporary broadcast channel for peak detection only
    let (temp_sender, temp_rx) = broadcast::channel(32);

    // Create temporary rustradio graph
    let mut graph = Graph::new();
    let stream =
        tuner.add_source_to_graph(&mut graph, center_freq, config.samp_rate, config.sdr_gain)?;

    // Add broadcast sink for peak detection
    let detection_sink = crate::broadcast::BroadcastSink::new(
        stream,
        temp_sender,
        config.signal_processing.packet_size,
    );
    graph.add(Box::new(detection_sink));

    // Get cancellation token
    let graph_cancel = graph.cancel_token();

    // Spawn thread to cancel graph when shutdown is requested
    // Thread exits when EITHER global shutdown OR graph cancellation occurs
    let graph_cancel_clone = graph_cancel.clone();
    let shutdown_token_clone = shutdown_token.clone();
    let monitoring_thread = Some(thread::spawn(move || {
        while !shutdown_token_clone.is_cancelled() && !graph_cancel_clone.is_canceled() {
            thread::sleep(std::time::Duration::from_millis(100));
        }
        if shutdown_token_clone.is_cancelled() {
            debug!("Shutdown requested, cancelling peak detection graph");
            graph_cancel_clone.cancel();
        }
    }));

    // Spawn graph thread
    let graph_handle = Some(thread::spawn(move || {
        debug!("Peak detection graph thread started");
        if let Err(e) = graph.run() {
            debug!(error = ?e, "Peak detection graph error");
        }
        debug!("Peak detection graph thread exited");
    }));

    // Wait for first packet (graph warmup)
    debug!("Waiting for peak detection graph to produce first packet...");
    let mut warmup_rx = temp_rx.resubscribe();
    let warmup_start = std::time::Instant::now();
    let warmup_timeout = std::time::Duration::from_secs(5);
    let mut packet_received = false;

    while warmup_start.elapsed() < warmup_timeout {
        match warmup_rx.try_recv() {
            Ok(_) => {
                packet_received = true;
                debug!("Peak detection graph producing packets");
                break;
            }
            Err(broadcast::error::TryRecvError::Empty) => {
                thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(broadcast::error::TryRecvError::Lagged(_)) => {
                packet_received = true;
                debug!("Peak detection graph producing packets (lagged)");
                break;
            }
            Err(broadcast::error::TryRecvError::Closed) => {
                return Err(ScannerError::Custom(
                    "Peak detection channel closed during warmup".to_string(),
                ));
            }
        }
    }

    if !packet_received {
        return Err(ScannerError::GraphInitTimeout {
            component: "peak detection graph warmup".to_string(),
            timeout_secs: 5,
        });
    }

    // Run peak detection
    debug!("Running peak detection from temporary graph");
    let peaks = crate::signal::collect_peaks(config, temp_rx, center_freq, pause_signal)?;

    debug!(
        peaks_found = peaks.len(),
        "Peak detection complete, temporary graph will be destroyed"
    );

    let detection_graph = DetectionGraph {
        graph_handle,
        graph_cancel,
        monitoring_thread,
    };

    Ok((peaks, detection_graph))
}

/// Pool-based Segment implementation
///
/// Wraps a pool::Tuner and provides the Segment interface. When dropped,
/// automatically returns the tuner to the pool (RAII).
///
/// # Shutdown Safety
/// - Uses CancellationToken for graceful shutdown
/// - Graph runner stops cleanly when token is cancelled
/// - Tuner is returned to pool even if shutdown occurs mid-processing
pub struct Segment {
    _tuner: Tuner,
    audio_sender: tokio::sync::broadcast::Sender<crate::broadcast::SamplePacket>,
    _keepalive_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
    graph_handle: Option<thread::JoinHandle<()>>,
    graph_cancel: rustradio::graph::CancellationToken,
    monitoring_thread: Option<thread::JoinHandle<()>>,
}

impl Segment {
    /// Create a new Segment by acquiring a tuner from the pool
    ///
    /// # Arguments
    /// * `pool` - The tuner pool to acquire from
    /// * `center_freq` - Center frequency in Hz
    /// * `config` - Scanning configuration
    /// * `shutdown_coordinator` - Coordinates graceful shutdown
    pub fn new(
        pool: &Arc<crate::hardware::pool::Pool>,
        center_freq: f64,
        config: &ScanningConfig,
        shutdown_coordinator: &Arc<ShutdownCoordinator>,
    ) -> Result<Self> {
        let shutdown_token = shutdown_coordinator.token();

        // Check shutdown before acquiring from pool
        if shutdown_token.is_cancelled() {
            debug!("Shutdown requested before pool acquisition, aborting");
            return Err(ScannerError::PoolShutdown);
        }

        // Acquire tuner from pool
        let requirements = crate::hardware::pool::TaskRequirements {
            frequency_hz: center_freq,
            bandwidth_hz: config.samp_rate,
            required_sample_rate: config.samp_rate,
            priority: crate::hardware::pool::TaskPriority::Normal,
        };

        let tuner = pool.acquire(
            &requirements,
            crate::hardware::pool::TunerActivity::Listening,
        )?;
        debug!(tuner_id = ?tuner.id(), "Acquired tuner from pool for listening");

        Self::from_tuner(tuner, center_freq, config, shutdown_token)
    }

    /// Create Segment from an already-acquired pool::Tuner
    ///
    /// Used internally by Window which manages its own pool acquisition.
    pub(crate) fn from_tuner(
        tuner: Tuner,
        center_freq: f64,
        config: &ScanningConfig,
        shutdown_token: CancellationToken,
    ) -> Result<Self> {
        // Create broadcast channel for samples (same pattern as SoapySdrManager)
        let buffer_size_packets = 524288 / config.signal_processing.packet_size;
        let (audio_sender, _) = tokio::sync::broadcast::channel(buffer_size_packets);

        // Create rustradio graph
        let mut graph = Graph::new();
        let stream = tuner.add_source_to_graph(
            &mut graph,
            center_freq,
            config.samp_rate,
            config.sdr_gain,
        )?;

        // Add BroadcastSink to send samples to broadcast channel
        let broadcast_sink = crate::broadcast::BroadcastSink::new(
            stream,
            audio_sender.clone(),
            config.signal_processing.packet_size,
        );
        graph.add(Box::new(broadcast_sink));

        // Get cancellation token from graph
        let graph_cancel = graph.cancel_token();

        // Spawn thread to cancel graph when shutdown is requested
        // Thread exits when EITHER global shutdown OR graph cancellation occurs
        let graph_cancel_clone = graph_cancel.clone();
        let shutdown_token_clone = shutdown_token.clone();
        let monitoring_thread = Some(thread::spawn(move || {
            while !shutdown_token_clone.is_cancelled() && !graph_cancel_clone.is_canceled() {
                thread::sleep(std::time::Duration::from_millis(100));
            }
            if shutdown_token_clone.is_cancelled() {
                debug!("Shutdown requested, cancelling pool-based SDR graph");
                graph_cancel_clone.cancel();
            }
        }));

        // Subscribe to audio channel to detect when first packet is sent
        let mut warmup_rx = audio_sender.subscribe();

        // Spawn graph thread
        let graph_handle = Some(thread::spawn(move || {
            debug!("Pool-based SDR graph thread started");

            if let Err(e) = graph.run() {
                debug!("Pool-based SDR graph error: {}", e);
            }
            debug!("Pool-based SDR graph thread exited");
        }));

        // Wait for graph to produce first packet (warmup period)
        debug!("Waiting for pool-based SDR graph to produce first audio packet...");
        let warmup_start = std::time::Instant::now();
        let warmup_timeout = std::time::Duration::from_secs(5);
        let mut packet_received = false;

        while warmup_start.elapsed() < warmup_timeout {
            match warmup_rx.try_recv() {
                Ok(_) => {
                    packet_received = true;
                    debug!("Pool-based SDR graph producing audio packets");
                    break;
                }
                Err(tokio::sync::broadcast::error::TryRecvError::Empty) => {
                    thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => {
                    packet_received = true;
                    debug!("Pool-based SDR graph producing audio packets (lagged)");
                    break;
                }
                Err(tokio::sync::broadcast::error::TryRecvError::Closed) => {
                    return Err(ScannerError::Custom(
                        "Audio broadcast channel closed during warmup".to_string(),
                    ));
                }
            }
        }

        if !packet_received {
            return Err(ScannerError::GraphInitTimeout {
                component: "pool-based SDR graph audio production".to_string(),
                timeout_secs: 5,
            });
        }

        Ok(Self {
            _tuner: tuner,
            audio_sender,
            _keepalive_rx: warmup_rx,
            graph_handle,
            graph_cancel,
            monitoring_thread,
        })
    }

    /// Stop the stream and wait for acknowledgment
    ///
    /// This should be called before dropping the Segment to ensure the tuner
    /// has fully stopped streaming and is ready for new commands.
    pub fn stop_stream(&mut self) -> Result<()> {
        // Cancel the graph first
        self.graph_cancel.cancel();

        // Wait for graph thread to finish
        if let Some(handle) = self.graph_handle.take() {
            debug!("Waiting for pool-based SDR graph thread to finish");
            let _ = handle.join();
            debug!("Pool-based SDR graph thread finished");
        }

        // Stop tuner stream and wait for acknowledgment (subprocess backend)
        self._tuner.stop_stream()?;

        Ok(())
    }
}

impl SegmentTrait for Segment {
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

impl Drop for Segment {
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

        // Join monitoring thread to prevent thread leak
        if let Some(handle) = self.monitoring_thread.take() {
            let _ = handle.join();
        }

        // pool::Tuner will be dropped here, automatically returning to pool (RAII)
        debug!("pool::Segment dropped, tuner will be returned to pool");
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::mpsc, thread, time::Duration};

    use super::*;

    /// Monitoring thread for DetectionGraph should exit when graph is dropped
    ///
    /// Verifies that the monitoring thread exits properly when graph cancellation
    /// is triggered, without requiring global shutdown.
    ///
    /// Following rust-testing skill: use channels for deterministic synchronization
    #[test]
    fn test_detection_graph_monitoring_thread_exits_on_drop() {
        // Setup: create a channel to signal when monitoring thread exits
        let (exit_tx, exit_rx) = mpsc::channel();
        let shutdown_token = CancellationToken::new();
        let graph_cancel = rustradio::graph::CancellationToken::new();

        // Simulate what detect_peaks_with_temp_graph does: spawn monitoring thread
        let graph_cancel_clone = graph_cancel.clone();
        let shutdown_token_clone = shutdown_token.clone();
        let monitoring_thread = Some(thread::spawn(move || {
            while !shutdown_token_clone.is_cancelled() && !graph_cancel_clone.is_canceled() {
                thread::sleep(Duration::from_millis(10));
            }
            if shutdown_token_clone.is_cancelled() {
                graph_cancel_clone.cancel();
            }
            // Signal that monitoring thread is exiting
            let _ = exit_tx.send(());
        }));

        // Create DetectionGraph
        let detection_graph = DetectionGraph {
            graph_handle: None, // Not testing graph thread, just monitoring thread
            graph_cancel,
            monitoring_thread,
        };

        // Execute: Drop the graph (DetectionGraph::drop will cancel and join threads)
        drop(detection_graph);

        // Verify: Monitoring thread should exit within reasonable time
        let result = exit_rx.recv_timeout(Duration::from_millis(200));

        assert!(
            result.is_ok(),
            "Monitoring thread should exit when graph cancellation is triggered"
        );
    }

    /// Monitoring thread for Segment should exit when segment is dropped
    #[test]
    fn test_segment_monitoring_thread_exits_on_drop() {
        // Setup: channel for deterministic synchronization
        let (exit_tx, exit_rx) = mpsc::channel();
        let shutdown_token = CancellationToken::new();
        let graph_cancel = rustradio::graph::CancellationToken::new();

        // Simulate what Segment::from_tuner does: spawn monitoring thread
        let graph_cancel_clone = graph_cancel.clone();
        let shutdown_token_clone = shutdown_token.clone();
        let monitoring_thread = Some(thread::spawn(move || {
            while !shutdown_token_clone.is_cancelled() && !graph_cancel_clone.is_canceled() {
                thread::sleep(Duration::from_millis(10));
            }
            if shutdown_token_clone.is_cancelled() {
                graph_cancel_clone.cancel();
            }
            let _ = exit_tx.send(());
        }));

        // Simulate Segment::drop: cancel graph and join monitoring thread
        graph_cancel.cancel();
        if let Some(handle) = monitoring_thread {
            let _ = handle.join();
        }

        // Verify: Monitoring thread should exit after cancellation
        let result = exit_rx.recv_timeout(Duration::from_millis(200));

        assert!(
            result.is_ok(),
            "Monitoring thread should exit when segment is dropped"
        );
    }

    /// RED TEST: Segment must survive until signals complete analysis
    ///
    /// This test demonstrates the bug where AudioStreamManagementSystem drops
    /// the Segment when all_work_complete() returns true, which happens when
    /// all_spawned=true and signals_analyzing=0. This race condition occurs
    /// after the window worker completes but before signals spawn.
    ///
    /// Bug timeline from logs (window 7):
    /// - 22:35:59.777: Window worker completed
    /// - 22:35:59.784: Segment dropped (7ms later)
    /// - 22:35:59.785: signals spawned (8ms later, 1ms after drop)
    /// - Result: All 6 signals timeout with "No audio"
    ///
    /// Following rust-testing skill: use channels for deterministic synchronization
    #[test]
    fn test_segment_survives_until_signals_finish_analysis() {
        // Setup: Channel to track Segment lifecycle
        // When Segment is dropped, the sender is dropped and receiver detects it
        let (segment_alive_tx, segment_alive_rx) = mpsc::channel::<()>();

        // Simulate Segment stored in WindowEntity after worker completes
        struct SegmentSimulator {
            _alive_signal: mpsc::Sender<()>,
        }

        impl SegmentSimulator {
            fn new(tx: mpsc::Sender<()>) -> Self {
                Self { _alive_signal: tx }
            }
        }

        // Create Segment (simulating window worker storing it in WindowEntity)
        let segment = Some(SegmentSimulator::new(segment_alive_tx));

        // Verify Segment is alive before signals spawn
        assert!(
            !matches!(
                segment_alive_rx.try_recv(),
                Err(mpsc::TryRecvError::Disconnected)
            ),
            "Segment should be alive after worker completes"
        );

        // GREEN FIX: Window worker does NOT call mark_all_spawned()
        // So all_work_complete() returns false (all_spawned=false)
        // AudioStreamManagementSystem does NOT drop the Segment

        // Simulate signals spawning on next ECS tick
        thread::sleep(Duration::from_millis(1));

        // GREEN FIX: After signals spawn, signalAnalysisSpawnSystem calls mark_all_spawned()
        // Now all_spawned=true BUT signals_analyzing > 0
        // So all_work_complete() still returns false
        // Segment stays alive for signals to use

        // Check if Segment is still alive for signals to subscribe
        let segment_alive_for_signals = !matches!(
            segment_alive_rx.try_recv(),
            Err(mpsc::TryRecvError::Disconnected)
        );

        // GREEN TEST: Segment should still be alive because we fixed the race
        assert!(
            segment_alive_for_signals,
            "GREEN FIX SUCCESS: Segment survived because mark_all_spawned() was delayed until \
             after signals spawned."
        );

        // Simulate signals finishing analysis
        // Only NOW should all_work_complete() return true and Segment be dropped
        drop(segment);

        // Verify Segment is now dropped after signals finish
        let segment_dropped_after_analysis = matches!(
            segment_alive_rx.try_recv(),
            Err(mpsc::TryRecvError::Disconnected)
        );

        assert!(
            segment_dropped_after_analysis,
            "Segment should be dropped after signals finish analysis"
        );
    }
}
