// Example: Multi-Consumer Broadcast Channel Setup
//
// This demonstrates how to set up broadcast channels for distributing
// SDR samples to multiple concurrent consumers (peak detection, signal
// quality analysis, FM demodulation).
//
// Key patterns:
// - BroadcastHub with independent channels per consumer
// - Arc-based sharing to minimize cloning overhead
// - Backpressure handling with try_send
// - Warmup period support

use std::sync::Arc;
use crossbeam::channel::{self, Sender, Receiver, TrySendError};
use num_complex::Complex;

// ============================================================================
// Broadcast Hub
// ============================================================================

#[derive(Debug)]
pub enum BroadcastError {
    AllDisconnected,
    SomeFailed(Vec<(usize, &'static str)>),
}

pub struct BroadcastHub<T> {
    senders: Vec<Sender<Arc<T>>>,
}

impl<T> BroadcastHub<T> {
    pub fn new() -> Self {
        Self {
            senders: Vec::new(),
        }
    }

    /// Add a new subscriber, returns receiver
    pub fn add_subscriber(&mut self, capacity: usize) -> Receiver<Arc<T>> {
        let (tx, rx) = channel::bounded(capacity);
        self.senders.push(tx);
        rx
    }

    /// Broadcast data to all subscribers
    pub fn broadcast(&self, item: Arc<T>) -> Result<(), BroadcastError> {
        if self.senders.is_empty() {
            return Err(BroadcastError::AllDisconnected);
        }

        let mut failed = Vec::new();

        for (idx, sender) in self.senders.iter().enumerate() {
            match sender.try_send(Arc::clone(&item)) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    failed.push((idx, "full"));
                }
                Err(TrySendError::Disconnected(_)) => {
                    failed.push((idx, "disconnected"));
                }
            }
        }

        if !failed.is_empty() {
            Err(BroadcastError::SomeFailed(failed))
        } else {
            Ok(())
        }
    }

    /// Remove disconnected subscribers
    pub fn remove_disconnected(&mut self) {
        self.senders.retain(|tx| !tx.is_disconnected());
    }

    /// Get count of active subscribers
    pub fn subscriber_count(&self) -> usize {
        self.senders.len()
    }
}

// ============================================================================
// Warmup Consumer Wrapper
// ============================================================================

pub struct WarmupConsumer<T> {
    rx: Receiver<Arc<T>>,
    samples_to_discard: std::sync::atomic::AtomicUsize,
}

impl<T> WarmupConsumer<T> {
    pub fn new(rx: Receiver<Arc<T>>, warmup_samples: usize) -> Self {
        Self {
            rx,
            samples_to_discard: std::sync::atomic::AtomicUsize::new(warmup_samples),
        }
    }

    /// Receive data, automatically discarding warmup samples
    pub fn recv(&self) -> Option<Arc<T>> {
        let data = self.rx.recv().ok()?;

        let to_discard = self.samples_to_discard.load(std::sync::atomic::Ordering::Relaxed);
        if to_discard > 0 {
            self.samples_to_discard.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            // Discard this sample, recv next
            return self.recv();
        }

        Some(data)
    }

    pub fn try_recv(&self) -> Option<Arc<T>> {
        let data = self.rx.try_recv().ok()?;

        let to_discard = self.samples_to_discard.load(std::sync::atomic::Ordering::Relaxed);
        if to_discard > 0 {
            self.samples_to_discard.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            return self.try_recv();
        }

        Some(data)
    }
}

// ============================================================================
// Example Consumer Threads
// ============================================================================

/// Peak detector consumer (fast, small buffer)
fn peak_detector_thread(rx: WarmupConsumer<Vec<Complex<f32>>>) {
    println!("Peak detector started");

    loop {
        match rx.recv() {
            Some(samples) => {
                // Simulate peak detection
                let peak = samples.iter()
                    .map(|s| s.norm())
                    .fold(0.0f32, f32::max);

                println!("Peak detector: max = {:.2}", peak);
            }
            None => {
                println!("Peak detector: channel disconnected");
                break;
            }
        }
    }
}

/// Signal quality analyzer consumer (moderate speed, medium buffer)
fn signal_quality_thread(rx: WarmupConsumer<Vec<Complex<f32>>>) {
    println!("Signal quality analyzer started");

    loop {
        match rx.recv() {
            Some(samples) => {
                // Simulate signal quality analysis (slower than peak detection)
                std::thread::sleep(std::time::Duration::from_millis(5));

                let power = samples.iter()
                    .map(|s| s.norm_sqr())
                    .sum::<f32>() / samples.len() as f32;

                println!("Signal quality: power = {:.2}", power);
            }
            None => {
                println!("Signal quality: channel disconnected");
                break;
            }
        }
    }
}

/// FM demodulator consumer (slowest, largest buffer)
fn fm_demod_thread(rx: WarmupConsumer<Vec<Complex<f32>>>) {
    println!("FM demodulator started");

    let mut prev_sample = Complex::new(0.0, 0.0);

    loop {
        match rx.recv() {
            Some(samples) => {
                // Simulate FM demodulation (slowest consumer)
                std::thread::sleep(std::time::Duration::from_millis(10));

                // Quadrature demodulation
                let mut audio_samples = Vec::new();
                for &sample in samples.iter() {
                    let product = sample * prev_sample.conj();
                    let audio = product.im.atan2(product.re);
                    audio_samples.push(audio);
                    prev_sample = sample;
                }

                println!("FM demod: processed {} audio samples", audio_samples.len());
            }
            None => {
                println!("FM demod: channel disconnected");
                break;
            }
        }
    }
}

// ============================================================================
// SDR Producer Thread
// ============================================================================

fn sdr_producer_thread(hub: Arc<std::sync::Mutex<BroadcastHub<Vec<Complex<f32>>>>>) {
    println!("SDR producer started");

    let mut sample_count = 0;

    loop {
        // Simulate reading from SDR hardware
        std::thread::sleep(std::time::Duration::from_millis(20));

        // Generate dummy samples
        let samples: Vec<Complex<f32>> = (0..1024)
            .map(|i| {
                let phase = (sample_count + i) as f32 * 0.01;
                Complex::new(phase.cos(), phase.sin())
            })
            .collect();

        sample_count += 1024;

        // Broadcast to all consumers
        let hub = hub.lock().unwrap();
        match hub.broadcast(Arc::new(samples)) {
            Ok(()) => {}
            Err(BroadcastError::AllDisconnected) => {
                println!("SDR producer: all consumers disconnected, exiting");
                break;
            }
            Err(BroadcastError::SomeFailed(failed)) => {
                eprintln!("SDR producer: {} consumers failed", failed.len());
                for (idx, reason) in failed {
                    eprintln!("  Consumer {}: {}", idx, reason);
                }
            }
        }

        // Periodically clean up disconnected consumers
        if sample_count % 10000 == 0 {
            drop(hub);
            let mut hub_mut = hub.lock().unwrap();
            hub_mut.remove_disconnected();
            println!("Active consumers: {}", hub_mut.subscriber_count());
        }

        // Stop after some samples for demo
        if sample_count > 50000 {
            break;
        }
    }

    println!("SDR producer stopped");
}

// ============================================================================
// Example Usage
// ============================================================================

fn main() {
    // Create broadcast hub
    let hub = Arc::new(std::sync::Mutex::new(BroadcastHub::new()));

    // Add subscribers with different buffer sizes
    let peak_detector_rx;
    let signal_quality_rx;
    let fm_demod_rx;

    {
        let mut hub_guard = hub.lock().unwrap();

        // Fast consumer, small buffer (4 chunks)
        peak_detector_rx = hub_guard.add_subscriber(4);

        // Medium consumer, medium buffer (16 chunks)
        signal_quality_rx = hub_guard.add_subscriber(16);

        // Slow consumer, large buffer (64 chunks)
        fm_demod_rx = hub_guard.add_subscriber(64);
    }

    // Wrap receivers with warmup logic
    // Discard first N samples to let filters settle
    let peak_detector_rx = WarmupConsumer::new(peak_detector_rx, 10);
    let signal_quality_rx = WarmupConsumer::new(signal_quality_rx, 10);
    let fm_demod_rx = WarmupConsumer::new(fm_demod_rx, 10);

    // Spawn consumer threads
    let peak_handle = std::thread::spawn(move || peak_detector_thread(peak_detector_rx));
    let quality_handle = std::thread::spawn(move || signal_quality_thread(signal_quality_rx));
    let demod_handle = std::thread::spawn(move || fm_demod_thread(fm_demod_rx));

    // Spawn producer thread
    let hub_clone = Arc::clone(&hub);
    let producer_handle = std::thread::spawn(move || sdr_producer_thread(hub_clone));

    // Wait for all threads
    producer_handle.join().unwrap();
    peak_handle.join().unwrap();
    quality_handle.join().unwrap();
    demod_handle.join().unwrap();

    println!("All threads completed");
}

// ============================================================================
// Buffer Sizing Helper
// ============================================================================

/// Calculate buffer size based on processing characteristics
pub fn calculate_buffer_size(
    sample_rate: f32,
    chunk_size: usize,
    processing_time_ms: f32,
    safety_factor: f32,
) -> usize {
    let chunks_per_sec = sample_rate / chunk_size as f32;
    let chunks_during_processing = chunks_per_sec * (processing_time_ms / 1000.0);
    let buffer_chunks = (chunks_during_processing * safety_factor).ceil() as usize;

    buffer_chunks.max(2) // Minimum 2 for ping-pong
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_sizing() {
        // 2 MHz sample rate, 1024 sample chunks, 5ms processing, 2x safety
        let size = calculate_buffer_size(2_000_000.0, 1024, 5.0, 2.0);
        assert_eq!(size, 20); // ~20 chunks

        // Fast consumer: 2ms processing
        let fast = calculate_buffer_size(2_000_000.0, 1024, 2.0, 1.5);
        assert_eq!(fast, 6); // ~6 chunks

        // Slow consumer: 10ms processing
        let slow = calculate_buffer_size(2_000_000.0, 1024, 10.0, 3.0);
        assert_eq!(slow, 59); // ~59 chunks
    }

    #[test]
    fn test_broadcast_hub() {
        let mut hub = BroadcastHub::new();

        let rx1 = hub.add_subscriber(10);
        let rx2 = hub.add_subscriber(10);

        // Broadcast should succeed
        let data = Arc::new(vec![1, 2, 3]);
        assert!(hub.broadcast(Arc::clone(&data)).is_ok());

        // Both receivers should get the data
        assert_eq!(*rx1.recv().unwrap(), vec![1, 2, 3]);
        assert_eq!(*rx2.recv().unwrap(), vec![1, 2, 3]);

        // Drop one receiver
        drop(rx1);

        // Broadcast should still work with one receiver
        hub.remove_disconnected();
        assert!(hub.broadcast(Arc::new(vec![4, 5, 6])).is_ok());
        assert_eq!(*rx2.recv().unwrap(), vec![4, 5, 6]);
    }
}
