use crate::{
    soapy::SdrSource,
    types::{Result, ScanningConfig},
};
use rustradio::{
    Complex,
    graph::{CancellationToken as CancelToken, Graph, GraphRunner},
};
use std::sync::{Arc, Mutex};
use std::thread;
use tokio::sync::broadcast;
use tracing::debug;

/// Manages a single SDR source, running it in a background thread and
/// broadcasting samples.
pub struct SdrManager {
    sdr_source: Arc<Mutex<SdrSource>>,
    samp_rate: f64,
    audio_sender: broadcast::Sender<Complex>,
    graph_handle: Option<thread::JoinHandle<()>>,
    cancel_token: Option<CancelToken>,

    // I/Q capture configuration
    capture_iq: Option<String>,
    capture_duration: f64,
    fft_size: usize,
    peak_detection_threshold: f32,
    peak_scan_duration: Option<f64>,
    driver: String,
}

impl SdrManager {
    pub fn new(config: &ScanningConfig) -> Result<Self> {
        let sdr_source = Arc::new(Mutex::new(SdrSource::when_ready(config.driver.clone())?));
        let (audio_sender, _) = broadcast::channel(524288); // Increased to 512K samples (~256ms at 2MHz for better buffering
        Ok(Self {
            sdr_source,
            samp_rate: config.samp_rate,
            audio_sender,
            graph_handle: None,
            cancel_token: None,

            // I/Q capture configuration
            capture_iq: config.capture_iq.clone(),
            capture_duration: config.capture_duration,
            fft_size: config.fft_size,
            peak_detection_threshold: config.peak_detection_threshold,
            peak_scan_duration: config.peak_scan_duration,
            driver: config.driver.clone(),
        })
    }

    /// Stops the current SDR graph, if one is running.
    pub fn stop(&mut self) -> Result<()> {
        if let Some(token) = self.cancel_token.take() {
            debug!("Cancelling SDR graph");
            token.cancel();
        }
        if let Some(handle) = self.graph_handle.take() {
            debug!("Waiting for SDR graph thread to finish");
            let _ = handle.join();
            debug!("SDR graph thread finished");
        }
        Ok(())
    }

    /// Sets the center frequency of the SDR. This will stop the current
    /// graph and start a new one.
    pub fn set_frequency(&mut self, freq: f64) -> Result<()> {
        self.stop()?;

        debug!("Building new SDR graph at {:.1} MHz", freq / 1e6);
        let mut graph = Graph::new();
        let (sdr_source_block, sdr_output_stream) = self
            .sdr_source
            .lock()
            .unwrap()
            .create_source_block(freq, self.samp_rate)?;

        graph.add(Box::new(sdr_source_block));

        // Insert I/Q capture block if capture is enabled
        let final_stream = if let Some(ref capture_file) = self.capture_iq {
            let (iq_capture_block, iq_output_stream) = crate::iq_capture::IqCaptureBlock::new(
                sdr_output_stream,
                capture_file.clone(),
                self.samp_rate,
                freq,
                self.capture_duration,
                self.fft_size,
                self.peak_detection_threshold,
                self.peak_scan_duration,
                self.driver.clone(),
            )?;
            graph.add(Box::new(iq_capture_block));
            iq_output_stream
        } else {
            sdr_output_stream
        };

        let broadcast_sink =
            crate::broadcast::BroadcastSink::new(final_stream, self.audio_sender.clone());
        graph.add(Box::new(broadcast_sink));

        self.cancel_token = Some(graph.cancel_token());
        self.graph_handle = Some(thread::spawn(move || {
            debug!("SDR graph thread started");
            if let Err(e) = graph.run() {
                debug!("SDR graph error: {}", e);
            }
            debug!("SDR graph thread exited");
        }));
        Ok(())
    }

    /// Returns a new receiver for the audio broadcast channel.
    pub fn get_audio_subscriber(&self) -> broadcast::Receiver<Complex> {
        self.audio_sender.subscribe()
    }
}

impl Drop for SdrManager {
    fn drop(&mut self) {
        if let Err(e) = self.stop() {
            debug!("Error stopping SDR Manager: {}", e);
        }
    }
}
