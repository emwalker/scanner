use crate::{soapy::SdrSource, types::Result};
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
}

impl SdrManager {
    pub fn new(driver: String, samp_rate: f64) -> Result<Self> {
        let sdr_source = Arc::new(Mutex::new(SdrSource::when_ready(driver)?));
        let (audio_sender, _) = broadcast::channel(262144); // Increased to 256K samples (~128ms at 2MHz for parallel processing)
        Ok(Self {
            sdr_source,
            samp_rate,
            audio_sender,
            graph_handle: None,
            cancel_token: None,
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

        let broadcast_sink =
            crate::broadcast::BroadcastSink::new(sdr_output_stream, self.audio_sender.clone());
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
