use crate::soapy;
use crate::types::{Result, ScanningConfig};
use rustradio::Complex;
use rustradio::graph::{CancellationToken, Graph, GraphRunner};
use std::sync::{Arc, Mutex};
use std::thread;
use tokio::sync::broadcast;
use tracing::debug;

// Wrapper for device string that creates soapysdr::Device on-demand
#[derive(Clone)]
pub struct Device(pub String);

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device").finish()
    }
}

impl Device {
    pub fn new(device_string: String) -> Result<Self> {
        Ok(Self(device_string))
    }
}

impl From<String> for Device {
    fn from(device_string: String) -> Self {
        Self(device_string)
    }
}

impl TryFrom<&Device> for soapysdr::Device {
    type Error = crate::types::ScannerError;

    fn try_from(device: &Device) -> Result<Self> {
        soapysdr::Device::new(device.0.as_str()).map_err(crate::types::ScannerError::from)
    }
}

impl crate::sdr::Device for Device {
    fn tune(
        &self,
        config: &ScanningConfig,
        center_freq: f64,
    ) -> Result<Box<dyn crate::sdr::Segment>> {
        let manager = SoapySdrManager::new(config, center_freq, self.clone())?;
        Ok(Box::new(manager))
    }
}

/// Manages a single SDR source, running it in a background thread and
/// broadcasting samples.
pub struct SoapySdrManager {
    sdr_source: Arc<Mutex<SoapySdrSource>>,
    samp_rate: f64,
    sdr_gain: f64,
    audio_sender: broadcast::Sender<Complex>,
    graph_handle: Option<thread::JoinHandle<()>>,
    cancel_token: Option<CancellationToken>,

    // I/Q capture configuration
    capture_iq: Option<String>,
    capture_duration: f64,
    fft_size: usize,
    peak_detection_threshold: f32,
    peak_scan_duration: Option<f64>,
    device: soapy::Device,
}

impl SoapySdrManager {
    pub fn new(config: &ScanningConfig, center_freq: f64, device: Device) -> Result<Self> {
        let sdr_source = Arc::new(Mutex::new(SoapySdrSource::new(device.clone())?));
        let (audio_sender, _) = broadcast::channel(524288); // Increased to 512K samples (~256ms at 2MHz for better buffering

        let mut manager = Self {
            sdr_source,
            samp_rate: config.samp_rate,
            sdr_gain: config.sdr_gain,
            audio_sender,
            graph_handle: None,
            cancel_token: None,

            // I/Q capture configuration
            capture_iq: config.capture_iq.clone(),
            capture_duration: config.capture_duration,
            fft_size: config.fft_size,
            peak_detection_threshold: config.peak_detection_threshold,
            peak_scan_duration: config.peak_scan_duration,
            device,
        };

        // Start the SDR graph immediately with the provided center frequency
        manager.start_sdr_graph(center_freq)?;

        Ok(manager)
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

    /// Starts the SDR graph with the specified center frequency.
    fn start_sdr_graph(&mut self, freq: f64) -> Result<()> {
        debug!("Building new SDR graph at {:.1} MHz", freq / 1e6);
        let mut graph = Graph::new();
        let (sdr_source_block, sdr_output_stream) = self
            .sdr_source
            .lock()
            .unwrap()
            .create_raw_source_block(freq, self.samp_rate, self.sdr_gain)?;

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
                self.device.0.to_string(),
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

        // Use channel-based synchronization to ensure SDR graph is ready to avoid a race
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();

        self.graph_handle = Some(thread::spawn(move || {
            debug!("SDR graph thread started");

            // Send ready signal just before starting graph.run()
            // This ensures the broadcast sink is fully initialized
            let _ = ready_tx.send(());
            debug!("SDR graph ready, signaling main thread");

            if let Err(e) = graph.run() {
                debug!("SDR graph error: {}", e);
            }
            debug!("SDR graph thread exited");
        }));

        // Wait for SDR graph thread to signal it's ready before returning
        debug!("Waiting for SDR graph to initialize...");
        match ready_rx.recv_timeout(std::time::Duration::from_secs(5)) {
            Ok(_) => {
                debug!("SDR graph ready");
                Ok(())
            }
            Err(_) => Err(crate::types::ScannerError::Custom(
                "SDR graph failed to initialize within 5 seconds".to_string(),
            )),
        }
    }
}

impl crate::sdr::Segment for SoapySdrManager {
    fn audio_subscriber(&self) -> broadcast::Receiver<Complex> {
        self.audio_sender.subscribe()
    }
}

impl Drop for SoapySdrManager {
    fn drop(&mut self) {
        if let Err(e) = self.stop() {
            debug!("Error stopping SDR Manager: {}", e);
        }
    }
}

/// Our abstraction over SoapySDR device access
/// Hides device instantiation details and provides a single point for device readiness checks
struct SoapySdrSource {
    device: Device,
}

impl SoapySdrSource {
    /// Create a new SdrSource with a Device (containing string)
    pub fn new(device: Device) -> Result<Self> {
        Ok(Self { device })
    }

    /// Create a source block for streaming IQ data
    /// This encapsulates the SoapySdrSource::builder pattern
    pub fn create_raw_source_block(
        &self,
        frequency_hz: f64,
        sample_rate: f64,
        gain_db: f64,
    ) -> Result<(
        rustradio::blocks::SoapySdrSource,
        rustradio::stream::ReadStream<Complex>,
    )> {
        // Create the soapysdr::Device only when actually needed
        let device = soapysdr::Device::try_from(&self.device)?;

        // Test device readiness by trying to access basic properties
        let _driver = device.driver_key().map_err(|e| {
            crate::types::ScannerError::Custom(format!(
                "Device not ready - cannot get driver: {}",
                e
            ))
        })?;
        let _hardware = device.hardware_key().map_err(|e| {
            crate::types::ScannerError::Custom(format!(
                "Device not ready - cannot get hardware info: {}",
                e
            ))
        })?;

        debug!(
            "Device verified ready: driver={}, hardware={}",
            _driver, _hardware
        );

        // Convert dB gain to normalized value (0.0-1.0) for rustradio
        // SDRplay gain range is 0-48 dB, so normalize the provided value
        let normalized_gain = gain_db.clamp(0.0, 48.0);
        debug!(
            "Using gain: {:.1} dB (normalized: {:.3})",
            gain_db, normalized_gain
        );

        debug!(
            "Creating rustradio SoapySdrSource: frequency={:.1} MHz, sample_rate={:.1} MHz, gain={:.3}",
            frequency_hz / 1e6,
            sample_rate / 1e6,
            normalized_gain
        );

        // Sleep for 3 seconds before rustradio source block is instantiated
        std::thread::sleep(std::time::Duration::from_secs(3));

        // Disable AGC to allow manual gain control
        if device.has_gain_mode(soapysdr::Direction::Rx, 0)? {
            debug!("Disabling AGC for manual gain control");
            device.set_gain_mode(soapysdr::Direction::Rx, 0, false)?;
        }

        let (sdr_source_block, sdr_output_stream) =
            rustradio::blocks::SoapySdrSource::builder(&device, frequency_hz, sample_rate)
                .igain(normalized_gain / 48.0) // Normalize gain_db to 0.0-1.0 range
                .build()?;

        debug!("rustradio SoapySdrSource created successfully");
        Ok((sdr_source_block, sdr_output_stream))
    }
}
