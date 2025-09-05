use crate::{broadcast::BroadcastSink, mpsc::ComplexMpscSink, soapy::SdrSource, types::Result};
use rustradio::{
    Complex,
    blocks::Tee,
    graph::{Graph, GraphRunner},
};
use tokio::sync::broadcast;
use tracing::debug;

/// Manages a single SDR source with a single active candidate sink
pub struct SdrManager {
    sdr_source: SdrSource,
    current_center_freq: f64,
    samp_rate: f64,
    detection_sink: Option<std::sync::mpsc::SyncSender<rustradio::Complex>>,
    audio_sender: broadcast::Sender<Complex>,
    sdr_graph: Option<Graph>,
}

impl SdrManager {
    pub fn new(driver: String, samp_rate: f64) -> Result<Self> {
        let sdr_source = SdrSource::when_ready(driver)?;
        let (audio_sender, _) = broadcast::channel(16384);
        Ok(Self {
            sdr_source,
            current_center_freq: 0.0,
            samp_rate,
            detection_sink: None,
            audio_sender,
            sdr_graph: None,
        })
    }

    pub fn set_center_frequency(&mut self, center_freq: f64) -> Result<()> {
        if (self.current_center_freq - center_freq).abs() > 1000.0 {
            // Only change if different by more than 1kHz
            self.current_center_freq = center_freq;
            self.rebuild_sdr_graph()?;
        }
        Ok(())
    }

    pub fn set_detection_sink(&mut self) -> Result<std::sync::mpsc::Receiver<rustradio::Complex>> {
        let (tx, rx) = std::sync::mpsc::sync_channel::<rustradio::Complex>(65536); // 4x larger buffer
        self.detection_sink = Some(tx);
        self.rebuild_sdr_graph()?;
        Ok(rx)
    }

    pub fn add_audio_sink(&mut self) -> broadcast::Receiver<Complex> {
        self.audio_sender.subscribe()
    }

    fn rebuild_sdr_graph(&mut self) -> Result<()> {
        if let Some(graph) = self.sdr_graph.take() {
            graph.cancel_token().cancel();
        }

        if self.detection_sink.is_some() {
            let mut new_graph = Graph::new();
            let (sdr_source_block, sdr_output_stream) = self
                .sdr_source
                .create_source_block(self.current_center_freq, self.samp_rate)?;

            new_graph.add(Box::new(sdr_source_block));

            let (tee, broadcast_stream, detection_stream) = Tee::new(sdr_output_stream);
            new_graph.add(Box::new(tee));

            let broadcast_sink = BroadcastSink::new(broadcast_stream, self.audio_sender.clone());
            new_graph.add(Box::new(broadcast_sink));

            if let Some(detection_sink) = self.detection_sink.take() {
                let complex_mpsc_sink =
                    ComplexMpscSink::new(detection_stream, detection_sink, "detection".to_string());
                new_graph.add(Box::new(complex_mpsc_sink));
            }

            self.sdr_graph = Some(new_graph);
        }
        Ok(())
    }

    pub fn start(&mut self) -> Result<()> {
        if let Some(ref mut graph) = self.sdr_graph {
            let _graph_handle = std::thread::spawn({
                let mut graph = std::mem::replace(graph, Graph::new());
                move || {
                    if let Err(e) = graph.run() {
                        debug!("SDR graph error: {}", e);
                    }
                }
            });
        }
        Ok(())
    }
}
