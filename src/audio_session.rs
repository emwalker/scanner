use crate::types::{Result, ScanningConfig, Signal};
use crate::window::Window;
use cpal::traits::StreamTrait;
use cpal::{BufferSize, SampleFormat, StreamConfig};
use rustradio::graph::GraphRunner;
use tracing::debug;

pub struct AudioSession {
    audio_tx: std::sync::mpsc::SyncSender<crate::mpsc::AudioPacket>,
    audio_packet_size: usize,
    _stream: cpal::Stream,
    current_graph: Option<GraphHandle>,
    current_segment: Option<Box<dyn crate::sdr::Segment>>,
}

struct GraphHandle {
    cancel_token: rustradio::graph::CancellationToken,
    thread_handle: std::thread::JoinHandle<()>,
}

impl AudioSession {
    pub fn new(config: &ScanningConfig) -> Result<Self> {
        let audio_packet_size = 4096;
        let audio_buffer_packets = 16;
        let (audio_tx, audio_rx) =
            std::sync::mpsc::sync_channel::<crate::mpsc::AudioPacket>(audio_buffer_packets);

        let (audio_device, supported_config) =
            Window::setup_audio_device(config.audio_sample_rate)?;
        let sample_format = supported_config.sample_format();
        let mut stream_config: StreamConfig = supported_config.into();
        stream_config.buffer_size = BufferSize::Fixed(config.audio_buffer_size);

        let stream = match sample_format {
            SampleFormat::F32 => {
                Window::create_audio_stream(&audio_device, &stream_config, audio_rx)?
            }
            _ => {
                return Err(crate::types::ScannerError::Custom(
                    "Unsupported audio format".to_string(),
                ));
            }
        };

        stream.play()?;
        debug!("AudioSession created: audio stream playing");

        Ok(Self {
            audio_tx,
            audio_packet_size,
            _stream: stream,
            current_graph: None,
            current_segment: None,
        })
    }

    pub fn tune_to_station(
        &mut self,
        signal: &Signal,
        segment: Box<dyn crate::sdr::Segment>,
        config: &ScanningConfig,
    ) -> Result<()> {
        self.stop_current_station();

        debug!(
            frequency_mhz = signal.frequency_hz / 1e6,
            "AudioSession: Tuning to station"
        );

        let sdr_rx = segment.audio_subscriber();

        let mut audio_graph = Window::create_audio_fm_graph(
            signal,
            sdr_rx,
            self.audio_tx.clone(),
            config,
            signal.detection_center_freq,
            self.audio_packet_size,
        )?;

        let cancel_token = audio_graph.cancel_token();
        let thread_handle = std::thread::spawn(move || {
            // Lower thread priority to reduce CPU impact
            let _ =
                thread_priority::set_current_thread_priority(thread_priority::ThreadPriority::Min);
            debug!("AudioSession: Audio graph thread started with low priority");
            if let Err(e) = audio_graph.run() {
                debug!(error = ?e, "AudioSession: Audio graph error");
            } else {
                debug!("AudioSession: Audio graph completed");
            }
            drop(audio_graph);
            debug!("AudioSession: Audio graph dropped");
        });

        self.current_graph = Some(GraphHandle {
            cancel_token,
            thread_handle,
        });
        self.current_segment = Some(segment);

        Ok(())
    }

    pub fn stop_current_station(&mut self) {
        if let Some(graph) = self.current_graph.take() {
            debug!("AudioSession: Stopping current station");
            graph.cancel_token.cancel();
            let _ = graph.thread_handle.join();
            debug!("AudioSession: Current station stopped");
        }
        if let Some(segment) = self.current_segment.take() {
            debug!("AudioSession: Dropping SDR segment");
            drop(segment);
            debug!("AudioSession: SDR segment dropped");
        }
    }
}

impl Drop for AudioSession {
    fn drop(&mut self) {
        debug!("AudioSession: Dropping");
        self.stop_current_station();
        debug!("AudioSession: Audio stream will stop");
    }
}
