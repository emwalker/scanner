use crate::core::types::{Result, ScannerError, ScanningConfig, Signal};
use crate::scanning::window::{create_audio_fm_graph, create_audio_stream, setup_audio_device};
use crate::shutdown::ShutdownCoordinator;
use cpal::traits::StreamTrait;
use cpal::{BufferSize, SampleFormat, StreamConfig};
use rustradio::graph::GraphRunner;
use std::sync::Arc;
use tracing::debug;

pub struct AudioSession {
    audio_tx: std::sync::mpsc::SyncSender<crate::mpsc::AudioPacket>,
    audio_packet_size: usize,
    _stream: cpal::Stream,
    current_graph_cancel: Option<rustradio::graph::CancellationToken>,
    current_graph_thread: Option<std::thread::JoinHandle<()>>,
    current_segment: Option<Box<dyn crate::hardware::pool::SegmentTrait>>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
}

impl AudioSession {
    pub fn new(
        config: &ScanningConfig,
        shutdown_coordinator: Arc<ShutdownCoordinator>,
    ) -> Result<Self> {
        let audio_packet_size = 4096;
        let audio_buffer_packets = 16;
        let (audio_tx, audio_rx) =
            std::sync::mpsc::sync_channel::<crate::mpsc::AudioPacket>(audio_buffer_packets);

        let (audio_device, supported_config) = setup_audio_device(config.audio_sample_rate)?;
        let sample_format = supported_config.sample_format();
        let mut stream_config: StreamConfig = supported_config.into();
        stream_config.buffer_size = BufferSize::Fixed(config.audio_buffer_size);

        let stream = match sample_format {
            SampleFormat::F32 => create_audio_stream(&audio_device, &stream_config, audio_rx)?,
            _ => {
                return Err(ScannerError::Custom("Unsupported audio format".to_string()));
            }
        };

        stream.play()?;
        debug!("AudioSession created: audio stream playing");

        Ok(Self {
            audio_tx,
            audio_packet_size,
            _stream: stream,
            current_graph_cancel: None,
            current_graph_thread: None,
            current_segment: None,
            shutdown_coordinator,
        })
    }

    pub fn tune_to_station(
        &mut self,
        signal: &Signal,
        segment: Box<dyn crate::hardware::pool::SegmentTrait>,
        config: &ScanningConfig,
    ) -> Result<()> {
        self.stop_current_station();

        debug!(
            frequency_mhz = signal.frequency_hz / 1e6,
            "AudioSession: Tuning to station"
        );

        let sdr_rx = segment.audio_subscriber();

        let mut audio_graph = create_audio_fm_graph(
            signal,
            sdr_rx,
            self.audio_tx.clone(),
            config,
            signal.detection_center_freq,
            self.audio_packet_size,
        )?;

        let cancel_token = audio_graph.cancel_token();
        let cancel_token_for_thread = cancel_token.clone();
        let shutdown_token = self.shutdown_coordinator.token();

        let handle = std::thread::spawn(move || {
            // Lower thread priority to reduce CPU impact
            let _ =
                thread_priority::set_current_thread_priority(thread_priority::ThreadPriority::Min);
            debug!("AudioSession: Audio graph thread started with low priority");

            // Bridge: If coordinator signals shutdown, cancel the audio graph
            if shutdown_token.is_cancelled() {
                debug!("AudioSession: Shutdown detected before graph start, cancelling");
                cancel_token_for_thread.cancel();
            }

            if let Err(e) = audio_graph.run() {
                debug!(error = ?e, "AudioSession: Audio graph error");
            } else {
                debug!("AudioSession: Audio graph completed");
            }
            drop(audio_graph);
            debug!("AudioSession: Audio graph dropped");
        });

        self.current_graph_cancel = Some(cancel_token);
        self.current_graph_thread = Some(handle);
        self.current_segment = Some(segment);

        Ok(())
    }

    pub fn stop_current_station(&mut self) {
        // CRITICAL: Must cancel audio graph thread AND wait for it to finish
        // BEFORE dropping SDR segment to avoid use-after-free

        // Step 1: Cancel the audio graph
        if let Some(cancel_token) = self.current_graph_cancel.take() {
            debug!("AudioSession: Stopping current station, cancelling audio graph");
            cancel_token.cancel();
            debug!("AudioSession: Audio graph cancellation requested");
        }

        // Step 2: Wait for the audio graph thread to finish
        // This ensures the thread is no longer using the segment's broadcast channel
        if let Some(handle) = self.current_graph_thread.take() {
            debug!("AudioSession: Waiting for audio graph thread to finish");
            if let Err(e) = handle.join() {
                debug!(error = ?e, "AudioSession: Audio graph thread panicked");
            } else {
                debug!("AudioSession: Audio graph thread joined successfully");
            }
        }

        // Step 3: Now it's safe to drop the SDR segment
        // The tuner will be returned to the pool via RAII
        if let Some(segment) = self.current_segment.take() {
            debug!("AudioSession: Dropping SDR segment (will return tuner to pool)");
            drop(segment);
            debug!("AudioSession: SDR segment dropped, tuner returned to pool");
        }

        debug!("AudioSession: Current station stopped");
    }
}

impl Drop for AudioSession {
    fn drop(&mut self) {
        debug!("AudioSession: Dropping");
        self.stop_current_station();
        debug!("AudioSession: Audio stream will stop");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::Backend;
    use crate::hardware::pool::{
        Pool, PoolFilter, Segment, TaskPriority, TaskRequirements, TunerActivity,
    };
    use std::time::SystemTime;

    /// Regression test for NoAvailableTuner error when switching stations
    ///
    /// Issue: When switching between stations in listening mode, the tuner
    /// from the first station wasn't being properly returned to the pool,
    /// causing NoAvailableTuner error when trying to tune to the second station.
    ///
    /// Root causes (both fixed):
    /// 1. AudioSession wasn't waiting for the audio graph thread to finish
    ///    before dropping the segment, so the tuner remained allocated.
    /// 2. MainThread was creating new Segment BEFORE calling stop_current_station(),
    ///    so tuner acquisition happened before the old tuner was released.
    #[test]
    fn test_switch_between_stations_releases_tuner() {
        let _ = tracing_subscriber::fmt::try_init();

        // Setup: Create pool with one device
        let filter = PoolFilter::new().with_driver("mock");
        let pool = Arc::new(Pool::new(filter));

        // Add a mock device to the pool
        let device_info = crate::hardware::DeviceInfo {
            id: crate::hardware::DeviceId::from_serial("mock", "test-device"),
            label: "Test SDR Device".to_string(),
        };

        let mock_backend = crate::hardware::Mock;
        let device = mock_backend.open_device(&device_info.id).unwrap();
        pool.add_device(device, "mock".to_string());

        let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());
        let config = ScanningConfig::default();

        // Create AudioSession
        let mut audio_session = AudioSession::new(&config, shutdown_coordinator.clone()).unwrap();

        // Station 1: Acquire tuner and create segment
        let requirements_1 = TaskRequirements {
            frequency_hz: 88_900_000.0,
            bandwidth_hz: config.samp_rate,
            required_sample_rate: config.samp_rate,
            priority: TaskPriority::Normal,
        };

        let tuner_1 = pool
            .acquire(&requirements_1, TunerActivity::Listening)
            .expect("Should acquire tuner for station 1");

        let segment_1 =
            Segment::from_tuner(tuner_1, 88_900_000.0, &config, shutdown_coordinator.token())
                .expect("Should create segment 1");

        let signal_1 = crate::core::types::Signal {
            frequency_hz: 88_900_000.0,
            signal_strength: 0.8,
            bandwidth_hz: 200_000.0,
            modulation: crate::core::types::ModulationType::WFM,
            audio_sample_rate: config.audio_sample_rate,
            detected_at: SystemTime::now(),
            analysis_duration_ms: 100,
            detection_center_freq: 88_900_000.0,
            audio_quality: crate::audio::quality::AudioQuality::Good,
        };

        // Tune to station 1
        audio_session
            .tune_to_station(&signal_1, Box::new(segment_1), &config)
            .expect("Should tune to station 1");

        // Verify tuner is allocated
        let status_after_station1 = pool.status();
        assert_eq!(
            status_after_station1.allocated_tuner_count, 1,
            "Tuner should be allocated for station 1"
        );
        assert_eq!(
            status_after_station1.available_tuner_count, 0,
            "No tuners should be available while station 1 is playing"
        );

        // Station 2: Switch to a different station
        // The fix: Call stop_current_station() BEFORE creating new segment
        // This releases station 1's tuner back to the pool first
        audio_session.stop_current_station();

        // Now we can successfully acquire a tuner for station 2
        let segment_2 = Segment::new(&pool, 90_800_000.0, &config, &shutdown_coordinator)
            .expect("Should be able to create segment for station 2 after stopping station 1");

        let signal_2 = crate::core::types::Signal {
            frequency_hz: 90_800_000.0,
            signal_strength: 0.7,
            bandwidth_hz: 200_000.0,
            modulation: crate::core::types::ModulationType::WFM,
            audio_sample_rate: config.audio_sample_rate,
            detected_at: SystemTime::now(),
            analysis_duration_ms: 100,
            detection_center_freq: 90_800_000.0,
            audio_quality: crate::audio::quality::AudioQuality::Good,
        };

        // Tune to station 2
        audio_session
            .tune_to_station(&signal_2, Box::new(segment_2), &config)
            .expect("Should successfully tune to station 2");

        // Verify tuner is still allocated (but for station 2 now)
        let status_after_station2 = pool.status();
        assert_eq!(
            status_after_station2.allocated_tuner_count, 1,
            "Tuner should be allocated for station 2"
        );
        assert_eq!(
            status_after_station2.available_tuner_count, 0,
            "No tuners should be available while station 2 is playing"
        );

        // Cleanup: Stop current station
        audio_session.stop_current_station();

        // Verify tuner is returned to pool
        let status_after_stop = pool.status();
        assert_eq!(
            status_after_stop.allocated_tuner_count, 0,
            "No tuners should be allocated after stopping"
        );
        assert_eq!(
            status_after_stop.available_tuner_count, 1,
            "Tuner should be back in the pool"
        );
    }
}
