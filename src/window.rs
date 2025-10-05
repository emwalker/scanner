use crate::sdr::Segment;
use crate::shutdown::ShutdownCoordinator;
use crate::terminal::{ProgressEvent, ProgressEventType, ProgressReporter};
use crate::types::{Result, ScanningConfig};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, SampleFormat, StreamConfig};
use rustradio::graph::GraphRunner;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::debug;

/// Metadata about the window that the TUI needs
#[derive(Debug, Clone, Copy)]
pub struct WindowMetadata {
    pub center_frequency_hz: f64,
    pub window_id: usize,
}

/// Configuration parameters for creating a Window
pub struct WindowConfig {
    pub center_freq: f64,
    pub window_num: usize,
    pub total_windows: usize,
    pub device: crate::soapy::Device,
    pub config: ScanningConfig,
    pub progress_reporter: Arc<dyn ProgressReporter>,
    pub shutdown_coordinator: Arc<ShutdownCoordinator>,
    pub pause_signal: Option<crate::scanner_state::PauseSignal>,
}

/// Represents a frequency window for band scanning with complete lifecycle management
pub struct Window {
    center_freq: f64,
    window_num: usize,
    total_windows: usize,
    station_mode: bool, // True if this is a specific station frequency, not band scanning
    device: crate::soapy::Device,
    config: ScanningConfig,
    progress_reporter: Arc<dyn ProgressReporter>,
    shutdown_token: CancellationToken,
    metadata: WindowMetadata,
    pause_signal: Option<crate::scanner_state::PauseSignal>,
}

impl Window {
    pub fn new(window_config: WindowConfig) -> Self {
        Self {
            center_freq: window_config.center_freq,
            window_num: window_config.window_num,
            total_windows: window_config.total_windows,
            station_mode: false,
            device: window_config.device,
            config: window_config.config,
            progress_reporter: window_config.progress_reporter,
            shutdown_token: window_config.shutdown_coordinator.token(),
            metadata: WindowMetadata {
                center_frequency_hz: window_config.center_freq,
                window_id: window_config.window_num,
            },
            pause_signal: window_config.pause_signal,
        }
    }

    pub fn for_station(
        center_freq: f64,
        window_num: usize,
        total_windows: usize,
        device: crate::soapy::Device,
        config: ScanningConfig,
        progress_reporter: Arc<dyn ProgressReporter>,
        shutdown_coordinator: Arc<ShutdownCoordinator>,
    ) -> Self {
        Self {
            center_freq,
            window_num,
            total_windows,
            station_mode: true,
            device,
            config,
            progress_reporter,
            shutdown_token: shutdown_coordinator.token(),
            metadata: WindowMetadata {
                center_frequency_hz: center_freq,
                window_id: window_num,
            },
            pause_signal: None,
        }
    }

    fn peaks(&self, device: &dyn Segment) -> Result<Vec<crate::types::Peak>> {
        if self.station_mode {
            // Station mode: Create a single peak at the exact station frequency
            debug!(
                "Station mode: Creating direct peak for {:.1} MHz",
                self.center_freq / 1e6
            );
            Ok(vec![crate::types::Peak {
                frequency_hz: self.center_freq,
                magnitude: 1.0, // Assume strong signal for station mode
            }])
        } else {
            // Band scanning mode: Do peak detection as usual
            let sdr_rx_for_peaks = device.audio_subscriber();
            crate::fm::collect_peaks(&self.config, sdr_rx_for_peaks, self.center_freq)
        }
    }

    fn debug_peaks(&self, peaks: &[crate::types::Peak]) {
        if self.config.debug_pipeline {
            debug!(
                message = "Band scanning window analysis",
                window_number = self.window_num,
                window_center_mhz = self.center_freq / 1e6,
                peaks_found = peaks.len()
            );

            for (peak_idx, peak) in peaks.iter().enumerate() {
                debug!(
                    message = "Peak detected",
                    window_number = self.window_num,
                    peak_index = peak_idx,
                    frequency_mhz = peak.frequency_hz / 1e6,
                    magnitude = peak.magnitude
                );
            }
        }
    }

    fn candidates_from_peaks(&self, peaks: &[crate::types::Peak]) -> Vec<crate::types::Candidate> {
        let mut candidates = Vec::new();

        if self.station_mode {
            // Station mode: Create candidate directly for the specific station frequency
            debug!(
                "Station mode: Creating direct candidate for {:.1} MHz",
                self.center_freq / 1e6
            );
            candidates.push(crate::types::Candidate::Fm(crate::fm::Candidate {
                frequency_hz: self.center_freq,
                signal_strength: "Strong".to_string(), // Assume strong signal for requested station
                peak_count: 1,
                max_magnitude: 1.0,
                avg_magnitude: 1.0,
            }));
            return candidates;
        }

        for candidate in crate::fm::find_candidates(peaks, &self.config, self.center_freq) {
            let candidate_freq = candidate.frequency_hz();

            // Check if this frequency (rounded to nearest 100 kHz) has already been processed
            let rounded_freq = (candidate_freq / 100000.0).round() * 100000.0;
            let frequency_khz = (rounded_freq / 1000.0) as u64;

            let already_processed = {
                let processed = crate::fm::PROCESSED_FREQUENCIES.lock().unwrap();
                processed.contains(&frequency_khz)
            };

            if already_processed {
                debug!(
                    candidate_frequency_mhz = candidate_freq / 1e6,
                    "Skipping candidate creation for already processed frequency"
                );
                continue;
            }

            if self.config.debug_pipeline {
                let frequency_offset = candidate_freq - self.center_freq;
                debug!(
                    message = "Candidate created",
                    candidate_frequency_mhz = candidate_freq / 1e6,
                    window_center_mhz = self.center_freq / 1e6,
                    frequency_offset_khz = frequency_offset / 1e3,
                    signal_strength = match &candidate {
                        crate::types::Candidate::Fm(fm_candidate) => &fm_candidate.signal_strength,
                    }
                );
            }
            candidates.push(candidate);
        }

        candidates
    }

    fn process_candidates(
        &self,
        candidates: Vec<crate::types::Candidate>,
        segment: &dyn Segment,
    ) -> Result<Vec<crate::types::Signal>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let candidate_count = candidates.len();
        let mut candidate_threads = Vec::new();
        let (signal_tx, signal_rx) = std::sync::mpsc::sync_channel::<crate::types::Signal>(100);

        for candidate in candidates.into_iter() {
            if self.config.print_candidates {
                tracing::info!(
                    "candidate found at {:.1} MHz",
                    candidate.frequency_hz() / 1e6
                );
                continue;
            }

            // Report audio analysis started
            let freq = match &candidate {
                crate::types::Candidate::Fm(fm_candidate) => fm_candidate.frequency_hz,
            };
            let candidate_id = format!("{:.1}-{}", freq / 1e6, self.window_num);
            self.progress_reporter.report(ProgressEvent {
                event_type: ProgressEventType::AudioAnalysisStarted,
                frequency_hz: freq,
                metadata: self.metadata,
                candidate_id: Some(candidate_id),
                audio_quality: None,
                signal_strength: None,
                timestamp: std::time::Instant::now(),
            });

            let sdr_rx = segment.audio_subscriber();
            let signal_tx_clone = signal_tx.clone();
            let config_clone = self.config.clone();
            let center_freq = self.center_freq;
            let device_clone = self.device.clone();
            let progress_reporter_clone = self.progress_reporter.clone();
            let window_num = self.window_num;
            let pause_signal_clone = self.pause_signal.clone();

            let handle = thread::spawn(move || -> Result<()> {
                // Early exit if pause requested before we even start
                if let Some(ref signal) = pause_signal_clone
                    && signal.is_paused()
                {
                    debug!("Candidate thread exiting early due to pause signal");
                    return Ok(());
                }

                let context = crate::pipeline::AnalysisContext {
                    config: &config_clone,
                    center_freq,
                    device: &device_clone,
                    progress_reporter: progress_reporter_clone,
                    metadata: crate::window::WindowMetadata {
                        center_frequency_hz: center_freq,
                        window_id: window_num,
                    },
                };
                candidate.analyze(sdr_rx, signal_tx_clone, &context)
            });
            candidate_threads.push(handle);
        }

        // Drop the sender so we can detect when all candidates are done
        drop(signal_tx);

        let window_timeout = Duration::from_secs(60);
        let threads_completed =
            self.wait_for_threads_with_timeout(candidate_threads, window_timeout);

        debug!(
            "Window {} at {:.1} MHz: {}/{} candidates completed processing",
            self.window_num,
            self.center_freq / 1e6,
            threads_completed,
            candidate_count
        );

        // Collect all signals from this window
        let mut signals = Vec::new();
        while let Ok(signal) = signal_rx.try_recv() {
            signals.push(signal);
        }

        debug!(
            "Window {} collected {} signals",
            self.window_num,
            signals.len()
        );

        Ok(signals)
    }

    pub(crate) fn setup_audio_device(
        audio_sample_rate: u32,
    ) -> Result<(cpal::Device, cpal::SupportedStreamConfig)> {
        let host = cpal::default_host();
        let audio_device = host
            .default_output_device()
            .expect("no output device available");

        let supported_configs_range = audio_device
            .supported_output_configs()
            .expect("error while querying configs");

        let supported_config = supported_configs_range
            .filter(|d| d.sample_format() == SampleFormat::F32)
            .find(|d| {
                d.min_sample_rate().0 <= audio_sample_rate
                    && d.max_sample_rate().0 >= audio_sample_rate
            })
            .expect("no supported config found")
            .with_sample_rate(cpal::SampleRate(audio_sample_rate));

        Ok((audio_device, supported_config))
    }

    pub(crate) fn create_audio_stream(
        device: &cpal::Device,
        stream_config: &StreamConfig,
        audio_rx: std::sync::mpsc::Receiver<crate::mpsc::AudioPacket>,
    ) -> Result<cpal::Stream> {
        let err_fn = |err| debug!("Audio error: {}", err);

        let underrun_counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let sample_counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let underrun_counter_clone = underrun_counter.clone();
        let sample_counter_clone = sample_counter.clone();

        let mut leftover: Option<(crate::mpsc::AudioPacket, usize)> = None;

        let stream = device.build_output_stream(
            stream_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut filled = 0;
                let mut underrun_occurred = false;

                // First, drain any leftover from previous packet
                if let Some((packet, offset)) = &leftover {
                    let remaining = &packet.as_slice()[*offset..];
                    let to_copy = remaining.len().min(data.len());
                    for i in 0..to_copy {
                        data[filled + i] = remaining[i].clamp(-1.0, 1.0);
                    }
                    filled += to_copy;

                    if to_copy >= remaining.len() {
                        leftover = None;
                    } else {
                        leftover = Some((packet.clone(), offset + to_copy));
                    }
                }

                // Then receive packets and fill buffer
                while filled < data.len() {
                    match audio_rx.try_recv() {
                        Ok(packet) => {
                            let samples = packet.as_slice();
                            let to_copy = samples.len().min(data.len() - filled);
                            for i in 0..to_copy {
                                data[filled + i] = samples[i].clamp(-1.0, 1.0);
                            }
                            filled += to_copy;

                            if to_copy < samples.len() {
                                leftover = Some((packet, to_copy));
                                break;
                            }
                        }
                        Err(_) => {
                            underrun_occurred = true;
                            for sample in &mut data[filled..] {
                                *sample = 0.0;
                            }
                            break;
                        }
                    }
                }

                sample_counter_clone.fetch_add(filled, std::sync::atomic::Ordering::Relaxed);

                if underrun_occurred {
                    let underrun_count = underrun_counter_clone
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                        + 1;
                    debug!(
                        underrun_count = underrun_count,
                        filled_samples = filled,
                        requested_samples = data.len(),
                        missing_samples = data.len() - filled,
                        "AUDIO UNDERRUN: Not enough packets from audio graph"
                    );
                }
            },
            err_fn,
            None,
        )?;

        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(5));
            let underruns = underrun_counter.load(std::sync::atomic::Ordering::Relaxed);
            let samples = sample_counter.load(std::sync::atomic::Ordering::Relaxed);
            debug!(
                total_underruns = underruns,
                total_samples_output = samples,
                "Audio stream statistics after 5 seconds"
            );
        });

        Ok(stream)
    }

    pub fn process_signal_for_audio(
        signal: &crate::types::Signal,
        sdr_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
        audio_tx: std::sync::mpsc::SyncSender<crate::mpsc::AudioPacket>,
        config: &ScanningConfig,
        shutdown_token: &CancellationToken,
        pause_signal: Option<&crate::scanner_state::PauseSignal>,
        audio_packet_size: usize,
    ) -> Result<()> {
        debug!(
            "Creating audio processing pipeline for {:.1} MHz",
            signal.frequency_hz / 1e6
        );

        let mut audio_graph = Window::create_audio_fm_graph(
            signal,
            sdr_rx,
            audio_tx,
            config,
            signal.detection_center_freq,
            audio_packet_size,
        )?;

        let duration = std::time::Duration::from_secs(config.duration);
        debug!("Playing audio for {:?}", duration);
        debug!("Starting audio graph thread...");

        let cancel_token = audio_graph.cancel_token();
        let graph_handle = std::thread::spawn(move || {
            // Lower thread priority to reduce CPU impact
            let _ =
                thread_priority::set_current_thread_priority(thread_priority::ThreadPriority::Min);
            debug!("Audio graph thread started with low priority, running graph...");
            if let Err(e) = audio_graph.run() {
                debug!("Audio graph error: {}", e);
            } else {
                debug!("Audio graph completed successfully");
            }
        });

        // Sleep with periodic cancellation checks instead of blocking sleep
        let check_interval = std::time::Duration::from_millis(100);
        let mut remaining = duration;

        while !remaining.is_zero() {
            let sleep_duration = std::cmp::min(remaining, check_interval);
            std::thread::sleep(sleep_duration);
            remaining = remaining.saturating_sub(sleep_duration);

            // Check if shutdown requested
            if shutdown_token.is_cancelled() {
                debug!("Shutdown requested during audio processing, stopping early");
                break;
            }

            // Check if pause requested
            if let Some(pause) = pause_signal
                && pause.is_paused()
            {
                debug!("Pause requested during audio processing, stopping early");
                break;
            }
        }

        debug!("Cancelling audio graph...");
        cancel_token.cancel();
        debug!("Waiting for audio graph thread to finish...");
        let _ = graph_handle.join();
        debug!("Audio graph thread finished");

        debug!(
            "Finished playing audio for {:.1} MHz [{}]",
            signal.frequency_hz / 1e6,
            signal.audio_quality.to_human_string()
        );
        Ok(())
    }

    fn setup_audio_graph_source(
        sdr_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
        graph: &mut rustradio::graph::Graph,
    ) -> rustradio::stream::ReadStream<rustradio::Complex> {
        let receiver_len = sdr_rx.len();
        debug!(receiver_len, "setup_audio_graph_source called");

        let clean_rx = if receiver_len > 1000 {
            debug!(
                discarded_samples = receiver_len,
                "Resubscribing to discard buffered IQ samples"
            );
            sdr_rx.resubscribe()
        } else {
            sdr_rx
        };

        let (source_block, stream) = crate::broadcast::BroadcastSource::new(clean_rx);
        graph.add(Box::new(source_block));
        stream
    }

    fn create_frequency_xlating_filter(
        prev: rustradio::stream::ReadStream<rustradio::Complex>,
        graph: &mut rustradio::graph::Graph,
        frequency_offset: f64,
        config: &ScanningConfig,
    ) -> Result<(rustradio::stream::ReadStream<rustradio::Complex>, u32)> {
        // Use shared pipeline builder for frequency xlating filter
        crate::fm::pipeline_builder::FmPipelineBuilder::create_frequency_xlating_filter(
            prev,
            graph,
            frequency_offset,
            config,
            crate::fm::filter_config::FilterPurpose::Audio,
        )
        .map_err(Into::into)
    }

    fn create_fm_demodulation_chain(
        prev: rustradio::stream::ReadStream<rustradio::Complex>,
        graph: &mut rustradio::graph::Graph,
        quad_rate: f32,
        signal: &crate::types::Signal,
    ) -> (rustradio::stream::ReadStream<rustradio::Float>, f32) {
        use rustradio::{blockchain, blocks::QuadratureDemod};

        // Calculate adaptive FM gain based on signal characteristics
        let base_gain = (quad_rate / (2.0 * 75_000.0)) * 0.8;
        let (fm_gain, quality_adjustment) = Self::calculate_adaptive_fm_gain(base_gain, signal);

        debug!(
            "FM demodulator gain: base={:.3}, adaptive={:.3}, signal_strength={:.6}",
            base_gain, fm_gain, signal.signal_strength
        );

        let prev = blockchain![graph, prev, QuadratureDemod::new(prev, fm_gain)];

        let (deemphasis_block, deemphasis_stream) =
            crate::fm::deemph::Deemphasis::new(prev, quad_rate, 75.0);
        graph.add(Box::new(deemphasis_block));
        (deemphasis_stream, quality_adjustment)
    }

    fn calculate_adaptive_fm_gain(base_gain: f32, signal: &crate::types::Signal) -> (f32, f32) {
        // Adaptive gain calculation based on signal strength (RMS value)
        let gain_adjustment = if signal.signal_strength < 0.001 {
            // Very weak signal - significant boost
            5.0
        } else if signal.signal_strength < 0.01 {
            // Weak signal - moderate boost
            3.0
        } else if signal.signal_strength < 0.1 {
            // Low signal - slight boost
            1.5
        } else {
            // Normal/strong signal - no adjustment
            1.0
        };

        // Additional adjustment based on audio quality
        let quality_adjustment = match signal.audio_quality {
            crate::audio_quality::AudioQuality::Good => 1.0,
            crate::audio_quality::AudioQuality::Moderate => 1.2,
            crate::audio_quality::AudioQuality::Poor => 1.5,
            crate::audio_quality::AudioQuality::NoAudio => 1.0, // Similar to Poor but signal present
            crate::audio_quality::AudioQuality::Static => 0.5,  // Reduce gain for static
            crate::audio_quality::AudioQuality::Unknown => 1.0,
        };

        let adaptive_gain = base_gain * gain_adjustment * quality_adjustment;
        let clamped_gain = adaptive_gain.clamp(0.05, 10.0);

        debug!(
            signal_strength = signal.signal_strength,
            audio_quality = ?signal.audio_quality,
            base_gain = base_gain,
            gain_adjustment = gain_adjustment,
            quality_adjustment = quality_adjustment,
            adaptive_gain = adaptive_gain,
            clamped_gain = clamped_gain,
            "Calculated adaptive FM gain"
        );

        (clamped_gain, quality_adjustment)
    }

    fn create_audio_decimation_chain(
        prev: rustradio::stream::ReadStream<rustradio::Float>,
        graph: &mut rustradio::graph::Graph,
        quad_rate: f32,
        config: &ScanningConfig,
    ) -> Result<rustradio::stream::ReadStream<rustradio::Float>> {
        // Use shared pipeline builder for audio decimation chain
        crate::fm::pipeline_builder::FmPipelineBuilder::create_audio_decimation_chain(
            prev, graph, quad_rate, config, "Audio",
        )
        .map_err(Into::into)
    }

    pub fn create_audio_fm_graph(
        signal: &crate::types::Signal,
        sdr_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
        audio_tx: std::sync::mpsc::SyncSender<crate::mpsc::AudioPacket>,
        config: &ScanningConfig,
        center_freq: f64,
        audio_packet_size: usize,
    ) -> Result<rustradio::graph::Graph> {
        let mut graph = rustradio::graph::Graph::new();
        let station_name = format!("{:.1}FM_Audio", signal.frequency_hz / 1e6);

        let frequency_offset = signal.frequency_hz - center_freq;
        debug!(
            "Audio graph: signal {:.1} MHz, center {:.1} MHz, offset {:.1} kHz",
            signal.frequency_hz / 1e6,
            center_freq / 1e6,
            frequency_offset / 1e3
        );

        let prev = Self::setup_audio_graph_source(sdr_rx, &mut graph);
        let (prev, decimation) =
            Self::create_frequency_xlating_filter(prev, &mut graph, frequency_offset, config)?;

        let decimated_samp_rate = config.samp_rate / decimation as f64;
        let quad_rate = decimated_samp_rate as f32;

        let (prev, quality_adjustment) =
            Self::create_fm_demodulation_chain(prev, &mut graph, quad_rate, signal);
        let prev = Self::create_audio_decimation_chain(prev, &mut graph, quad_rate, config)?;

        let (diagnostic_block, prev) =
            crate::broadcast::AudioDiagnostic::new(prev, quality_adjustment);
        graph.add(Box::new(diagnostic_block));

        graph.add(Box::new(crate::mpsc::MpscSink::new(
            prev,
            audio_tx,
            station_name,
            audio_packet_size,
        )));
        Ok(graph)
    }

    fn play_signals(
        &self,
        signals: Vec<crate::types::Signal>,
        segment: &dyn Segment,
    ) -> Result<()> {
        if signals.is_empty() {
            return Ok(());
        }

        // Sort signals by frequency (lowest first)
        let mut sorted_signals = signals;
        sorted_signals.sort_by(|a, b| a.frequency_hz.partial_cmp(&b.frequency_hz).unwrap());

        debug!(
            "Window {} playing {} signals in frequency order",
            self.window_num,
            sorted_signals.len()
        );

        // Create audio infrastructure for this window
        let audio_packet_size = 4096;
        let audio_buffer_packets = 16;
        let (audio_tx, audio_rx) =
            std::sync::mpsc::sync_channel::<crate::mpsc::AudioPacket>(audio_buffer_packets);

        // Setup audio device and stream
        let (audio_device, supported_config) =
            Window::setup_audio_device(self.config.audio_sample_rate)?;
        let sample_format = supported_config.sample_format();
        let mut stream_config: StreamConfig = supported_config.into();
        stream_config.buffer_size = BufferSize::Fixed(self.config.audio_buffer_size);

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
        debug!("Audio system ready for window {}", self.window_num);

        for signal in sorted_signals.iter() {
            // Report audio playback start
            let candidate_id = format!("{:.1}-{}", signal.frequency_hz / 1e6, self.window_num);
            let signal_metadata = crate::window::WindowMetadata {
                center_frequency_hz: signal.detection_center_freq,
                window_id: self.window_num,
            };
            self.progress_reporter.report(ProgressEvent {
                event_type: ProgressEventType::AudioPlaybackStarted,
                frequency_hz: signal.frequency_hz,
                metadata: signal_metadata,
                candidate_id: Some(candidate_id),
                audio_quality: None,
                signal_strength: None,
                timestamp: std::time::Instant::now(),
            });

            tracing::info!(
                "playing {:.1} MHz [{}]",
                signal.frequency_hz / 1e6,
                signal.audio_quality.to_human_string()
            );
            let sdr_rx = segment.audio_subscriber();

            if let Err(e) = Window::process_signal_for_audio(
                signal,
                sdr_rx,
                audio_tx.clone(),
                &self.config,
                &self.shutdown_token,
                self.pause_signal.as_ref(),
                audio_packet_size,
            ) {
                debug!("Error processing signal for audio: {}", e);
            }

            // Report audio playback completion
            let candidate_id = format!("{:.1}-{}", signal.frequency_hz / 1e6, self.window_num);
            self.progress_reporter.report(ProgressEvent {
                event_type: ProgressEventType::AudioPlaybackCompleted,
                frequency_hz: signal.frequency_hz,
                metadata: signal_metadata,
                candidate_id: Some(candidate_id),
                audio_quality: None,
                signal_strength: None,
                timestamp: std::time::Instant::now(),
            });
        }

        Ok(())
    }

    pub fn process(&self, segment: &dyn Segment) -> Result<()> {
        debug!(
            "Scanning window {} of {} at {:.1} MHz",
            self.window_num,
            self.total_windows,
            self.center_freq / 1e6
        );

        // Get peaks based on mode (station or band scanning)
        let peaks = self.peaks(segment)?;

        // Report peak detection events
        for peak in peaks.iter() {
            let candidate_id = format!("{:.1}-{}", peak.frequency_hz / 1e6, self.window_num);
            self.progress_reporter.report(ProgressEvent {
                event_type: ProgressEventType::PeakDetected,
                frequency_hz: peak.frequency_hz,
                metadata: self.metadata,
                candidate_id: Some(candidate_id),
                audio_quality: None,
                signal_strength: None,
                timestamp: std::time::Instant::now(),
            });
        }

        if !peaks.is_empty() {
            debug!("Found {} peaks in this window", peaks.len());
            self.debug_peaks(&peaks);
            let candidates = self.candidates_from_peaks(&peaks);

            // Report candidate creation events
            for candidate in candidates.iter() {
                let freq = match candidate {
                    crate::types::Candidate::Fm(fm_candidate) => fm_candidate.frequency_hz,
                };
                let candidate_id = format!("{:.1}-{}", freq / 1e6, self.window_num);
                self.progress_reporter.report(ProgressEvent {
                    event_type: ProgressEventType::CandidateCreated,
                    frequency_hz: freq,
                    metadata: self.metadata,
                    candidate_id: Some(candidate_id),
                    audio_quality: None,
                    signal_strength: None,
                    timestamp: std::time::Instant::now(),
                });
            }

            // Process candidates while SDR is still running
            // Candidate analysis now properly waits for detection graphs to complete
            let signals = self.process_candidates(candidates, segment)?;

            // Check for shutdown after candidate processing
            if self.shutdown_token.is_cancelled() {
                debug!("Shutdown requested after candidate processing, skipping audio playback");
                return Ok(());
            }

            // No sleep needed - candidate analysis threads wait for detection completion
            self.play_signals(signals, segment)
        } else {
            debug!("No peaks detected in this window");
            self.debug_peaks(&peaks);
            Ok(())
        }
    }

    /// Wait for threads to complete with a timeout, returns number of threads that completed
    fn wait_for_threads_with_timeout(
        &self,
        threads: Vec<thread::JoinHandle<Result<()>>>,
        timeout: Duration,
    ) -> usize {
        use std::time::Instant;

        let start_time = Instant::now();
        let mut completed = 0;
        let mut remaining_threads = threads;

        // Check threads periodically until timeout
        let check_interval = Duration::from_millis(100);

        while !remaining_threads.is_empty() && start_time.elapsed() < timeout {
            // Check shutdown signal - if shutdown requested, break early
            if self.shutdown_token.is_cancelled() {
                debug!(
                    "Shutdown signal detected, stopping wait for {} remaining threads",
                    remaining_threads.len()
                );
                break;
            }

            // Check pause signal - if paused, continue waiting for threads to finish naturally
            // We can't force-join them because they might be in the middle of rustradio graph execution
            if let Some(ref signal) = self.pause_signal
                && signal.is_paused()
            {
                debug!(
                    "Pause signal detected, continuing to wait for {} remaining threads to finish",
                    remaining_threads.len()
                );
                // Continue the loop to keep checking/joining threads as they complete
            }

            let mut still_running = Vec::new();

            for handle in remaining_threads.into_iter() {
                if handle.is_finished() {
                    // Thread has completed, join it to get result
                    match handle.join() {
                        Ok(Ok(())) => {
                            completed += 1;
                            debug!("Thread {} completed successfully", completed);
                        }
                        Ok(Err(e)) => {
                            completed += 1;
                            debug!("Thread {} completed with error: {}", completed, e);
                        }
                        Err(_) => {
                            debug!("Thread {} panicked", completed + 1);
                        }
                    }
                } else {
                    // Thread still running, keep it for next check
                    still_running.push(handle);
                }
            }

            remaining_threads = still_running;

            if !remaining_threads.is_empty() && start_time.elapsed() < timeout {
                std::thread::sleep(check_interval);
            }
        }

        // Join any remaining threads that timed out to ensure cleanup
        if !remaining_threads.is_empty() {
            debug!(
                "{} threads timed out after {:?}, joining them now",
                remaining_threads.len(),
                timeout
            );
            for handle in remaining_threads.into_iter() {
                let _ = handle.join();
                completed += 1;
            }
        }

        completed
    }
}
