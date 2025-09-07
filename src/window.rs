use crate::types::{Result, ScanningConfig};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, SampleFormat, StreamConfig};
use rustradio::graph::GraphRunner;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tracing::debug;

/// Represents a frequency window for band scanning with complete lifecycle management
pub struct Window {
    center_freq: f64,
    window_num: usize,
    total_windows: usize,
    station_mode: bool, // True if this is a specific station frequency, not band scanning
}

impl Window {
    pub fn new(center_freq: f64, window_num: usize, total_windows: usize) -> Self {
        Self {
            center_freq,
            window_num,
            total_windows,
            station_mode: false,
        }
    }

    pub fn for_station(center_freq: f64, window_num: usize, total_windows: usize) -> Self {
        Self {
            center_freq,
            window_num,
            total_windows,
            station_mode: true,
        }
    }

    fn get_peaks(
        &self,
        config: &ScanningConfig,
        sdr_manager: &Arc<Mutex<crate::sdr::SdrManager>>,
    ) -> Result<Vec<crate::types::Peak>> {
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
            let sdr_rx_for_peaks = sdr_manager.lock().unwrap().get_audio_subscriber();
            crate::fm::collect_peaks(config, sdr_rx_for_peaks, self.center_freq)
        }
    }

    fn debug_peaks(&self, peaks: &[crate::types::Peak], config: &ScanningConfig) {
        if config.debug_pipeline {
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

    fn create_candidates_from_peaks(
        &self,
        peaks: &[crate::types::Peak],
        config: &ScanningConfig,
    ) -> Vec<crate::types::Candidate> {
        let mut candidates = Vec::new();

        for candidate in crate::fm::find_candidates(peaks, config, self.center_freq) {
            if config.debug_pipeline {
                let frequency_offset = candidate.frequency_hz() - self.center_freq;
                debug!(
                    message = "Candidate created",
                    candidate_frequency_mhz = candidate.frequency_hz() / 1e6,
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
        config: &ScanningConfig,
        sdr_manager: &Arc<Mutex<crate::sdr::SdrManager>>,
    ) -> Result<Vec<crate::types::Signal>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let candidate_count = candidates.len();
        let mut candidate_threads = Vec::new();
        let (signal_tx, signal_rx) = std::sync::mpsc::sync_channel::<crate::types::Signal>(100);

        for candidate in candidates {
            if config.print_candidates {
                tracing::info!(
                    "candidate found at {:.1} MHz",
                    candidate.frequency_hz() / 1e6
                );
                continue;
            }

            let sdr_rx = sdr_manager.lock().unwrap().get_audio_subscriber();
            let signal_tx_clone = signal_tx.clone();
            let config_clone = config.clone();
            let center_freq = self.center_freq;

            let handle = thread::spawn(move || -> Result<()> {
                candidate.analyze(&config_clone, sdr_rx, center_freq, signal_tx_clone)
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

    pub fn setup_audio_device(
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

    pub fn create_audio_stream(
        device: &cpal::Device,
        stream_config: &StreamConfig,
        audio_rx: std::sync::mpsc::Receiver<f32>,
    ) -> Result<cpal::Stream> {
        let err_fn = |err| debug!("Audio error: {}", err);

        let stream = device.build_output_stream(
            stream_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                // More efficient batch processing - try to fill buffer in chunks
                let mut filled = 0;
                while filled < data.len() {
                    match audio_rx.try_recv() {
                        Ok(audio_sample) => {
                            data[filled] = audio_sample.clamp(-1.0, 1.0);
                            filled += 1;
                        }
                        Err(_) => {
                            // Fill remaining with silence to avoid underruns
                            for sample in &mut data[filled..] {
                                *sample = 0.0;
                            }
                            break;
                        }
                    }
                }
            },
            err_fn,
            None,
        )?;

        Ok(stream)
    }

    pub fn process_signal_for_audio(
        signal: &crate::types::Signal,
        sdr_rx: tokio::sync::broadcast::Receiver<rustradio::Complex>,
        audio_tx: std::sync::mpsc::SyncSender<f32>,
        config: &ScanningConfig,
        shutdown_flag: Arc<std::sync::atomic::AtomicBool>,
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
        )?;

        let duration = std::time::Duration::from_secs(config.duration);
        debug!("Playing audio for {:?}", duration);
        debug!("Starting audio graph thread...");

        let cancel_token = audio_graph.cancel_token();
        let graph_handle = std::thread::spawn(move || {
            debug!("Audio graph thread started, running graph...");
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
            if shutdown_flag.load(std::sync::atomic::Ordering::Relaxed) {
                debug!("Shutdown requested during audio processing, stopping early");
                break;
            }
        }

        debug!("Cancelling audio graph...");
        cancel_token.cancel();
        debug!("Waiting for audio graph thread to finish...");
        let _ = graph_handle.join();
        debug!("Audio graph thread finished");

        debug!(
            "Finished playing audio for {:.1} MHz",
            signal.frequency_hz / 1e6
        );
        Ok(())
    }

    fn setup_audio_graph_source(
        sdr_rx: tokio::sync::broadcast::Receiver<rustradio::Complex>,
        graph: &mut rustradio::graph::Graph,
    ) -> rustradio::stream::ReadStream<rustradio::Complex> {
        let (source_block, stream) = crate::broadcast::BroadcastSource::new(sdr_rx);
        graph.add(Box::new(source_block));
        stream
    }

    fn create_frequency_xlating_filter(
        prev: rustradio::stream::ReadStream<rustradio::Complex>,
        graph: &mut rustradio::graph::Graph,
        frequency_offset: f64,
        config: &ScanningConfig,
    ) -> Result<(rustradio::stream::ReadStream<rustradio::Complex>, u32)> {
        use rustradio::{fir, window::WindowType};

        let filter_config = crate::fm::filter_config::FmFilterConfig::for_purpose(
            crate::fm::filter_config::FilterPurpose::Audio,
            config.samp_rate,
        );

        debug!(
            message = "Audio stage filter configuration",
            passband_khz = filter_config.channel_bandwidth / 1000.0,
            transition_khz = filter_config.transition_width / 1000.0,
            decimation = filter_config.decimation,
            estimated_taps = filter_config.estimated_taps,
            estimated_mflops = filter_config.estimated_mflops
        );

        let taps = fir::low_pass(
            config.samp_rate as f32,
            filter_config.cutoff_frequency(),
            filter_config.transition_width,
            &WindowType::Hamming,
        );

        let decimation = filter_config.decimation;
        let (freq_xlating_block, stream) = crate::freq_xlating_fir::FreqXlatingFir::with_real_taps(
            prev,
            &taps,
            frequency_offset as f32,
            config.samp_rate as f32,
            decimation,
        );
        graph.add(Box::new(freq_xlating_block));

        Ok((stream, decimation.try_into().unwrap()))
    }

    fn create_fm_demodulation_chain(
        prev: rustradio::stream::ReadStream<rustradio::Complex>,
        graph: &mut rustradio::graph::Graph,
        quad_rate: f32,
    ) -> rustradio::stream::ReadStream<rustradio::Float> {
        use rustradio::{blockchain, blocks::QuadratureDemod};

        let fm_gain = (quad_rate / (2.0 * 75_000.0)) * 0.8;
        let prev = blockchain![graph, prev, QuadratureDemod::new(prev, fm_gain)];

        let (deemphasis_block, deemphasis_stream) =
            crate::fm::deemph::Deemphasis::new(prev, quad_rate, 75.0);
        graph.add(Box::new(deemphasis_block));
        deemphasis_stream
    }

    fn create_audio_decimation_chain(
        prev: rustradio::stream::ReadStream<rustradio::Float>,
        graph: &mut rustradio::graph::Graph,
        quad_rate: f32,
    ) -> Result<rustradio::stream::ReadStream<rustradio::Float>> {
        use rustradio::{
            Float, blockchain, blocks::RationalResampler, fir, fir::FirFilter, window::WindowType,
        };

        let target_audio_rate = 48_000.0;
        let decimation_factor = (quad_rate / target_audio_rate).round() as usize;
        let actual_audio_rate = quad_rate / decimation_factor as f32;

        debug!(
            "Audio decimation: {:.1} kHz → {:.1} kHz (factor {})",
            quad_rate / 1000.0,
            actual_audio_rate / 1000.0,
            decimation_factor
        );

        let nyquist_freq = actual_audio_rate / 2.0;
        let audio_cutoff = (nyquist_freq * 0.8).min(15_000.0);
        let transition_width = nyquist_freq * 0.4;

        let taps = fir::low_pass(
            quad_rate,
            audio_cutoff,
            transition_width,
            &WindowType::Hamming,
        );
        let prev = blockchain![graph, prev, FirFilter::new(prev, &taps)];

        let prev = blockchain![
            graph,
            prev,
            RationalResampler::<Float>::builder()
                .interp(1)
                .deci(decimation_factor)
                .build(prev)?
        ];

        Ok(prev)
    }

    fn create_audio_fm_graph(
        signal: &crate::types::Signal,
        sdr_rx: tokio::sync::broadcast::Receiver<rustradio::Complex>,
        audio_tx: std::sync::mpsc::SyncSender<f32>,
        config: &ScanningConfig,
        center_freq: f64,
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

        let prev = Self::create_fm_demodulation_chain(prev, &mut graph, quad_rate);
        let prev = Self::create_audio_decimation_chain(prev, &mut graph, quad_rate)?;

        graph.add(Box::new(crate::mpsc::MpscSink::new(
            prev,
            audio_tx,
            station_name,
        )));
        Ok(graph)
    }

    fn play_signals(
        &self,
        signals: Vec<crate::types::Signal>,
        config: &ScanningConfig,
        sdr_manager: &Arc<Mutex<crate::sdr::SdrManager>>,
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
        let audio_buffer_samples = (config.audio_sample_rate as f32 * 0.25) as usize;
        let (audio_tx, audio_rx) = std::sync::mpsc::sync_channel::<f32>(audio_buffer_samples);

        // Setup audio device and stream
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
        debug!("Audio system ready for window {}", self.window_num);

        for signal in sorted_signals {
            tracing::info!("playing {:.1} MHz", signal.frequency_hz / 1e6);
            let sdr_rx = sdr_manager.lock().unwrap().get_audio_subscriber();
            let shutdown_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

            if let Err(e) = Window::process_signal_for_audio(
                &signal,
                sdr_rx,
                audio_tx.clone(),
                config,
                shutdown_flag,
            ) {
                debug!("Error processing signal for audio: {}", e);
            }
        }

        Ok(())
    }

    /// Process this window completely: tune SDR, find candidates, run detection/audio, wait for completion
    pub fn process(&self, config: &ScanningConfig) -> Result<()> {
        debug!(
            "Scanning window {} of {} at {:.1} MHz",
            self.window_num,
            self.total_windows,
            self.center_freq / 1e6
        );

        // Create a dedicated SDR manager for this window with the specific center frequency
        // This ensures the SDR runs continuously throughout the entire window processing
        let sdr_manager = crate::sdr::SdrManager::new(config, self.center_freq)?;
        let sdr_manager_arc = Arc::new(Mutex::new(sdr_manager));

        // Get a subscriber that we'll keep alive throughout the entire window processing
        let _sdr_rx_keeper = sdr_manager_arc.lock().unwrap().get_audio_subscriber();

        // Get peaks based on mode (station or band scanning)
        let peaks = self.get_peaks(config, &sdr_manager_arc)?;

        if !peaks.is_empty() {
            debug!("Found {} peaks in this window", peaks.len());
            self.debug_peaks(&peaks, config);
            let candidates = self.create_candidates_from_peaks(&peaks, config);

            // Process candidates while SDR is still running
            // Candidate analysis now properly waits for detection graphs to complete
            let signals = self.process_candidates(candidates, config, &sdr_manager_arc)?;

            // No sleep needed - candidate analysis threads wait for detection completion
            self.play_signals(signals, config, &sdr_manager_arc)
        } else {
            debug!("No peaks detected in this window");
            self.debug_peaks(&peaks, config);
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

        for (i, handle) in threads.into_iter().enumerate() {
            let remaining_time = timeout.saturating_sub(start_time.elapsed());

            if remaining_time.is_zero() {
                debug!("Thread {} timed out, not waiting for completion", i + 1);
                break;
            }

            // Try to join the thread
            match handle.join() {
                Ok(Ok(())) => {
                    completed += 1;
                    debug!("Thread {} completed successfully", i + 1);
                }
                Ok(Err(e)) => {
                    completed += 1;
                    debug!("Thread {} completed with error: {}", i + 1, e);
                }
                Err(_) => {
                    debug!("Thread {} panicked", i + 1);
                }
            }
        }

        completed
    }
}
