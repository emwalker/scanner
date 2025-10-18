use crate::core::types::{Result, ScannerError, ScanningConfig};
use crate::hardware::pool::SegmentTrait;
use crate::pause_signal::PauseSignal;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, SampleFormat, StreamConfig};
use rustradio::graph::{Graph, GraphRunner};
use tokio_util::sync::CancellationToken;
use tracing::debug;

pub fn setup_audio_device(
    audio_sample_rate: u32,
) -> Result<(cpal::Device, cpal::SupportedStreamConfig)> {
    let host = cpal::default_host();
    let audio_device = host
        .default_output_device()
        .ok_or_else(|| ScannerError::Custom("No audio output device available".to_string()))?;

    let supported_configs_range = audio_device.supported_output_configs()?;

    let supported_config = supported_configs_range
        .filter(|d| d.sample_format() == SampleFormat::F32)
        .find(|d| {
            d.min_sample_rate().0 <= audio_sample_rate && d.max_sample_rate().0 >= audio_sample_rate
        })
        .ok_or_else(|| {
            ScannerError::UnsupportedAudioFormat(format!(
                "No F32 audio config found for sample rate {}",
                audio_sample_rate
            ))
        })?
        .with_sample_rate(cpal::SampleRate(audio_sample_rate));

    Ok((audio_device, supported_config))
}

pub fn create_audio_stream(
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
                let underrun_count =
                    underrun_counter_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
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
    signal: &crate::core::types::Signal,
    sdr_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
    audio_tx: std::sync::mpsc::SyncSender<crate::mpsc::AudioPacket>,
    config: &ScanningConfig,
    shutdown_token: &CancellationToken,
    pause_signal: Option<&PauseSignal>,
    audio_packet_size: usize,
) -> Result<()> {
    debug!(
        "Creating audio processing pipeline for {:.1} MHz",
        signal.frequency_hz / 1e6
    );

    let mut audio_graph = create_audio_fm_graph(
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
        let _ = thread_priority::set_current_thread_priority(thread_priority::ThreadPriority::Min);
        debug!("Audio graph thread started with low priority, running graph...");
        if let Err(e) = audio_graph.run() {
            debug!("Audio graph error: {}", e);
        } else {
            debug!("Audio graph completed successfully");
        }
    });

    let check_interval = std::time::Duration::from_millis(100);
    let mut remaining = duration;

    while !remaining.is_zero() {
        let sleep_duration = std::cmp::min(remaining, check_interval);
        std::thread::sleep(sleep_duration);
        remaining = remaining.saturating_sub(sleep_duration);

        if shutdown_token.is_cancelled() {
            debug!("Shutdown requested during audio processing, stopping early");
            break;
        }

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
    graph: &mut Graph,
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
    graph: &mut Graph,
    frequency_offset: f64,
    config: &ScanningConfig,
) -> Result<(rustradio::stream::ReadStream<rustradio::Complex>, u32)> {
    crate::signal::pipeline_builder::FmPipelineBuilder::create_frequency_xlating_filter(
        prev,
        graph,
        frequency_offset,
        config,
        crate::signal::filter_config::FilterPurpose::Audio,
    )
    .map_err(Into::into)
}

fn create_fm_demodulation_chain(
    prev: rustradio::stream::ReadStream<rustradio::Complex>,
    graph: &mut Graph,
    quad_rate: f32,
    signal: &crate::core::types::Signal,
) -> (rustradio::stream::ReadStream<rustradio::Float>, f32) {
    use rustradio::{blockchain, blocks::QuadratureDemod};

    let base_gain = (quad_rate / (2.0 * 75_000.0)) * 0.8;
    let (fm_gain, quality_adjustment) = calculate_adaptive_fm_gain(base_gain, signal);

    debug!(
        "FM demodulator gain: base={:.3}, adaptive={:.3}, signal_strength={:.6}",
        base_gain, fm_gain, signal.signal_strength
    );

    let prev = blockchain![graph, prev, QuadratureDemod::new(prev, fm_gain)];

    let (deemphasis_block, deemphasis_stream) =
        crate::signal::deemph::Deemphasis::new(prev, quad_rate, 75.0);
    graph.add(Box::new(deemphasis_block));
    (deemphasis_stream, quality_adjustment)
}

fn calculate_adaptive_fm_gain(base_gain: f32, signal: &crate::core::types::Signal) -> (f32, f32) {
    let gain_adjustment = if signal.signal_strength < 0.001 {
        5.0
    } else if signal.signal_strength < 0.01 {
        3.0
    } else if signal.signal_strength < 0.1 {
        1.5
    } else {
        1.0
    };

    let quality_adjustment = match signal.audio_quality {
        crate::audio::quality::AudioQuality::Good => 1.0,
        crate::audio::quality::AudioQuality::Moderate => 1.2,
        crate::audio::quality::AudioQuality::Poor => 1.5,
        crate::audio::quality::AudioQuality::NoAudio => 1.0,
        crate::audio::quality::AudioQuality::Static => 0.5,
        crate::audio::quality::AudioQuality::Unknown => 1.0,
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
    graph: &mut Graph,
    quad_rate: f32,
    config: &ScanningConfig,
) -> Result<rustradio::stream::ReadStream<rustradio::Float>> {
    crate::signal::pipeline_builder::FmPipelineBuilder::create_audio_decimation_chain(
        prev, graph, quad_rate, config, "Audio",
    )
    .map_err(Into::into)
}

pub fn create_audio_fm_graph(
    signal: &crate::core::types::Signal,
    sdr_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
    audio_tx: std::sync::mpsc::SyncSender<crate::mpsc::AudioPacket>,
    config: &ScanningConfig,
    center_freq: f64,
    audio_packet_size: usize,
) -> Result<Graph> {
    let mut graph = Graph::new();
    let station_name = format!("{:.1}FM_Audio", signal.frequency_hz / 1e6);

    let frequency_offset = signal.frequency_hz - center_freq;
    debug!(
        "Audio graph: signal {:.1} MHz, center {:.1} MHz, offset {:.1} kHz",
        signal.frequency_hz / 1e6,
        center_freq / 1e6,
        frequency_offset / 1e3
    );

    let prev = setup_audio_graph_source(sdr_rx, &mut graph);
    let (prev, decimation) =
        create_frequency_xlating_filter(prev, &mut graph, frequency_offset, config)?;

    let decimated_samp_rate = config.samp_rate / decimation as f64;
    let quad_rate = decimated_samp_rate as f32;

    let (prev, quality_adjustment) =
        create_fm_demodulation_chain(prev, &mut graph, quad_rate, signal);
    let prev = create_audio_decimation_chain(prev, &mut graph, quad_rate, config)?;

    let (diagnostic_block, prev) = crate::broadcast::AudioDiagnostic::new(prev, quality_adjustment);
    graph.add(Box::new(diagnostic_block));

    graph.add(Box::new(crate::mpsc::MpscSink::new(
        prev,
        audio_tx,
        station_name,
        audio_packet_size,
    )));
    Ok(graph)
}

pub(super) fn play_signals(
    window_num: usize,
    config: &ScanningConfig,
    shutdown_token: &CancellationToken,
    pause_signal: &Option<PauseSignal>,
    signals: Vec<crate::core::types::Signal>,
    segment: &dyn SegmentTrait,
    candidate_entities: &Option<crate::ecs::Entities<crate::ecs::CandidateEntity>>,
) -> Result<()> {
    if signals.is_empty() {
        return Ok(());
    }

    let mut sorted_signals = signals;
    sorted_signals.sort_by(|a, b| {
        a.frequency_hz
            .partial_cmp(&b.frequency_hz)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    debug!(
        "Window {} playing {} signals in frequency order",
        window_num,
        sorted_signals.len()
    );

    let audio_packet_size = 4096;
    let audio_buffer_packets = 16;
    let (audio_tx, audio_rx) =
        std::sync::mpsc::sync_channel::<crate::mpsc::AudioPacket>(audio_buffer_packets);

    let (audio_device, supported_config) = setup_audio_device(config.audio.sample_rate)?;
    let sample_format = supported_config.sample_format();
    let mut stream_config: StreamConfig = supported_config.into();
    stream_config.buffer_size = BufferSize::Fixed(config.audio.buffer_size);

    let stream = match sample_format {
        SampleFormat::F32 => create_audio_stream(&audio_device, &stream_config, audio_rx)?,
        _ => {
            return Err(ScannerError::UnsupportedAudioFormat(
                "WAV format required".to_string(),
            ));
        }
    };

    stream.play()?;
    debug!("Audio system ready for window {}", window_num);

    for signal in sorted_signals.iter() {
        // Update entity to Playing state
        if let Some(entities_arc) = candidate_entities {
            use crate::ecs::CandidateId;
            let id = CandidateId::new(signal.frequency_hz, window_num);
            if let Ok(mut entities) = entities_arc.write()
                && let Some(entity) = entities.get_mut(&id)
            {
                entity.start_playback();
            }
        }

        tracing::info!(
            "playing {:.1} MHz [{}]",
            signal.frequency_hz / 1e6,
            signal.audio_quality.to_human_string()
        );
        let sdr_rx = segment.audio_subscriber();

        if let Err(e) = process_signal_for_audio(
            signal,
            sdr_rx,
            audio_tx.clone(),
            config,
            shutdown_token,
            pause_signal.as_ref(),
            audio_packet_size,
        ) {
            debug!("Error processing signal for audio: {}", e);
        }

        // Update entity to Completed state
        if let Some(entities_arc) = candidate_entities {
            use crate::ecs::CandidateId;
            let id = CandidateId::new(signal.frequency_hz, window_num);
            if let Ok(mut entities) = entities_arc.write()
                && let Some(entity) = entities.get_mut(&id)
            {
                entity.complete_playback();
            }
        }
    }

    Ok(())
}
