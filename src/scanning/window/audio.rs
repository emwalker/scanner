use cpal::{
    BufferSize, SampleFormat, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use rustradio::graph::{Graph, GraphRunner};
use tokio_util::sync::CancellationToken;
use tracing::debug;

use crate::{
    core::types::{Result, ScannerError, ScanningConfig, Signal},
    ecs::{AudioEntity, Entity},
    pause_signal::PauseSignal,
};

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
    volume: f32,
) -> Result<cpal::Stream> {
    let err_fn = |err| debug!("Audio error: {}", err);

    let underrun_counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let sample_counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let last_log_at = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let underrun_counter_clone = underrun_counter.clone();
    let sample_counter_clone = sample_counter.clone();
    let last_log_at_clone = last_log_at.clone();

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
                    data[filled + i] = (remaining[i] * volume).clamp(-1.0, 1.0);
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
                            data[filled + i] = (samples[i] * volume).clamp(-1.0, 1.0);
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

            let total_samples = sample_counter_clone
                .fetch_add(filled, std::sync::atomic::Ordering::Relaxed)
                + filled;

            // Log audio playback every 48000 samples (1 second at 48kHz)
            let last_log = last_log_at_clone.load(std::sync::atomic::Ordering::Relaxed);
            if total_samples - last_log >= 48000 {
                last_log_at_clone.store(total_samples, std::sync::atomic::Ordering::Relaxed);
                debug!(total_samples = total_samples, "Audio streaming to device");
            }

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

/// Spawn audio entity with all resources allocated
///
/// This function is used by both:
/// - AudioPlaybackSystem (coordinator-based modes: TUI, LogMode)
/// - Window::process (scanning mode)
///
/// Creates an AudioEntity with:
/// - cpal stream (returned separately, can't be in entity - not Send)
/// - Audio graph thread (worker, does FM demod)
/// - All handles stored in allocation component
///
/// Returns (AudioEntity, cpal::Stream) - caller must hold stream to keep audio alive
pub fn spawn_audio_entity(
    signal: Signal,
    sdr_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
    config: &ScanningConfig,
    center_freq: f64,
) -> Result<(AudioEntity, cpal::Stream)> {
    let audio_packet_size = 4096;
    let audio_buffer_packets = 16;
    let (audio_tx, audio_rx) =
        std::sync::mpsc::sync_channel::<crate::mpsc::AudioPacket>(audio_buffer_packets);

    let (audio_device, supported_config) = setup_audio_device(config.audio.sample_rate)?;
    let sample_format = supported_config.sample_format();
    let mut stream_config: StreamConfig = supported_config.into();
    stream_config.buffer_size = BufferSize::Fixed(config.audio.buffer_size);

    let stream = match sample_format {
        SampleFormat::F32 => {
            create_audio_stream(&audio_device, &stream_config, audio_rx, config.audio.volume)?
        }
        _ => {
            return Err(ScannerError::UnsupportedAudioFormat(
                "WAV format required".to_string(),
            ));
        }
    };

    stream.play()?;
    debug!(
        frequency_mhz = signal.frequency_hz / 1e6,
        "Audio stream started"
    );

    let mut audio_graph = create_audio_fm_graph(
        &signal,
        sdr_rx,
        audio_tx,
        config,
        center_freq,
        audio_packet_size,
    )?;

    let cancel_token = audio_graph.cancel_token();
    let frequency_for_log = signal.frequency_hz;
    let graph_handle = std::thread::spawn(move || {
        let _ = thread_priority::set_current_thread_priority(thread_priority::ThreadPriority::Min);
        debug!(
            frequency_mhz = frequency_for_log / 1e6,
            thread_id = ?std::thread::current().id(),
            "Audio graph thread STARTED with low priority, running graph..."
        );
        if let Err(e) = audio_graph.run() {
            debug!(
                frequency_mhz = frequency_for_log / 1e6,
                error = %e,
                "Audio graph error"
            );
        } else {
            debug!(
                frequency_mhz = frequency_for_log / 1e6,
                "Audio graph completed successfully"
            );
        }
        debug!(
            frequency_mhz = frequency_for_log / 1e6,
            thread_id = ?std::thread::current().id(),
            "Audio graph thread EXITING"
        );
    });

    let mut entity = AudioEntity::new(signal.clone(), center_freq, None);
    entity.allocation.set_graph(cancel_token, graph_handle);

    debug!(
        frequency_mhz = signal.frequency_hz / 1e6,
        audio_id = ?entity.id(),
        "AudioEntity spawned with graph resources (stream returned separately)"
    );

    Ok((entity, stream))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use super::*;
    use crate::{
        audio::quality::AudioQuality,
        ecs::{EntityWorld, SignalEntity, SignalId, TaskId, WindowId},
    };

    #[test]
    fn test_signal_analysis_completed_before_playback() {
        let task_id = TaskId::new("test-scan");
        let window_index = 1;
        let frequency_hz = 88.9e6;
        let window_id = WindowId::new(task_id.clone(), window_index);

        let signal = SignalEntity::new(frequency_hz, window_id.clone());
        assert!(
            signal.analysis.is_not_started(),
            "Candidate should start with NotStarted analysis"
        );

        let mut world = EntityWorld::new();
        world.insert(signal);
        let signal_entities = Arc::new(RwLock::new(world));

        let signal = Signal::new_fm(
            frequency_hz,
            0.8,
            200_000.0,
            48000,
            100,
            frequency_hz,
            AudioQuality::Good,
        );

        let id = SignalId::new(signal.frequency_hz, window_id.clone());
        if let Ok(mut entities) = signal_entities.write()
            && let Some(entity) = entities.get_mut(&id)
        {
            entity
                .analysis
                .confirm_analysis(signal.audio_quality, signal.signal_strength as f64);
            entity.info.set_audio_quality(Some(signal.audio_quality));
            entity
                .info
                .set_signal_strength(Some(signal.signal_strength as f64));

            entity
                .playback
                .transition_to(crate::ecs::components::PlaybackState::Playing);
        }

        let entities = signal_entities.read().unwrap();
        let entity = entities.get(&id).unwrap();

        assert!(
            entity.analysis.is_done(),
            "Candidate analysis must be Complete when playback state is Playing"
        );
        assert_eq!(
            entity.playback.state(),
            crate::ecs::components::PlaybackState::Playing,
            "Candidate should be in Playing state"
        );

        let status = entity.status();
        assert!(
            matches!(
                status,
                crate::ecs::components::AnalysisStatus::Signal { .. }
            ),
            "Candidate status should be Signal, not Detected"
        );

        if let crate::ecs::components::AnalysisStatus::Signal { quality, strength } = status {
            assert_eq!(
                quality,
                AudioQuality::Good,
                "Audio quality should be available"
            );
            assert!(
                (strength - 0.8).abs() < 0.01,
                "Signal strength should be approximately 0.8, got {}",
                strength
            );
        }
    }

    #[test]
    fn test_playing_signal_has_non_blank_audio_quality() {
        let task_id = TaskId::new("test-scan");
        let window_index = 2;
        let frequency_hz = 89.3e6;
        let window_id = WindowId::new(task_id.clone(), window_index);

        let signal = SignalEntity::new(frequency_hz, window_id.clone());

        let mut world = EntityWorld::new();
        world.insert(signal);
        let signal_entities = Arc::new(RwLock::new(world));

        let signal = Signal::new_fm(
            frequency_hz,
            0.65,
            200_000.0,
            48000,
            100,
            frequency_hz,
            AudioQuality::Moderate,
        );

        let id = SignalId::new(signal.frequency_hz, window_id.clone());
        if let Ok(mut entities) = signal_entities.write()
            && let Some(entity) = entities.get_mut(&id)
        {
            entity
                .analysis
                .confirm_analysis(signal.audio_quality, signal.signal_strength as f64);
            entity.info.set_audio_quality(Some(signal.audio_quality));
            entity
                .info
                .set_signal_strength(Some(signal.signal_strength as f64));

            entity
                .playback
                .transition_to(crate::ecs::components::PlaybackState::Playing);
        }

        let entities = signal_entities.read().unwrap();
        let entity = entities.get(&id).unwrap();
        let status = entity.status();

        if let crate::ecs::components::AnalysisStatus::Signal { quality, .. } = status {
            assert_eq!(
                quality,
                AudioQuality::Moderate,
                "Playing signal must have audio quality data (not blank)"
            );
        } else {
            panic!("Playing signal must have Signal status with audio quality");
        }
    }
}
