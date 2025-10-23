use std::sync::{Arc, atomic::AtomicU8};

use rustradio::{
    blockchain,
    blocks::QuadratureDemod,
    graph::{Graph, GraphRunner},
};
use tracing::debug;

use crate::{
    core::config::ScanningConfig,
    ecs::WindowId,
    file::AudioCaptureBlock,
    signal::{
        deemph::Deemphasis,
        filter_config::FilterPurpose,
        pipeline_builder,
        squelch::{SquelchBlock, SquelchConfig},
    },
};

pub struct DetectionGraphConfig<'a> {
    pub source_receiver: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
    pub samp_rate: f64,
    pub config: &'a ScanningConfig,
    pub center_freq: f64,
    pub tune_freq: f64,
    pub signal_tx: Option<std::sync::mpsc::Sender<crate::core::types::Signal>>,
    pub audio_analyzer: crate::audio::quality::AudioAnalyzer,
    pub window_id: WindowId,
}

pub fn create_detection_graph(
    graph_config: DetectionGraphConfig,
) -> rustradio::Result<(Graph, Arc<AtomicU8>)> {
    let DetectionGraphConfig {
        source_receiver,
        samp_rate,
        config,
        center_freq,
        tune_freq,
        signal_tx,
        audio_analyzer,
        window_id,
    } = graph_config;
    let mut graph = Graph::new();

    let (source_block, prev) = crate::broadcast::BroadcastSource::new(source_receiver);
    graph.add(Box::new(source_block));

    let frequency_offset = tune_freq - center_freq;

    let (prev, decimation) = pipeline_builder::FmPipelineBuilder::create_frequency_xlating_filter(
        prev,
        &mut graph,
        frequency_offset,
        config,
        FilterPurpose::Audio,
    )?;

    let decimated_samp_rate = samp_rate / decimation as f64;

    let quad_rate = decimated_samp_rate as f32;

    let fm_gain = (quad_rate / (2.0 * 75_000.0)) * 0.8;
    let prev = blockchain![graph, prev, QuadratureDemod::new(prev, fm_gain)];

    let (deemphasis_block, prev) = Deemphasis::new(prev, quad_rate, 75.0);
    graph.add(Box::new(deemphasis_block));

    let prev = pipeline_builder::FmPipelineBuilder::create_audio_decimation_chain(
        prev,
        &mut graph,
        quad_rate,
        config,
        "Detection",
    )?;

    let analysis_rate = config.audio.sample_rate as f32;

    let audio_capturer = if let Some(ref capture_dir) = config.capture.audio_path {
        let audio_config = crate::file::AudioCaptureConfig {
            output_dir: capture_dir.clone(),
            sample_rate: analysis_rate,
            capture_duration: config.capture.audio_duration,
            frequency_hz: tune_freq,
            modulation_type: crate::core::types::ModulationType::WFM,
        };
        match crate::file::AudioCaptureSink::new(audio_config) {
            Ok(capturer) => Some(capturer),
            Err(e) => {
                debug!("Failed to create audio capturer: {}", e);
                None
            }
        }
    } else {
        None
    };

    let (audio_capture_block, audio_capture_output) = AudioCaptureBlock::new(prev, None);
    graph.add(Box::new(audio_capture_block));
    let prev = audio_capture_output;

    let squelch_config = SquelchConfig {
        sample_rate: analysis_rate,
        learning_duration: config.audio.squelch.learning_duration,
        signal_tx,
        frequency_hz: tune_freq,
        center_freq,
        squelch_disabled: config.audio.squelch.disabled,
        threshold: config.audio.squelch.threshold,
        fft_size: config.peak_detection.fft_size,
        audio_analyzer,
        audio_capturer,
        window_id,
        tuner_id: None,
    };
    let (squelch_block, decision_state) = SquelchBlock::new(prev, squelch_config);
    graph.add(Box::new(squelch_block));

    Ok((graph, decision_state))
}
