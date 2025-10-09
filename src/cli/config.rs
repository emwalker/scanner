use crate::audio::quality::{AudioAnalyzer, AudioQuality};
use crate::core::types::{Format, Result, ScannerError, ScanningConfig};

use super::args::{AudioClassifier, ScanArgs};

pub fn parse_squelch_threshold(threshold_str: &str) -> Result<AudioQuality> {
    match threshold_str.to_lowercase().as_str() {
        "static" => Ok(AudioQuality::Static),
        "no-audio" => Ok(AudioQuality::NoAudio),
        "poor" => Ok(AudioQuality::Poor),
        "moderate" => Ok(AudioQuality::Moderate),
        "good" => Ok(AudioQuality::Good),
        _ => Err(ScannerError::Custom(format!(
            "Invalid squelch threshold '{}'. Valid values: static, no-audio, poor, moderate, good",
            threshold_str
        ))),
    }
}

pub fn create_audio_analyzer(
    classifier_type: AudioClassifier,
    sample_rate: f32,
    model_path: Option<&str>,
) -> Result<AudioAnalyzer> {
    let classifier: Box<dyn crate::audio::quality::Classifier> = match classifier_type {
        AudioClassifier::Heuristic1 => Box::new(
            crate::audio::quality::heuristic1::Classifier::new(sample_rate),
        ),
        AudioClassifier::Heuristic2 => Box::new(
            crate::audio::quality::heuristic2::Classifier::new(sample_rate),
        ),
        AudioClassifier::Heuristic3 => Box::new(
            crate::audio::quality::heuristic3::Classifier::new(sample_rate),
        ),
        AudioClassifier::RandomForest => {
            use super::model::discover_latest_model;

            let discovered_path = model_path
                .map(|s| s.to_string())
                .or_else(discover_latest_model);

            match discovered_path {
                Some(path) => {
                    tracing::debug!(model_path = %path, "Attempting to load Random Forest model");
                    match crate::audio::quality::random_forest::Classifier::load_pretrained(&path) {
                        Ok(classifier) => {
                            tracing::debug!(model_path = %path, "Successfully loaded Random Forest model");
                            Box::new(classifier)
                        }
                        Err(e) => {
                            tracing::warn!(
                                model_path = %path,
                                error = %e,
                                "Failed to load pre-trained model, falling back to heuristic1 classifier"
                            );
                            Box::new(crate::audio::quality::heuristic1::Classifier::new(
                                sample_rate,
                            ))
                        }
                    }
                }
                None => {
                    tracing::warn!(
                        "No Random Forest model found, falling back to heuristic1 classifier"
                    );
                    tracing::info!("To train a model, run: scanner train");
                    Box::new(crate::audio::quality::heuristic1::Classifier::new(
                        sample_rate,
                    ))
                }
            }
        }
    };
    Ok(AudioAnalyzer::new(classifier))
}

pub fn determine_format(args: &ScanArgs) -> Format {
    if args.json {
        Format::Json
    } else if args.log {
        Format::Log
    } else {
        Format::Text
    }
}

pub fn build_scanning_config(args: &ScanArgs) -> Result<ScanningConfig> {
    let squelch_threshold = if args.disable_squelch {
        AudioQuality::Static
    } else {
        parse_squelch_threshold(&args.squelch_threshold)?
    };

    let audio_analyzer = create_audio_analyzer(
        args.audio_classifier.clone(),
        48000.0,
        args.model_path.as_deref(),
    )?;

    tracing::debug!(
        classifier = audio_analyzer.classifier_name(),
        squelch_threshold = format!("{:?}", squelch_threshold),
        "Audio analyzer initialized"
    );

    Ok(ScanningConfig {
        audio_buffer_size: 8192,
        audio_sample_rate: 48000,
        band: args.band,
        capture_audio_duration: args.audio_capture_duration,
        capture_audio: args.audio_capture_dir.clone(),
        capture_duration: args.capture_duration,
        capture_iq: args.capture_iq.clone(),
        debug_pipeline: args.debug_pipeline,
        duration: args.duration,
        sdr_gain: args.gain.unwrap_or(24.0),
        scanning_windows: args.scanning_windows,
        fft_size: 1024,
        peak_detection_threshold: 1.0,
        peak_scan_duration: args.peak_scan_duration,
        print_candidates: args.print_candidates,
        samp_rate: 2_000_000.0f64,
        squelch_learning_duration: args.learning_duration,
        frequency_tracking_method: args.frequency_tracking.clone(),
        tracking_accuracy: args.tracking_accuracy,
        disable_frequency_tracking: args.disable_frequency_tracking,
        spectral_threshold: args.spectral_threshold,
        agc_settling_time: args.agc_settling_time,
        window_overlap: args.window_overlap,
        disable_squelch: args.disable_squelch,
        squelch_threshold,
        disable_if_agc: args.disable_if_agc,
        audio_analyzer,
        enable_exponential_smoothing: !args.disable_signal_averaging,
        enable_multi_frame_averaging: !args.disable_signal_averaging,
        enable_coherent_integration: !args.disable_signal_averaging,
        enable_moving_average_filter: !args.disable_signal_averaging,
        enable_cfar_detection: !args.disable_cfar,
        enable_windowing: !args.disable_spectral_preprocessing,
        enable_dynamic_noise_floor: args.enable_dynamic_noise_floor,
        enable_multi_frame_integration: args.enable_multi_frame_integration,
        ..Default::default()
    })
}
