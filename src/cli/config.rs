use super::args::{AudioClassifier, ScanArgs};
use crate::audio::quality::{AudioAnalyzer, AudioQuality};
use crate::core::config::{
    AudioConfig, AveragingConfig, CaptureConfig, CfarConfig, DebugConfig,
    ExponentialSmoothingConfig, FrequencyTrackingConfig, MovingAverageConfig,
    MultiFrameAveragingConfig, MultiFrameConfig, NoiseFloorConfig, PeakDetectionConfig,
    SignalProcessingConfig, SquelchConfig, WindowingConfig,
};
use crate::core::types::{Format, Result, ScannerError, ScanningConfig};

pub fn parse_squelch_threshold(threshold_str: &str) -> Result<AudioQuality> {
    match threshold_str.to_lowercase().as_str() {
        "static" => Ok(AudioQuality::Static),
        "no-audio" => Ok(AudioQuality::NoAudio),
        "poor" => Ok(AudioQuality::Poor),
        "moderate" => Ok(AudioQuality::Moderate),
        "good" => Ok(AudioQuality::Good),
        _ => Err(ScannerError::InvalidSquelchThreshold {
            value: threshold_str.to_string(),
            valid_values: "static, no-audio, poor, moderate, good",
        }),
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
        band: args.band,
        duration: args.duration,
        samp_rate: 2_000_000.0f64,
        sdr_gain: args.gain.unwrap_or(24.0),
        scanning_windows: args.scanning_windows,
        audio: AudioConfig {
            buffer_size: 8192,
            sample_rate: 48000,
            analyzer: audio_analyzer,
            squelch: SquelchConfig {
                disabled: args.disable_squelch,
                threshold: squelch_threshold,
                learning_duration: args.learning_duration,
            },
        },
        peak_detection: PeakDetectionConfig {
            fft_size: 1024,
            threshold: 1.0,
            scan_duration: args.peak_scan_duration,
            spectral_threshold: args.spectral_threshold,
            cfar: CfarConfig {
                enabled: !args.disable_cfar,
                ..Default::default()
            },
            noise_floor: NoiseFloorConfig {
                enabled: args.enable_dynamic_noise_floor,
                ..Default::default()
            },
            windowing: WindowingConfig {
                enabled: !args.disable_spectral_preprocessing,
                ..Default::default()
            },
            averaging: AveragingConfig {
                exponential_smoothing: ExponentialSmoothingConfig {
                    enabled: !args.disable_signal_averaging,
                    ..Default::default()
                },
                multi_frame_averaging: MultiFrameAveragingConfig {
                    enabled: !args.disable_signal_averaging,
                    ..Default::default()
                },
                coherent_integration_enabled: !args.disable_signal_averaging,
                moving_average: MovingAverageConfig {
                    enabled: !args.disable_signal_averaging,
                    ..Default::default()
                },
            },
            multi_frame: MultiFrameConfig {
                enabled: args.enable_multi_frame_integration,
                ..Default::default()
            },
        },
        signal_processing: SignalProcessingConfig {
            agc_settling_time: args.agc_settling_time,
            disable_if_agc: args.disable_if_agc,
            window_overlap: args.window_overlap,
            frequency_tracking: FrequencyTrackingConfig {
                disabled: args.disable_frequency_tracking,
                method: args.frequency_tracking.clone(),
                accuracy: args.tracking_accuracy,
            },
            ..Default::default()
        },
        capture: CaptureConfig {
            audio_path: args.audio_capture_dir.clone(),
            audio_duration: args.audio_capture_duration,
            iq_path: args.capture_iq.clone(),
            iq_duration: args.capture_duration,
        },
        debug: DebugConfig {
            pipeline: args.debug_pipeline,
            print_candidates: args.print_candidates,
        },
    })
}
