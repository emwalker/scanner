use clap::{Parser, Subcommand, ValueEnum};
use scanner::audio_quality::AudioAnalyzer;
use scanner::audio_quality::AudioQuality;
use scanner::logging::DefaultLogger;
use scanner::main_thread::{DefaultConsoleWriter, MainThread};
use scanner::soapy;
use scanner::types::{Band, Format, Result, ScanningConfig};
use std::sync::Arc;

const DEFAULT_DRIVER: &str = "driver=sdrplay";

#[derive(ValueEnum, Clone, Debug)]
enum AudioClassifier {
    Heuristic1,
    Heuristic2,
    Heuristic3,
    RandomForest,
}

#[derive(Parser, Debug)]
#[command(name = "scanner")]
#[command(about = "FM radio scanner with audio quality analysis")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Scan for FM radio stations
    Scan(Box<ScanArgs>),
    /// Train audio quality machine learning model
    Train(TrainArgs),
}

#[derive(Parser, Debug)]
struct ScanArgs {
    /// AGC settling time in seconds before peak scanning begins
    #[arg(long, default_value_t = 3.0)]
    agc_settling_time: f64,

    #[arg(long)]
    audio_capture_dir: Option<String>,

    #[arg(long, default_value_t = 2.0)]
    audio_capture_duration: f64,

    /// Audio quality classifier to use
    #[arg(long, default_value = "random-forest")]
    audio_classifier: AudioClassifier,

    #[arg(long, default_value_t = Band::Fm)]
    band: Band,

    #[arg(long, default_value_t = 2.0)]
    capture_duration: f64,

    #[arg(long)]
    capture_iq: Option<String>,

    #[arg(long)]
    debug_pipeline: bool,

    #[arg(long)]
    device_args: Option<String>,

    /// Disable frequency tracking (use FFT estimates directly)
    #[arg(long)]
    disable_frequency_tracking: bool,

    /// Disable IF AGC in both detection and audio pipelines (AGC enabled by default)
    #[arg(long)]
    disable_if_agc: bool,

    /// Disable squelch analysis and generate signals from all candidates regardless of audio quality
    #[arg(long)]
    disable_squelch: bool,

    #[arg(long, default_value_t = 3)]
    duration: u64,

    /// Frequency tracking method (pll, spectral, correlation)
    #[arg(long, default_value = "pll")]
    frequency_tracking: String,

    #[arg(long, help = "SDR gain in dB (0 to 48 for SDRplay, default 24)")]
    gain: Option<f64>,

    /// Output format: JSON
    #[arg(long, group = "output_format")]
    json: bool,

    /// Duration in seconds for squelch to analyze audio vs noise
    #[arg(long, default_value_t = 2.0)]
    learning_duration: f32,

    /// Output format: structured logging
    #[arg(long, group = "output_format")]
    log: bool,

    /// Path to pre-trained model file (if not specified, auto-discovers latest)
    #[arg(long)]
    model_path: Option<String>,

    #[arg(long)]
    peak_scan_duration: Option<f64>,

    #[arg(long)]
    print_candidates: bool,

    /// Maximum number of scanning windows to process (default: all windows)
    #[arg(long)]
    scanning_windows: Option<usize>,

    /// Minimum spectral score threshold for candidate creation (0.0-1.0)
    #[arg(long, default_value_t = 0.2)]
    spectral_threshold: f32,

    /// Audio quality threshold for squelch ("static", "no-audio", "poor", "moderate", "good")
    /// Signals below this threshold will be filtered out. Default: "moderate"
    #[arg(long, default_value = "moderate")]
    squelch_threshold: String,

    #[arg(long)]
    stations: Option<String>,

    /// Output format: plain text (default)
    #[arg(long, group = "output_format")]
    text: bool,

    /// Required accuracy for frequency tracking convergence (Hz)
    #[arg(long, default_value_t = 5000.0)]
    tracking_accuracy: f64,

    #[arg(long)]
    verbose: bool,

    /// Window overlap percentage for band scanning (0.0-1.0, where 0.75 = 75% overlap)
    #[arg(long, default_value_t = 0.75)]
    window_overlap: f64,
}

#[derive(Parser, Debug)]
struct TrainArgs {
    /// Model version string
    #[arg(long, default_value = "0.1.0")]
    model_version: String,

    /// Output path for trained model (if not specified, auto-generates versioned filename)
    #[arg(long)]
    output_model: Option<String>,

    /// Sample rate for feature extraction
    #[arg(long, default_value_t = 48000.0)]
    sample_rate: f32,

    /// Directory containing training audio files
    #[arg(long, default_value = "tests/data/audio/quality")]
    training_data_dir: String,

    #[arg(long)]
    verbose: bool,
}

fn parse_squelch_threshold(threshold_str: &str) -> Result<AudioQuality> {
    match threshold_str.to_lowercase().as_str() {
        "static" => Ok(AudioQuality::Static),
        "no-audio" => Ok(AudioQuality::NoAudio),
        "poor" => Ok(AudioQuality::Poor),
        "moderate" => Ok(AudioQuality::Moderate),
        "good" => Ok(AudioQuality::Good),
        _ => Err(scanner::types::ScannerError::Custom(format!(
            "Invalid squelch threshold '{}'. Valid values: static, no-audio, poor, moderate, good",
            threshold_str
        ))),
    }
}

fn create_audio_analyzer_for_scan(
    classifier_type: AudioClassifier,
    sample_rate: f32,
    model_path: Option<&str>,
) -> Result<AudioAnalyzer> {
    let classifier: Box<dyn scanner::audio_quality::Classifier> = match classifier_type {
        AudioClassifier::Heuristic1 => Box::new(
            scanner::audio_quality::heuristic1::Classifier::new(sample_rate),
        ),
        AudioClassifier::Heuristic2 => Box::new(
            scanner::audio_quality::heuristic2::Classifier::new(sample_rate),
        ),
        AudioClassifier::Heuristic3 => Box::new(
            scanner::audio_quality::heuristic3::Classifier::new(sample_rate),
        ),
        AudioClassifier::RandomForest => {
            // Discover model path if not provided
            let discovered_path = model_path
                .map(|s| s.to_string())
                .or_else(discover_latest_model);

            match discovered_path {
                Some(path) => {
                    tracing::debug!(model_path = %path, "Attempting to load Random Forest model");
                    match scanner::audio_quality::random_forest::Classifier::load_pretrained(&path)
                    {
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
                            Box::new(scanner::audio_quality::heuristic1::Classifier::new(
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
                    Box::new(scanner::audio_quality::heuristic1::Classifier::new(
                        sample_rate,
                    ))
                }
            }
        }
    };
    Ok(AudioAnalyzer::new(classifier))
}

fn handle_scan_command(args: ScanArgs) -> Result<()> {
    // Parse squelch threshold, with --disable-squelch overriding to "static"
    let squelch_threshold = if args.disable_squelch {
        AudioQuality::Static
    } else {
        parse_squelch_threshold(&args.squelch_threshold)?
    };

    // Always create a real audio analyzer - threshold filtering happens in squelch logic
    let audio_analyzer = create_audio_analyzer_for_scan(
        args.audio_classifier.clone(),
        48000.0,
        args.model_path.as_deref(),
    )?;

    tracing::debug!(
        classifier = audio_analyzer.classifier_name(),
        squelch_threshold = format!("{:?}", squelch_threshold),
        "Audio analyzer initialized"
    );

    // Determine format from flags (default to Text if none specified)
    let format = if args.json {
        Format::Json
    } else if args.log {
        Format::Log
    } else {
        Format::Text // Default when no format specified or when --text is used
    };

    let config = ScanningConfig {
        audio_buffer_size: 8192, // Increased from 4K to 8K samples for better buffering
        audio_sample_rate: 48000,
        band: args.band,
        capture_audio_duration: args.audio_capture_duration,
        capture_audio: args.audio_capture_dir,
        capture_duration: args.capture_duration,
        capture_iq: args.capture_iq,
        debug_pipeline: args.debug_pipeline,
        duration: args.duration,
        sdr_gain: args.gain.unwrap_or(24.0), // Default to middle of 0-48 dB range
        scanning_windows: args.scanning_windows,
        fft_size: 1024,
        peak_detection_threshold: 1.0,
        peak_scan_duration: args.peak_scan_duration,
        print_candidates: args.print_candidates,
        samp_rate: 2_000_000.0f64,
        squelch_learning_duration: args.learning_duration,

        // Frequency tracking configuration
        frequency_tracking_method: args.frequency_tracking,
        tracking_accuracy: args.tracking_accuracy,
        disable_frequency_tracking: args.disable_frequency_tracking,

        // Spectral analysis configuration
        spectral_threshold: args.spectral_threshold,

        // AGC and window configuration
        agc_settling_time: args.agc_settling_time,
        window_overlap: args.window_overlap,
        // Squelch configuration
        disable_squelch: args.disable_squelch,
        squelch_threshold,
        // IF AGC configuration
        disable_if_agc: args.disable_if_agc,
        // Audio analyzer
        audio_analyzer,
    };

    let driver = args.device_args.as_deref().unwrap_or(DEFAULT_DRIVER);
    let console_writer = Arc::new(DefaultConsoleWriter);
    let logger = Arc::new(DefaultLogger::new(args.verbose, format));

    // Initialize logging before SDR operations to suppress library messages
    scanner::logging::init(logger.as_ref(), args.verbose)?;

    // Enumerate devices and serialize them to strings for later re-instantiation
    let device_strings = soapysdr::enumerate(driver)?
        .into_iter()
        .map(|args| soapy::Device(format!("{}", args)))
        .collect::<Vec<soapy::Device>>();

    // Create and setup MainThread with device strings
    MainThread::new(config, console_writer, logger, device_strings)?.run(args.stations)
}

fn generate_versioned_filename(model_version: &str) -> String {
    let date = chrono::Utc::now().format("%Y%m%d").to_string();
    format!("models/audio_quality_rf_v{}_{}.bin", model_version, date)
}

fn discover_latest_model() -> Option<String> {
    use std::fs;
    use std::path::Path;

    let models_dir = Path::new("models");
    if !models_dir.exists() {
        return None;
    }

    let mut versioned_models = Vec::new();

    // Scan for versioned model files
    if let Ok(entries) = fs::read_dir(models_dir) {
        for entry in entries.flatten() {
            let filename = entry.file_name().to_string_lossy().to_string();

            // Match pattern: audio_quality_rf_v{version}_{date}.bin
            if filename.starts_with("audio_quality_rf_v") && filename.ends_with(".bin") {
                // Extract version and date parts
                if let Some(version_part) = filename.strip_prefix("audio_quality_rf_v")
                    && let Some(base) = version_part.strip_suffix(".bin")
                {
                    // Split on last underscore to separate version from date
                    if let Some(last_underscore) = base.rfind('_') {
                        let version = &base[..last_underscore];
                        let date = &base[last_underscore + 1..];

                        versioned_models.push((
                            version.to_string(),
                            date.to_string(),
                            entry.path().to_string_lossy().to_string(),
                        ));
                    }
                }
            }
        }
    }

    if !versioned_models.is_empty() {
        // Sort by version (semantic) then by date (newest first)
        versioned_models.sort_by(|a, b| {
            // First compare by version using simple string comparison (works for semantic versioning)
            match b.0.cmp(&a.0) {
                std::cmp::Ordering::Equal => b.1.cmp(&a.1), // If versions equal, sort by date (newest first)
                other => other,
            }
        });

        return Some(versioned_models[0].2.clone());
    }

    // Fall back to legacy filename
    let legacy_path = "models/audio_quality_rf.bin";
    if Path::new(legacy_path).exists() {
        Some(legacy_path.to_string())
    } else {
        None
    }
}

fn handle_train_command(args: TrainArgs) -> Result<()> {
    use std::fs;
    use std::path::Path;

    // Initialize logging for train command
    let logger = std::sync::Arc::new(scanner::logging::DefaultLogger::new(
        args.verbose,
        scanner::types::Format::Text,
    ));
    scanner::logging::init(logger.as_ref(), args.verbose)?;

    // Generate output filename if not provided
    let output_model = args
        .output_model
        .unwrap_or_else(|| generate_versioned_filename(&args.model_version));

    tracing::debug!("Training Random Forest audio quality classifier");
    tracing::debug!(training_data_dir = %args.training_data_dir, "Using training data directory");
    tracing::debug!(output_model = %output_model, "Output model path");

    // Create output directory if it doesn't exist
    if let Some(parent) = Path::new(&output_model).parent() {
        fs::create_dir_all(parent)?;
    }

    // Create and train the classifier
    let mut classifier = scanner::audio_quality::random_forest::Classifier::new(args.sample_rate);

    tracing::debug!("Loading training data and extracting features");
    classifier.train()?;

    tracing::debug!(output_model = %output_model, "Saving trained model");
    classifier.save_model(&output_model, &args.model_version)?;

    tracing::debug!("Training complete! Model saved successfully");
    tracing::debug!(
        model_path = %output_model,
        "Model ready to use with: scanner scan --model-path {}",
        output_model
    );

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Scan(args) => handle_scan_command(*args),
        Commands::Train(args) => handle_train_command(args),
    }
}
