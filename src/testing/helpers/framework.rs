use super::fixtures::load_iq_fixture;
use crate::core::types::{Format, ScanningConfig};
use tracing::debug;

/// Test helper for isolating peak detection with known I/Q signals
pub fn test_peak_detection_isolated(
    iq_file_path: &str,
    expected_peaks: &[f64],
    config: &ScanningConfig,
    debug: bool,
) -> crate::core::types::Result<TestPeakResult> {
    let (mut sample_source, metadata) = load_iq_fixture(iq_file_path)?;

    if debug {
        debug!(
            message = "Peak detection test started",
            iq_file = iq_file_path,
            sample_rate_mhz = metadata.sample_rate / 1e6,
            center_frequency_mhz = metadata.center_frequency / 1e6,
            expected_peaks_count = expected_peaks.len()
        );

        for (idx, expected_freq) in expected_peaks.iter().enumerate() {
            debug!(
                message = "Expected peak",
                peak_index = idx,
                frequency_mhz = expected_freq / 1e6
            );
        }
    }

    let peaks = crate::signal::peaks::collect_peaks_from_source(config, &mut sample_source)?;

    if debug {
        debug!(
            message = "Peak detection completed",
            peaks_found = peaks.len()
        );

        for (idx, peak) in peaks.iter().enumerate() {
            debug!(
                message = "Peak found",
                peak_index = idx,
                frequency_mhz = peak.frequency_hz / 1e6,
                magnitude = peak.magnitude
            );
        }
    }

    // Analyze peak detection accuracy
    let mut found_expected = Vec::new();
    let tolerance_hz = 50_000.0; // 50 kHz tolerance

    for expected_freq in expected_peaks {
        let found = peaks
            .iter()
            .any(|peak| (peak.frequency_hz - expected_freq).abs() <= tolerance_hz);
        found_expected.push(found);

        if debug {
            debug!(
                message = "Peak detection validation",
                expected_frequency_mhz = expected_freq / 1e6,
                found = found,
                tolerance_khz = tolerance_hz / 1e3
            );
        }
    }

    let all_expected_found = found_expected.iter().all(|&found| found);

    Ok(TestPeakResult {
        peaks,
        metadata,
        expected_found: found_expected,
        all_expected_found,
    })
}

#[derive(Debug)]
pub struct TestPeakResult {
    pub peaks: Vec<crate::core::types::Peak>,
    pub metadata: crate::file::IqFileMetadata,
    pub expected_found: Vec<bool>,
    pub all_expected_found: bool,
}

#[derive(Debug)]
pub struct FrequencyTranslationResult {
    pub center_freq: f64,
    pub tune_freq: f64,
    pub frequency_offset: f64,
    pub within_nyquist: bool,
    pub filter_bandwidth: f64,
    pub filter_cutoff: f64,
    pub translation_valid: bool,
}

#[derive(Debug)]
pub struct PipelineTestResult {
    pub peak_result: TestPeakResult,
    pub candidates: Vec<crate::core::types::Candidate>,
    pub translation_results: Vec<FrequencyTranslationResult>,
    pub target_found: bool,
    pub scanning_mode: ScanningMode,
}

#[derive(Debug, Clone)]
pub enum ScanningMode {
    Stations(f64),   // Direct station frequency
    BandWindow(f64), // Window center frequency
}

/// Test-safe logging initialization that captures logs for analysis
/// Returns a LogBuffer that can be used to retrieve captured log messages
pub fn init_test_logging(
    verbose: bool,
    format: Format,
) -> crate::core::types::Result<crate::logging::LogBuffer> {
    use tracing::Level;
    use tracing_subscriber::FmtSubscriber;

    let level = if verbose { Level::DEBUG } else { Level::INFO };
    let log_buffer = crate::logging::LogBuffer::default();

    match format {
        Format::Json => {
            let subscriber = FmtSubscriber::builder()
                .json()
                .with_max_level(level)
                .with_writer(log_buffer.clone())
                .finish();
            tracing::subscriber::set_global_default(subscriber)?;
        }
        Format::Text => {
            let subscriber = FmtSubscriber::builder()
                .with_max_level(level)
                .with_writer(log_buffer.clone())
                .without_time()
                .with_target(false)
                .with_level(false)
                .finish();
            tracing::subscriber::set_global_default(subscriber)?;
        }
        Format::Log => {
            let subscriber = FmtSubscriber::builder()
                .with_max_level(level)
                .with_writer(log_buffer.clone())
                .with_target(false)
                .finish();
            tracing::subscriber::set_global_default(subscriber)?;
        }
    }

    Ok(log_buffer)
}

/// Test helper that runs a function with captured logging and returns both result and logs
pub fn with_captured_logs<F, R>(
    verbose: bool,
    format: Format,
    test_fn: F,
) -> crate::core::types::Result<(R, String)>
where
    F: FnOnce() -> crate::core::types::Result<R>,
{
    let log_buffer = init_test_logging(verbose, format)?;
    let result = test_fn()?;
    let logs = log_buffer.into_string();
    Ok((result, logs))
}
