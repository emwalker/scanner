//! Performance regression testing module

use super::benchmark_datasets::BenchmarkDataset;
use crate::core::types::Result;
use tracing::debug;

/// Performance requirements for regression testing
#[derive(Debug, Clone)]
pub struct PerformanceRequirements {
    pub max_processing_time_ms: u64,
    pub max_memory_usage_mb: u64,
    pub max_variance_percent: f64,
    pub min_throughput_samples_per_sec: f64,
}

impl Default for PerformanceRequirements {
    fn default() -> Self {
        Self {
            max_processing_time_ms: 200, // 200ms max for typical scenarios
            max_memory_usage_mb: 100,    // 100MB max memory usage
            max_variance_percent: 20.0,  // 20% variance tolerance
            min_throughput_samples_per_sec: 10_000_000.0, // 10M samples/sec minimum
        }
    }
}

/// Run performance regression tests to ensure optimizations don't degrade performance
pub fn run_performance_regression_tests(
    config: &crate::core::types::ScanningConfig,
    requirements: &PerformanceRequirements,
    num_runs: usize,
) -> Result<PerformanceRegressionResult> {
    debug!(
        "Running performance regression tests with {} runs",
        num_runs
    );

    let mut performance_measurements = Vec::new();

    // Test all benchmark datasets for comprehensive performance coverage
    for dataset in BenchmarkDataset::all_datasets() {
        debug!("Testing performance with dataset: {}", dataset.name);

        let mut dataset_measurements = Vec::new();

        for run in 0..num_runs {
            let measurement = measure_peak_detection_performance(
                config,
                &dataset,
                &format!("{}_{}", dataset.name, run),
            )?;

            dataset_measurements.push(measurement);
        }

        performance_measurements.push(DatasetPerformanceResult {
            dataset_name: dataset.name.clone(),
            measurements: dataset_measurements,
        });
    }

    // Analyze results against requirements
    let analysis = analyze_performance_regression(&performance_measurements, requirements);

    Ok(PerformanceRegressionResult {
        requirements: requirements.clone(),
        dataset_results: performance_measurements,
        analysis,
        num_runs,
    })
}

/// Measure performance of peak detection for a specific dataset
fn measure_peak_detection_performance(
    config: &crate::core::types::ScanningConfig,
    dataset: &BenchmarkDataset,
    test_name: &str,
) -> Result<PerformanceMeasurement> {
    let mut generator = dataset.generate_signals();
    let expected_peaks = dataset.expected_frequencies();

    let start_time = std::time::Instant::now();
    let start_memory = memory_usage_mb();

    // Perform peak detection
    let peaks = crate::signal::peaks::collect_peaks_from_source(config, &mut generator)?;

    let processing_time = start_time.elapsed();
    let end_memory = memory_usage_mb();
    let memory_used = end_memory.saturating_sub(start_memory);

    // Calculate throughput
    let total_samples = (dataset.sample_rate * dataset.duration_seconds) as u64;
    let throughput = total_samples as f64 / processing_time.as_secs_f64();

    // Calculate accuracy metrics
    let tolerance_hz = 50_000.0;
    let detected_count = expected_peaks
        .iter()
        .filter(|&&expected| {
            peaks
                .iter()
                .any(|peak| (peak.frequency_hz - expected).abs() <= tolerance_hz)
        })
        .count();

    let detection_accuracy = detected_count as f64 / expected_peaks.len() as f64;

    Ok(PerformanceMeasurement {
        test_name: test_name.to_string(),
        processing_time,
        memory_used_mb: memory_used,
        throughput_samples_per_sec: throughput,
        peaks_detected: peaks.len(),
        expected_peaks: expected_peaks.len(),
        detection_accuracy,
        dataset_name: dataset.name.clone(),
    })
}

/// Analyze performance measurements against requirements
fn analyze_performance_regression(
    measurements: &[DatasetPerformanceResult],
    requirements: &PerformanceRequirements,
) -> PerformanceAnalysis {
    let mut all_measurements = Vec::new();
    let mut violations = Vec::new();

    for dataset_result in measurements {
        for measurement in &dataset_result.measurements {
            all_measurements.push(measurement);

            // Check individual requirements
            if measurement.processing_time.as_millis() as u64 > requirements.max_processing_time_ms
            {
                violations.push(format!(
                    "{}: Processing time {}ms exceeds limit {}ms",
                    measurement.test_name,
                    measurement.processing_time.as_millis(),
                    requirements.max_processing_time_ms
                ));
            }

            if measurement.memory_used_mb > requirements.max_memory_usage_mb {
                violations.push(format!(
                    "{}: Memory usage {}MB exceeds limit {}MB",
                    measurement.test_name,
                    measurement.memory_used_mb,
                    requirements.max_memory_usage_mb
                ));
            }

            if measurement.throughput_samples_per_sec < requirements.min_throughput_samples_per_sec
            {
                violations.push(format!(
                    "{}: Throughput {:.0} samples/sec below minimum {:.0}",
                    measurement.test_name,
                    measurement.throughput_samples_per_sec,
                    requirements.min_throughput_samples_per_sec
                ));
            }
        }
    }

    // Calculate overall statistics
    let processing_times: Vec<f64> = all_measurements
        .iter()
        .map(|m| m.processing_time.as_secs_f64() * 1000.0)
        .collect();

    let mean_processing_time = processing_times.iter().sum::<f64>() / processing_times.len() as f64;
    let processing_variance = if processing_times.len() > 1 {
        let variance = processing_times
            .iter()
            .map(|&x| (x - mean_processing_time).powi(2))
            .sum::<f64>()
            / (processing_times.len() - 1) as f64;
        variance.sqrt() / mean_processing_time * 100.0 // Coefficient of variation
    } else {
        0.0
    };

    if processing_variance > requirements.max_variance_percent {
        violations.push(format!(
            "Performance variance {:.1}% exceeds limit {:.1}%",
            processing_variance, requirements.max_variance_percent
        ));
    }

    let mean_accuracy = all_measurements
        .iter()
        .map(|m| m.detection_accuracy)
        .sum::<f64>()
        / all_measurements.len() as f64;

    let mean_throughput = all_measurements
        .iter()
        .map(|m| m.throughput_samples_per_sec)
        .sum::<f64>()
        / all_measurements.len() as f64;

    PerformanceAnalysis {
        total_tests: all_measurements.len(),
        mean_processing_time_ms: mean_processing_time,
        processing_variance_percent: processing_variance,
        mean_detection_accuracy: mean_accuracy,
        mean_throughput_samples_per_sec: mean_throughput,
        passed_all_requirements: violations.is_empty(),
        requirement_violations: violations,
    }
}

/// Get current memory usage in MB (simplified implementation)
fn memory_usage_mb() -> u64 {
    // This is a simplified implementation. In production, you would use
    // a proper memory profiling library like `memory-stats` or `procfs`
    0 // Placeholder - actual implementation would read from /proc/self/status
}

/// Validate that current performance meets baseline expectations
pub fn validate_performance_baseline(config: &crate::core::types::ScanningConfig) -> Result<bool> {
    let requirements = PerformanceRequirements::default();
    let result = run_performance_regression_tests(config, &requirements, 3)?;

    if !result.analysis.passed_all_requirements {
        debug!("Performance baseline validation failed:");
        for violation in &result.analysis.requirement_violations {
            debug!("  - {}", violation);
        }
        return Ok(false);
    }

    debug!(
        "Performance baseline validation passed: {:.1}ms avg processing time, {:.1}% accuracy",
        result.analysis.mean_processing_time_ms,
        result.analysis.mean_detection_accuracy * 100.0
    );

    Ok(true)
}

#[derive(Debug)]
pub struct PerformanceRegressionResult {
    pub requirements: PerformanceRequirements,
    pub dataset_results: Vec<DatasetPerformanceResult>,
    pub analysis: PerformanceAnalysis,
    pub num_runs: usize,
}

#[derive(Debug)]
pub struct DatasetPerformanceResult {
    pub dataset_name: String,
    pub measurements: Vec<PerformanceMeasurement>,
}

#[derive(Debug)]
pub struct PerformanceMeasurement {
    pub test_name: String,
    pub processing_time: std::time::Duration,
    pub memory_used_mb: u64,
    pub throughput_samples_per_sec: f64,
    pub peaks_detected: usize,
    pub expected_peaks: usize,
    pub detection_accuracy: f64,
    pub dataset_name: String,
}

#[derive(Debug)]
pub struct PerformanceAnalysis {
    pub total_tests: usize,
    pub mean_processing_time_ms: f64,
    pub processing_variance_percent: f64,
    pub mean_detection_accuracy: f64,
    pub mean_throughput_samples_per_sec: f64,
    pub requirement_violations: Vec<String>,
    pub passed_all_requirements: bool,
}
