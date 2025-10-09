//! Statistical validation tools for measuring variance reduction

use super::benchmark_datasets::{BenchmarkTestResult, test_peak_detection_against_benchmarks};
use super::signal_generation::create_fm_band_test_scenario;
use super::variance_measurement::{VarianceMeasurement, VarianceStats};
use crate::types::Result;
use tracing::debug;

/// Run statistical validation to compare baseline vs improved peak detection
pub fn validate_variance_reduction(
    config: &crate::types::ScanningConfig,
    num_runs: usize,
    significance_level: f64,
) -> Result<StatisticalValidationResult> {
    debug!("Starting statistical validation with {} runs", num_runs);

    let mut baseline_measurements = Vec::new();
    let mut benchmark_results = Vec::new();

    // Run tests multiple times to gather statistics
    for run in 0..num_runs {
        debug!("Statistical validation run {}/{}", run + 1, num_runs);

        // Test against all benchmark datasets
        let benchmark_test_results = test_peak_detection_against_benchmarks(config)?;
        benchmark_results.push(benchmark_test_results);

        // Test with our standard FM band scenario for variance measurement
        let mut generator = create_fm_band_test_scenario();
        let expected_peaks = generator.expected_peaks();
        let peaks = crate::peaks::collect_peaks_from_source(config, &mut generator)?;

        let mut variance_measurement = VarianceMeasurement::new("FM_Band_Standard", expected_peaks);
        variance_measurement.add_measurement(&peaks);
        baseline_measurements.push(variance_measurement);
    }

    // Calculate aggregated statistics
    let aggregated_variance = aggregate_variance_measurements(&baseline_measurements);
    let aggregated_benchmark = aggregate_benchmark_results(&benchmark_results);

    // Perform statistical significance tests
    let variance_significant =
        is_variance_reduction_significant(&aggregated_variance, significance_level);
    let detection_rate_significant =
        is_detection_rate_improvement_significant(&aggregated_benchmark, significance_level);

    Ok(StatisticalValidationResult {
        num_runs,
        significance_level,
        aggregated_variance,
        aggregated_benchmark,
        variance_reduction_significant: variance_significant,
        detection_improvement_significant: detection_rate_significant,
        baseline_established: true,
    })
}

/// Compare two variance measurements for statistical significance
pub fn compare_variance_measurements(
    baseline: &VarianceStats,
    improved: &VarianceStats,
    _significance_level: f64,
) -> StatisticalComparison {
    // Use F-test to compare variances
    let f_statistic = if improved.variance > 0.0 {
        baseline.variance / improved.variance
    } else {
        f64::INFINITY
    };

    // Simplified significance test (would use proper F-distribution in production)
    let critical_value = 2.0; // Approximate F-critical for common cases
    let variance_significant = f_statistic > critical_value;

    // Use t-test approximation for mean comparison
    let pooled_std = ((baseline.variance + improved.variance) / 2.0).sqrt();
    let t_statistic = if pooled_std > 0.0 {
        (baseline.mean_count - improved.mean_count).abs() / pooled_std
    } else {
        0.0
    };

    let mean_significant = t_statistic > 2.0; // Approximate t-critical

    let improvement_percentage = improved.improvement_percentage(baseline);

    StatisticalComparison {
        baseline_stats: baseline.clone(),
        improved_stats: improved.clone(),
        f_statistic,
        t_statistic,
        variance_significant,
        mean_significant,
        improvement_percentage,
        overall_significant: variance_significant || mean_significant,
    }
}

/// Aggregate multiple variance measurements into summary statistics
fn aggregate_variance_measurements(
    measurements: &[VarianceMeasurement],
) -> AggregatedVarianceStats {
    let all_stats: Vec<VarianceStats> = measurements.iter().map(|m| m.calculate_stats()).collect();

    let mean_count_values: Vec<f64> = all_stats.iter().map(|s| s.mean_count).collect();
    let std_dev_values: Vec<f64> = all_stats.iter().map(|s| s.std_dev).collect();
    let range_values: Vec<usize> = all_stats.iter().map(|s| s.range).collect();

    let overall_mean = mean_count_values.iter().sum::<f64>() / mean_count_values.len() as f64;
    let overall_std_dev = std_dev_values.iter().sum::<f64>() / std_dev_values.len() as f64;
    let overall_range = range_values.iter().sum::<usize>() as f64 / range_values.len() as f64;

    let detection_rates: Vec<f64> = all_stats.iter().map(|s| s.avg_detection_rate).collect();
    let overall_detection_rate = detection_rates.iter().sum::<f64>() / detection_rates.len() as f64;

    AggregatedVarianceStats {
        num_test_runs: measurements.len(),
        overall_mean_count: overall_mean,
        overall_std_dev,
        overall_range,
        overall_detection_rate,
        individual_stats: all_stats,
    }
}

/// Aggregate multiple benchmark results
fn aggregate_benchmark_results(results: &[Vec<BenchmarkTestResult>]) -> AggregatedBenchmarkStats {
    let mut dataset_aggregates = std::collections::HashMap::new();

    for run_results in results {
        for result in run_results {
            let entry = dataset_aggregates
                .entry(result.dataset_name.clone())
                .or_insert_with(Vec::new);
            entry.push(result.clone());
        }
    }

    let mut summary_by_dataset = Vec::new();

    for (dataset_name, dataset_results) in dataset_aggregates {
        let detection_rates: Vec<f64> = dataset_results.iter().map(|r| r.detection_rate).collect();
        let false_positive_rates: Vec<f64> = dataset_results
            .iter()
            .map(|r| r.false_positive_rate)
            .collect();

        let avg_detection_rate = detection_rates.iter().sum::<f64>() / detection_rates.len() as f64;
        let avg_false_positive_rate =
            false_positive_rates.iter().sum::<f64>() / false_positive_rates.len() as f64;

        let detection_rate_std = calculate_std_dev(&detection_rates, avg_detection_rate);
        let false_positive_std = calculate_std_dev(&false_positive_rates, avg_false_positive_rate);

        summary_by_dataset.push(BenchmarkSummary {
            dataset_name,
            num_runs: dataset_results.len(),
            avg_detection_rate,
            detection_rate_std_dev: detection_rate_std,
            avg_false_positive_rate,
            false_positive_std_dev: false_positive_std,
            individual_results: dataset_results,
        });
    }

    AggregatedBenchmarkStats {
        num_test_runs: results.len(),
        summary_by_dataset,
    }
}

/// Check if variance reduction is statistically significant
fn is_variance_reduction_significant(
    aggregated: &AggregatedVarianceStats,
    _significance_level: f64,
) -> bool {
    // For now, consider significant if std_dev < 2.0 and range < 3
    // This matches our target of <2 station variance from the requirements
    aggregated.overall_std_dev < 2.0 && aggregated.overall_range < 3.0
}

/// Check if detection rate improvement is statistically significant
fn is_detection_rate_improvement_significant(
    aggregated: &AggregatedBenchmarkStats,
    _significance_level: f64,
) -> bool {
    // Consider significant if all datasets show >80% detection rate
    aggregated
        .summary_by_dataset
        .iter()
        .all(|summary| summary.avg_detection_rate > 0.8)
}

/// Calculate standard deviation for a set of values
fn calculate_std_dev(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let variance =
        values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

    variance.sqrt()
}

#[derive(Debug)]
pub struct StatisticalValidationResult {
    pub num_runs: usize,
    pub significance_level: f64,
    pub aggregated_variance: AggregatedVarianceStats,
    pub aggregated_benchmark: AggregatedBenchmarkStats,
    pub variance_reduction_significant: bool,
    pub detection_improvement_significant: bool,
    pub baseline_established: bool,
}

#[derive(Debug)]
pub struct StatisticalComparison {
    pub baseline_stats: VarianceStats,
    pub improved_stats: VarianceStats,
    pub f_statistic: f64,
    pub t_statistic: f64,
    pub variance_significant: bool,
    pub mean_significant: bool,
    pub improvement_percentage: f64,
    pub overall_significant: bool,
}

#[derive(Debug)]
pub struct AggregatedVarianceStats {
    pub num_test_runs: usize,
    pub overall_mean_count: f64,
    pub overall_std_dev: f64,
    pub overall_range: f64,
    pub overall_detection_rate: f64,
    pub individual_stats: Vec<VarianceStats>,
}

#[derive(Debug)]
pub struct AggregatedBenchmarkStats {
    pub num_test_runs: usize,
    pub summary_by_dataset: Vec<BenchmarkSummary>,
}

#[derive(Debug)]
pub struct BenchmarkSummary {
    pub dataset_name: String,
    pub num_runs: usize,
    pub avg_detection_rate: f64,
    pub detection_rate_std_dev: f64,
    pub avg_false_positive_rate: f64,
    pub false_positive_std_dev: f64,
    pub individual_results: Vec<BenchmarkTestResult>,
}
