//! Variance measurement tools for peak detection consistency testing

/// Variance measurement helper for peak detection consistency testing
pub struct VarianceMeasurement {
    pub test_name: String,
    pub peak_counts: Vec<usize>,
    pub peak_frequencies: Vec<Vec<f64>>,
    pub expected_peaks: Vec<f64>,
}

impl VarianceMeasurement {
    pub fn new(test_name: &str, expected_peaks: Vec<f64>) -> Self {
        Self {
            test_name: test_name.to_string(),
            peak_counts: Vec::new(),
            peak_frequencies: Vec::new(),
            expected_peaks,
        }
    }

    pub fn add_measurement(&mut self, peaks: &[crate::core::types::Peak]) {
        let frequencies: Vec<f64> = peaks.iter().map(|p| p.frequency_hz).collect();
        self.peak_counts.push(peaks.len());
        self.peak_frequencies.push(frequencies);
    }

    pub fn calculate_stats(&self) -> VarianceStats {
        let mean_count = if self.peak_counts.is_empty() {
            0.0
        } else {
            self.peak_counts.iter().sum::<usize>() as f64 / self.peak_counts.len() as f64
        };

        let variance = if self.peak_counts.len() < 2 {
            0.0
        } else {
            let sum_sq_diff: f64 = self
                .peak_counts
                .iter()
                .map(|&count| {
                    let diff = count as f64 - mean_count;
                    diff * diff
                })
                .sum();
            sum_sq_diff / (self.peak_counts.len() - 1) as f64
        };

        let std_dev = variance.sqrt();

        let min_count = self.peak_counts.iter().min().copied().unwrap_or(0);
        let max_count = self.peak_counts.iter().max().copied().unwrap_or(0);
        let range = max_count.saturating_sub(min_count);

        // Calculate expected peak detection rate
        let tolerance_hz = 50_000.0; // 50 kHz tolerance
        let mut detection_rates = Vec::new();

        for expected_freq in &self.expected_peaks {
            let detections = self
                .peak_frequencies
                .iter()
                .filter(|frequencies| {
                    frequencies
                        .iter()
                        .any(|&freq| (freq - expected_freq).abs() <= tolerance_hz)
                })
                .count();
            let rate = if self.peak_frequencies.is_empty() {
                0.0
            } else {
                detections as f64 / self.peak_frequencies.len() as f64
            };
            detection_rates.push(rate);
        }

        let avg_detection_rate = if detection_rates.is_empty() {
            0.0
        } else {
            detection_rates.iter().sum::<f64>() / detection_rates.len() as f64
        };

        VarianceStats {
            test_name: self.test_name.clone(),
            num_runs: self.peak_counts.len(),
            mean_count,
            std_dev,
            variance,
            min_count,
            max_count,
            range,
            detection_rates,
            avg_detection_rate,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VarianceStats {
    pub test_name: String,
    pub num_runs: usize,
    pub mean_count: f64,
    pub std_dev: f64,
    pub variance: f64,
    pub min_count: usize,
    pub max_count: usize,
    pub range: usize,
    pub detection_rates: Vec<f64>,
    pub avg_detection_rate: f64,
}

impl VarianceStats {
    pub fn is_variance_improved(&self, baseline: &VarianceStats) -> bool {
        self.std_dev < baseline.std_dev && self.range < baseline.range
    }

    pub fn improvement_percentage(&self, baseline: &VarianceStats) -> f64 {
        if baseline.std_dev == 0.0 {
            return 0.0;
        }
        ((baseline.std_dev - self.std_dev) / baseline.std_dev) * 100.0
    }
}

/// Performance benchmark helper
pub struct PerformanceBenchmark {
    pub test_name: String,
    pub processing_times: Vec<std::time::Duration>,
}

impl PerformanceBenchmark {
    pub fn new(test_name: &str) -> Self {
        Self {
            test_name: test_name.to_string(),
            processing_times: Vec::new(),
        }
    }

    pub fn add_measurement(&mut self, duration: std::time::Duration) {
        self.processing_times.push(duration);
    }

    pub fn calculate_stats(&self) -> PerformanceStats {
        if self.processing_times.is_empty() {
            return PerformanceStats {
                test_name: self.test_name.clone(),
                num_runs: 0,
                mean_ms: 0.0,
                std_dev_ms: 0.0,
                min_ms: 0.0,
                max_ms: 0.0,
                median_ms: 0.0,
            };
        }

        let times_ms: Vec<f64> = self
            .processing_times
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .collect();

        let mean_ms = times_ms.iter().sum::<f64>() / times_ms.len() as f64;

        let variance = if times_ms.len() < 2 {
            0.0
        } else {
            let sum_sq_diff: f64 = times_ms
                .iter()
                .map(|&time| {
                    let diff = time - mean_ms;
                    diff * diff
                })
                .sum();
            sum_sq_diff / (times_ms.len() - 1) as f64
        };

        let std_dev_ms = variance.sqrt();

        let min_ms = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_ms = times_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut sorted_times = times_ms.clone();
        sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ms = if sorted_times.len().is_multiple_of(2) {
            (sorted_times[sorted_times.len() / 2 - 1] + sorted_times[sorted_times.len() / 2]) / 2.0
        } else {
            sorted_times[sorted_times.len() / 2]
        };

        PerformanceStats {
            test_name: self.test_name.clone(),
            num_runs: times_ms.len(),
            mean_ms,
            std_dev_ms,
            min_ms,
            max_ms,
            median_ms,
        }
    }
}

#[derive(Debug)]
pub struct PerformanceStats {
    pub test_name: String,
    pub num_runs: usize,
    pub mean_ms: f64,
    pub std_dev_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub median_ms: f64,
}

impl PerformanceStats {
    pub fn is_performance_acceptable(
        &self,
        baseline: &PerformanceStats,
        tolerance_percent: f64,
    ) -> bool {
        let degradation = ((self.mean_ms - baseline.mean_ms) / baseline.mean_ms) * 100.0;
        degradation <= tolerance_percent
    }
}
