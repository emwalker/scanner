//! Benchmark datasets with known peak locations for testing

use tracing::debug;

use super::signal_generation::{PeakTestSignalGenerator, TestSignal};
use crate::core::types::Result;

/// Benchmark dataset containing predefined test scenarios with known peak locations
#[derive(Debug, Clone)]
pub struct BenchmarkDataset {
    pub name: String,
    pub description: String,
    pub sample_rate: f64,
    pub center_frequency: f64,
    pub expected_peaks: Vec<BenchmarkPeak>,
    pub noise_floor: f32,
    pub duration_seconds: f64,
}

#[derive(Debug, Clone)]
pub struct BenchmarkPeak {
    pub frequency_hz: f64,
    pub magnitude_db: f32,
    pub signal_type: SignalType,
    pub quality: SignalQuality,
}

#[derive(Debug, Clone)]
pub enum SignalType {
    FmStation,
    CarrierOnly,
    ModulatedCarrier,
    Interference,
}

#[derive(Debug, Clone)]
pub enum SignalQuality {
    Strong,   // >20 dB SNR
    Moderate, // 10-20 dB SNR
    Weak,     // 3-10 dB SNR
    VeryWeak, // 0-3 dB SNR
}

impl BenchmarkDataset {
    /// Create the "FM Band Typical" benchmark dataset
    pub fn fm_band_typical() -> Self {
        Self {
            name: "FM Band Typical".to_string(),
            description: "Typical FM band with strong stations at common frequencies".to_string(),
            sample_rate: 2_000_000.0,
            center_frequency: 101_000_000.0, // 101 MHz
            expected_peaks: vec![
                BenchmarkPeak {
                    frequency_hz: 88_900_000.0,
                    magnitude_db: 25.0,
                    signal_type: SignalType::FmStation,
                    quality: SignalQuality::Strong,
                },
                BenchmarkPeak {
                    frequency_hz: 95_500_000.0,
                    magnitude_db: 22.0,
                    signal_type: SignalType::FmStation,
                    quality: SignalQuality::Strong,
                },
                BenchmarkPeak {
                    frequency_hz: 101_100_000.0,
                    magnitude_db: 28.0,
                    signal_type: SignalType::FmStation,
                    quality: SignalQuality::Strong,
                },
                BenchmarkPeak {
                    frequency_hz: 107_300_000.0,
                    magnitude_db: 20.0,
                    signal_type: SignalType::FmStation,
                    quality: SignalQuality::Moderate,
                },
            ],
            noise_floor: -45.0,
            duration_seconds: 2.0,
        }
    }

    /// Create the "Weak Signals" benchmark dataset
    pub fn weak_signals() -> Self {
        Self {
            name: "Weak Signals".to_string(),
            description: "Challenging dataset with weak signals near noise floor".to_string(),
            sample_rate: 2_000_000.0,
            center_frequency: 95_000_000.0,
            expected_peaks: vec![
                BenchmarkPeak {
                    frequency_hz: 89_300_000.0,
                    magnitude_db: 5.0,
                    signal_type: SignalType::FmStation,
                    quality: SignalQuality::Weak,
                },
                BenchmarkPeak {
                    frequency_hz: 92_100_000.0,
                    magnitude_db: 2.0,
                    signal_type: SignalType::FmStation,
                    quality: SignalQuality::VeryWeak,
                },
                BenchmarkPeak {
                    frequency_hz: 98_700_000.0,
                    magnitude_db: 8.0,
                    signal_type: SignalType::FmStation,
                    quality: SignalQuality::Weak,
                },
            ],
            noise_floor: -42.0,
            duration_seconds: 3.0,
        }
    }

    /// Create the "High Interference" benchmark dataset
    pub fn high_interference() -> Self {
        Self {
            name: "High Interference".to_string(),
            description: "Dataset with strong stations and various interference patterns"
                .to_string(),
            sample_rate: 2_000_000.0,
            center_frequency: 99_000_000.0,
            expected_peaks: vec![
                BenchmarkPeak {
                    frequency_hz: 96_500_000.0,
                    magnitude_db: 25.0,
                    signal_type: SignalType::FmStation,
                    quality: SignalQuality::Strong,
                },
                BenchmarkPeak {
                    frequency_hz: 99_100_000.0,
                    magnitude_db: 30.0,
                    signal_type: SignalType::FmStation,
                    quality: SignalQuality::Strong,
                },
                BenchmarkPeak {
                    frequency_hz: 97_800_000.0,
                    magnitude_db: 15.0,
                    signal_type: SignalType::Interference,
                    quality: SignalQuality::Moderate,
                },
                BenchmarkPeak {
                    frequency_hz: 100_200_000.0,
                    magnitude_db: 12.0,
                    signal_type: SignalType::CarrierOnly,
                    quality: SignalQuality::Moderate,
                },
            ],
            noise_floor: -40.0,
            duration_seconds: 2.5,
        }
    }

    /// Create the "Edge Cases" benchmark dataset
    pub fn edge_cases() -> Self {
        Self {
            name: "Edge Cases".to_string(),
            description: "Edge cases including band edges and closely spaced signals".to_string(),
            sample_rate: 2_000_000.0,
            center_frequency: 90_000_000.0,
            expected_peaks: vec![
                BenchmarkPeak {
                    frequency_hz: 88_100_000.0, // Near band edge
                    magnitude_db: 18.0,
                    signal_type: SignalType::FmStation,
                    quality: SignalQuality::Moderate,
                },
                BenchmarkPeak {
                    frequency_hz: 91_100_000.0, // Closely spaced pair
                    magnitude_db: 22.0,
                    signal_type: SignalType::FmStation,
                    quality: SignalQuality::Strong,
                },
                BenchmarkPeak {
                    frequency_hz: 91_300_000.0, // Closely spaced pair
                    magnitude_db: 20.0,
                    signal_type: SignalType::FmStation,
                    quality: SignalQuality::Strong,
                },
                BenchmarkPeak {
                    frequency_hz: 107_900_000.0, // Near upper band edge
                    magnitude_db: 16.0,
                    signal_type: SignalType::FmStation,
                    quality: SignalQuality::Moderate,
                },
            ],
            noise_floor: -43.0,
            duration_seconds: 2.0,
        }
    }

    /// Generate test signals matching this benchmark dataset
    pub fn generate_signals(&self) -> PeakTestSignalGenerator {
        let mut generator = PeakTestSignalGenerator::new(
            self.sample_rate,
            self.center_frequency,
            (self.sample_rate * self.duration_seconds) as usize,
            10.0_f32.powf(self.noise_floor / 20.0), // Convert dB to linear
        );

        for peak in &self.expected_peaks {
            let _frequency_offset = peak.frequency_hz - self.center_frequency;
            let amplitude = 10.0_f32.powf(peak.magnitude_db / 20.0); // Convert dB to linear

            generator.add_signal(TestSignal::new(
                peak.frequency_hz,
                amplitude,
                &format!("{:?}_{:.1}MHz", peak.signal_type, peak.frequency_hz / 1e6),
            ));
        }

        generator
    }

    /// Get expected peak frequencies for validation
    pub fn expected_frequencies(&self) -> Vec<f64> {
        self.expected_peaks.iter().map(|p| p.frequency_hz).collect()
    }

    /// Get all predefined benchmark datasets
    pub fn all_datasets() -> Vec<Self> {
        vec![
            Self::fm_band_typical(),
            Self::weak_signals(),
            Self::high_interference(),
            Self::edge_cases(),
        ]
    }
}

/// Run peak detection tests against all benchmark datasets
pub fn test_peak_detection_against_benchmarks(
    config: &crate::core::types::ScanningConfig,
) -> Result<Vec<BenchmarkTestResult>> {
    let mut results = Vec::new();

    for dataset in BenchmarkDataset::all_datasets() {
        debug!("Testing peak detection against benchmark: {}", dataset.name);

        let mut generator = dataset.generate_signals();
        let expected_peaks = dataset.expected_frequencies();

        let peaks = crate::signal::peaks::collect_peaks_from_source(config, &mut generator)?;

        let tolerance_hz = 50_000.0; // 50 kHz tolerance
        let mut detected_count = 0;
        let mut false_positives = 0;
        let mut detection_details = Vec::new();

        // Check how many expected peaks were detected
        for expected_freq in &expected_peaks {
            let detected = peaks
                .iter()
                .any(|peak| (peak.frequency_hz - expected_freq).abs() <= tolerance_hz);

            if detected {
                detected_count += 1;
            }

            detection_details.push(BenchmarkDetectionDetail {
                expected_frequency: *expected_freq,
                detected,
                closest_peak_distance: peaks
                    .iter()
                    .map(|p| (p.frequency_hz - expected_freq).abs())
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(f64::INFINITY),
            });
        }

        // Count false positives (detected peaks not near any expected frequency)
        for peak in &peaks {
            let near_expected = expected_peaks
                .iter()
                .any(|&expected| (peak.frequency_hz - expected).abs() <= tolerance_hz);
            if !near_expected {
                false_positives += 1;
            }
        }

        let detection_rate = detected_count as f64 / expected_peaks.len() as f64;
        let false_positive_rate = false_positives as f64 / peaks.len() as f64;

        results.push(BenchmarkTestResult {
            dataset_name: dataset.name.clone(),
            expected_count: expected_peaks.len(),
            detected_count,
            total_peaks_found: peaks.len(),
            detection_rate,
            false_positive_count: false_positives,
            false_positive_rate,
            detection_details,
        });

        debug!(
            "Benchmark {} results: {}/{} detected ({:.1}%), {} false positives ({:.1}%)",
            dataset.name,
            detected_count,
            expected_peaks.len(),
            detection_rate * 100.0,
            false_positives,
            false_positive_rate * 100.0
        );
    }

    Ok(results)
}

#[derive(Debug, Clone)]
pub struct BenchmarkTestResult {
    pub dataset_name: String,
    pub expected_count: usize,
    pub detected_count: usize,
    pub total_peaks_found: usize,
    pub detection_rate: f64,
    pub false_positive_count: usize,
    pub false_positive_rate: f64,
    pub detection_details: Vec<BenchmarkDetectionDetail>,
}

#[derive(Debug, Clone)]
pub struct BenchmarkDetectionDetail {
    pub expected_frequency: f64,
    pub detected: bool,
    pub closest_peak_distance: f64,
}
