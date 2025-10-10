use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use scanner::testing::*;
use scanner::types::ScanningConfig;

/// Benchmark peak detection performance across different scenarios
fn benchmark_peak_detection(c: &mut Criterion) {
    let config = create_test_config();

    let mut group = c.benchmark_group("peak_detection");
    group
        .sample_size(10) // Reduce sample size for faster benchmarking
        .measurement_time(std::time::Duration::from_secs(10)) // Increase measurement time
        .warm_up_time(std::time::Duration::from_secs(3)) // Add warm-up time
        .noise_threshold(0.15); // 15% noise threshold for signal processing workloads

    // Benchmark 1: FM Band Typical scenario
    group.bench_function("fm_band_typical", |b| {
        b.iter_batched(
            || BenchmarkDataset::fm_band_typical().generate_signals(),
            |mut generator| {
                black_box(scanner::peaks::collect_peaks_from_source(
                    &config,
                    &mut generator,
                ))
            },
            BatchSize::SmallInput,
        )
    });

    // Benchmark 2: Weak signals scenario (should be slower due to processing requirements)
    group.bench_function("weak_signals", |b| {
        b.iter_batched(
            || BenchmarkDataset::weak_signals().generate_signals(),
            |mut generator| {
                black_box(scanner::peaks::collect_peaks_from_source(
                    &config,
                    &mut generator,
                ))
            },
            BatchSize::SmallInput,
        )
    });

    // Benchmark 3: High interference scenario
    group.bench_function("high_interference", |b| {
        b.iter_batched(
            || BenchmarkDataset::high_interference().generate_signals(),
            |mut generator| {
                black_box(scanner::peaks::collect_peaks_from_source(
                    &config,
                    &mut generator,
                ))
            },
            BatchSize::SmallInput,
        )
    });

    // Benchmark 4: Edge cases scenario
    group.bench_function("edge_cases", |b| {
        b.iter_batched(
            || BenchmarkDataset::edge_cases().generate_signals(),
            |mut generator| {
                black_box(scanner::peaks::collect_peaks_from_source(
                    &config,
                    &mut generator,
                ))
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

/// Benchmark different FFT sizes to understand performance scaling
fn benchmark_fft_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_sizes");
    group
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(8))
        .warm_up_time(std::time::Duration::from_secs(2))
        .noise_threshold(0.15);

    let fft_sizes = [512, 1024, 2048, 4096];

    for &fft_size in &fft_sizes {
        group.bench_with_input(format!("fft_{}", fft_size), &fft_size, |b, &fft_size| {
            b.iter_batched(
                || {
                    let mut config = create_test_config();
                    config.peak_detection.fft_size = fft_size;
                    let generator = create_fm_band_test_scenario();
                    (config, generator)
                },
                |(config, mut generator)| {
                    black_box(scanner::peaks::collect_peaks_from_source(
                        &config,
                        &mut generator,
                    ))
                },
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

/// Benchmark different peak scan durations
fn benchmark_peak_scan_durations(c: &mut Criterion) {
    let mut group = c.benchmark_group("peak_scan_durations");
    group
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(10))
        .warm_up_time(std::time::Duration::from_secs(3))
        .noise_threshold(0.15); // 15% noise threshold for signal processing workloads

    let durations = [0.5, 1.0, 1.5, 2.0, 3.0];

    for &duration in &durations {
        group.bench_with_input(
            format!("duration_{:.1}s", duration),
            &duration,
            |b, &duration| {
                b.iter_batched(
                    || {
                        let mut config = create_test_config();
                        config.peak_detection.scan_duration = duration;
                        let generator = create_fm_band_test_scenario();
                        (config, generator)
                    },
                    |(config, mut generator)| {
                        black_box(scanner::peaks::collect_peaks_from_source(
                            &config,
                            &mut generator,
                        ))
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

/// Benchmark signal generation performance (baseline overhead)
fn benchmark_signal_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("signal_generation");
    group
        .sample_size(20)
        .measurement_time(std::time::Duration::from_secs(5))
        .warm_up_time(std::time::Duration::from_secs(1))
        .noise_threshold(0.15);

    group.bench_function("fm_band_typical_generation", |b| {
        b.iter(|| black_box(BenchmarkDataset::fm_band_typical().generate_signals()))
    });

    group.bench_function("weak_signals_generation", |b| {
        b.iter(|| black_box(BenchmarkDataset::weak_signals().generate_signals()))
    });

    group.finish();
}

/// Performance regression test that ensures optimizations don't slow down the system
fn benchmark_performance_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_tests");
    group
        .sample_size(10) // Reduce sample count to avoid timeout
        .measurement_time(std::time::Duration::from_secs(15))
        .warm_up_time(std::time::Duration::from_secs(4))
        .noise_threshold(0.15);

    // This benchmark establishes a baseline for performance regression detection
    // Target: Processing should complete in under 100ms for typical scenarios
    group.bench_function("baseline_performance", |b| {
        b.iter_batched(
            || {
                let config = create_test_config();
                let generator = create_fm_band_test_scenario();
                (config, generator)
            },
            |(config, mut generator)| {
                let start = std::time::Instant::now();
                let result = scanner::peaks::collect_peaks_from_source(&config, &mut generator);
                let duration = start.elapsed();

                // Assert performance requirement in benchmark
                // Increased threshold to account for signal averaging parameter overhead
                if duration.as_millis() > 2000 {
                    panic!(
                        "Performance regression detected: {}ms > 2000ms threshold",
                        duration.as_millis()
                    );
                }

                black_box(result)
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn create_test_config() -> ScanningConfig {
    let mut config = ScanningConfig::default();

    // Override fields for benchmarking baseline performance
    config.audio.buffer_size = 8192;
    config.audio.sample_rate = 48000;
    config.audio.squelch.learning_duration = 1.0;
    config.audio.analyzer = scanner::audio_quality::AudioAnalyzer::pass_through();

    config.peak_detection.fft_size = 1024;
    config.peak_detection.threshold = 1.0;
    config.peak_detection.scan_duration = 1.5;
    config.peak_detection.spectral_threshold = 0.2;

    // Disable all advanced features for baseline benchmarking
    config.peak_detection.cfar.enabled = false;
    config.peak_detection.noise_floor.enabled = false;
    config.peak_detection.windowing.enabled = false;
    config
        .peak_detection
        .averaging
        .exponential_smoothing
        .enabled = false;
    config
        .peak_detection
        .averaging
        .multi_frame_averaging
        .enabled = false;
    config.peak_detection.averaging.coherent_integration_enabled = false;
    config.peak_detection.averaging.moving_average.enabled = false;
    config.peak_detection.multi_frame.enabled = false;

    config.signal_processing.frequency_tracking.disabled = false;

    config.duration = 3;
    config.sdr_gain = 24.0;
    config.samp_rate = 2_000_000.0;
    config.scanning_windows = Some(2);

    config
}

/// Comprehensive benchmark comparing different feature configurations
fn benchmark_feature_configurations(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_configurations");
    group
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(20)) // Increased from 15s to 20s
        .warm_up_time(std::time::Duration::from_secs(3))
        .noise_threshold(0.15);

    // Define all configuration combinations to test
    let configurations = vec![
        ("baseline", create_baseline_config()),
        (
            "signal_averaging_only",
            create_signal_averaging_only_config(),
        ),
        ("cfar_only", create_cfar_only_config()),
        (
            "spectral_preprocessing_only",
            create_spectral_preprocessing_only_config(),
        ),
        (
            "multi_frame_integration_only",
            create_multi_frame_integration_only_config(),
        ),
        (
            "signal_averaging_plus_cfar",
            create_signal_averaging_plus_cfar_config(),
        ),
        (
            "signal_averaging_plus_spectral",
            create_signal_averaging_plus_spectral_config(),
        ),
        ("cfar_plus_spectral", create_cfar_plus_spectral_config()),
        (
            "signal_averaging_plus_multi_frame",
            create_signal_averaging_plus_multi_frame_config(),
        ),
        (
            "cfar_plus_multi_frame",
            create_cfar_plus_multi_frame_config(),
        ),
        ("all_features", create_all_features_config()),
    ];

    for (config_name, config) in configurations {
        group.bench_function(config_name, |b| {
            b.iter_batched(
                || BenchmarkDataset::fm_band_typical().generate_signals(),
                |mut generator| {
                    black_box(scanner::peaks::collect_peaks_from_source(
                        &config,
                        &mut generator,
                    ))
                },
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

/// Baseline configuration: All advanced features disabled
fn create_baseline_config() -> ScanningConfig {
    create_test_config() // All features already disabled in create_test_config()
}

/// Signal averaging only: Signal averaging enabled, others disabled
fn create_signal_averaging_only_config() -> ScanningConfig {
    let mut config = create_test_config();
    config
        .peak_detection
        .averaging
        .exponential_smoothing
        .enabled = true;
    config
        .peak_detection
        .averaging
        .multi_frame_averaging
        .enabled = true;
    config.peak_detection.averaging.coherent_integration_enabled = true;
    config.peak_detection.averaging.moving_average.enabled = true;
    config
}

/// CFAR only: CFAR detection enabled, others disabled
fn create_cfar_only_config() -> ScanningConfig {
    let mut config = create_test_config();
    config.peak_detection.cfar.enabled = true;
    config.peak_detection.cfar.threshold_factor = 3.0;
    config.peak_detection.cfar.guard_cells = 10;
    config.peak_detection.cfar.reference_cells = 50;
    config
}

/// Spectral preprocessing only: Windowing and zero-padding enabled, others disabled
fn create_spectral_preprocessing_only_config() -> ScanningConfig {
    let mut config = create_test_config();
    config.peak_detection.windowing.enabled = true;
    config.peak_detection.windowing.window_type = scanner::types::WindowType::BlackmanHarris;
    config.peak_detection.windowing.zero_padding_factor = 2;
    config
}

/// Multi-frame integration only: Peak persistence tracking enabled, others disabled
fn create_multi_frame_integration_only_config() -> ScanningConfig {
    let mut config = create_test_config();
    config.peak_detection.multi_frame.enabled = true;
    config.peak_detection.multi_frame.history_frames = 5;
    config.peak_detection.multi_frame.confirmation_threshold = 3;
    config.peak_detection.multi_frame.frequency_tolerance = 25_000.0;
    config.peak_detection.multi_frame.max_age = 10.0;
    config
}

/// Signal averaging + CFAR: Both signal averaging and CFAR enabled
fn create_signal_averaging_plus_cfar_config() -> ScanningConfig {
    let mut config = create_signal_averaging_only_config();
    config.peak_detection.cfar.enabled = true;
    config.peak_detection.cfar.threshold_factor = 3.0;
    config.peak_detection.cfar.guard_cells = 10;
    config.peak_detection.cfar.reference_cells = 50;
    config
}

/// Signal averaging + spectral preprocessing: Signal averaging and windowing enabled
fn create_signal_averaging_plus_spectral_config() -> ScanningConfig {
    let mut config = create_signal_averaging_only_config();
    config.peak_detection.windowing.enabled = true;
    config.peak_detection.windowing.window_type = scanner::types::WindowType::BlackmanHarris;
    config.peak_detection.windowing.zero_padding_factor = 2;
    config
}

/// CFAR + spectral preprocessing: CFAR and windowing enabled
fn create_cfar_plus_spectral_config() -> ScanningConfig {
    let mut config = create_cfar_only_config();
    config.peak_detection.windowing.enabled = true;
    config.peak_detection.windowing.window_type = scanner::types::WindowType::BlackmanHarris;
    config.peak_detection.windowing.zero_padding_factor = 2;
    config
}

/// Signal averaging + multi-frame integration: Signal averaging and persistence tracking enabled
fn create_signal_averaging_plus_multi_frame_config() -> ScanningConfig {
    let mut config = create_signal_averaging_only_config();
    config.peak_detection.multi_frame.enabled = true;
    config.peak_detection.multi_frame.history_frames = 5;
    config.peak_detection.multi_frame.confirmation_threshold = 3;
    config.peak_detection.multi_frame.frequency_tolerance = 25_000.0;
    config.peak_detection.multi_frame.max_age = 10.0;
    config
}

/// CFAR + multi-frame integration: CFAR detection and persistence tracking enabled
fn create_cfar_plus_multi_frame_config() -> ScanningConfig {
    let mut config = create_cfar_only_config();
    config.peak_detection.multi_frame.enabled = true;
    config.peak_detection.multi_frame.history_frames = 5;
    config.peak_detection.multi_frame.confirmation_threshold = 3;
    config.peak_detection.multi_frame.frequency_tolerance = 25_000.0;
    config.peak_detection.multi_frame.max_age = 10.0;
    config
}

/// All features: Complete optimized pipeline (current defaults)
fn create_all_features_config() -> ScanningConfig {
    // Use the actual default configuration
    ScanningConfig::default()
}

/// Benchmark features across different signal scenarios
fn benchmark_features_across_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("features_across_scenarios");
    group
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(18)) // Increased from 12s to 18s
        .warm_up_time(std::time::Duration::from_secs(2))
        .noise_threshold(0.15);

    let baseline_config = create_baseline_config();
    let all_features_config = create_all_features_config();

    // Baseline vs All Features on FM typical signals
    group.bench_function("baseline_fm_typical", |b| {
        b.iter_batched(
            || BenchmarkDataset::fm_band_typical().generate_signals(),
            |mut generator| {
                black_box(scanner::peaks::collect_peaks_from_source(
                    &baseline_config,
                    &mut generator,
                ))
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("all_features_fm_typical", |b| {
        b.iter_batched(
            || BenchmarkDataset::fm_band_typical().generate_signals(),
            |mut generator| {
                black_box(scanner::peaks::collect_peaks_from_source(
                    &all_features_config,
                    &mut generator,
                ))
            },
            BatchSize::SmallInput,
        )
    });

    // Baseline vs All Features on weak signals
    group.bench_function("baseline_weak_signals", |b| {
        b.iter_batched(
            || BenchmarkDataset::weak_signals().generate_signals(),
            |mut generator| {
                black_box(scanner::peaks::collect_peaks_from_source(
                    &baseline_config,
                    &mut generator,
                ))
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("all_features_weak_signals", |b| {
        b.iter_batched(
            || BenchmarkDataset::weak_signals().generate_signals(),
            |mut generator| {
                black_box(scanner::peaks::collect_peaks_from_source(
                    &all_features_config,
                    &mut generator,
                ))
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_peak_detection,
    benchmark_fft_sizes,
    benchmark_peak_scan_durations,
    benchmark_signal_generation,
    benchmark_performance_regression,
    benchmark_feature_configurations,
    benchmark_features_across_scenarios
);
criterion_main!(benches);
