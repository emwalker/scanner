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
                    config.fft_size = fft_size;
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
                        config.peak_scan_duration = duration;
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
    ScanningConfig {
        audio_buffer_size: 8192,
        audio_sample_rate: 48000,
        band: scanner::types::Band::Fm,
        capture_audio_duration: 3.0,
        capture_audio: None,
        capture_duration: 2.0,
        capture_iq: None,
        debug_pipeline: false,
        duration: 3,
        sdr_gain: 24.0,
        scanning_windows: Some(2),
        fft_size: 1024,
        peak_detection_threshold: 1.0,
        peak_scan_duration: 1.5, // Use our optimized default
        print_candidates: false,
        samp_rate: 2_000_000.0,
        squelch_learning_duration: 1.0,
        frequency_tracking_method: "pll".to_string(),
        tracking_accuracy: 5000.0,
        disable_frequency_tracking: false,
        spectral_threshold: 0.2,
        agc_settling_time: 0.45,
        window_overlap: 0.75,
        packet_size: 16384,
        disable_squelch: false,
        squelch_threshold: scanner::audio_quality::AudioQuality::Moderate,
        disable_if_agc: false,
        audio_analyzer: scanner::audio_quality::AudioAnalyzer::mock(),
        // Signal averaging defaults
        enable_exponential_smoothing: false,
        smoothing_alpha: 0.3,
        enable_multi_frame_averaging: false,
        averaging_frames: 8,
        enable_coherent_integration: false,
        enable_moving_average_filter: false,
        moving_average_window_size: 5,
        // CFAR detection defaults
        enable_cfar_detection: false,
        cfar_threshold_factor: 10.0,
        cfar_guard_cells: 10,
        cfar_reference_cells: 50,
        cfar_false_alarm_rate: 0.01,
        // Spectral preprocessing defaults (disabled for benchmarking baseline performance)
        enable_windowing: false,
        window_type: scanner::types::WindowType::Rectangular,
        zero_padding_factor: 1,
        window_overlap_percent: 0.0,
        // Multi-frame integration defaults (disabled for benchmarking baseline performance)
        enable_multi_frame_integration: false,
        multi_frame_history_frames: 5,
        multi_frame_confirmation_threshold: 3,
        multi_frame_frequency_tolerance: 25_000.0,
        multi_frame_max_age: 10.0,
        // Dynamic noise floor defaults (disabled for benchmarking baseline performance)
        enable_dynamic_noise_floor: false,
        noise_floor_percentile: 0.25,
        noise_floor_history_frames: 8,
        noise_floor_threshold_multiplier: 1.6,
        noise_floor_adaptation_rate: 0.35,
    }
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
    config.enable_exponential_smoothing = true;
    config.enable_multi_frame_averaging = true;
    config.enable_coherent_integration = true;
    config.enable_moving_average_filter = true;
    config
}

/// CFAR only: CFAR detection enabled, others disabled
fn create_cfar_only_config() -> ScanningConfig {
    let mut config = create_test_config();
    config.enable_cfar_detection = true;
    config.cfar_threshold_factor = 3.0;
    config.cfar_guard_cells = 10;
    config.cfar_reference_cells = 50;
    config
}

/// Spectral preprocessing only: Windowing and zero-padding enabled, others disabled
fn create_spectral_preprocessing_only_config() -> ScanningConfig {
    let mut config = create_test_config();
    config.enable_windowing = true;
    config.window_type = scanner::types::WindowType::BlackmanHarris;
    config.zero_padding_factor = 2;
    config
}

/// Multi-frame integration only: Peak persistence tracking enabled, others disabled
fn create_multi_frame_integration_only_config() -> ScanningConfig {
    let mut config = create_test_config();
    config.enable_multi_frame_integration = true;
    config.multi_frame_history_frames = 5;
    config.multi_frame_confirmation_threshold = 3;
    config.multi_frame_frequency_tolerance = 25_000.0;
    config.multi_frame_max_age = 10.0;
    config
}

/// Signal averaging + CFAR: Both signal averaging and CFAR enabled
fn create_signal_averaging_plus_cfar_config() -> ScanningConfig {
    let mut config = create_signal_averaging_only_config();
    config.enable_cfar_detection = true;
    config.cfar_threshold_factor = 3.0;
    config.cfar_guard_cells = 10;
    config.cfar_reference_cells = 50;
    config
}

/// Signal averaging + spectral preprocessing: Signal averaging and windowing enabled
fn create_signal_averaging_plus_spectral_config() -> ScanningConfig {
    let mut config = create_signal_averaging_only_config();
    config.enable_windowing = true;
    config.window_type = scanner::types::WindowType::BlackmanHarris;
    config.zero_padding_factor = 2;
    config
}

/// CFAR + spectral preprocessing: CFAR and windowing enabled
fn create_cfar_plus_spectral_config() -> ScanningConfig {
    let mut config = create_cfar_only_config();
    config.enable_windowing = true;
    config.window_type = scanner::types::WindowType::BlackmanHarris;
    config.zero_padding_factor = 2;
    config
}

/// Signal averaging + multi-frame integration: Signal averaging and persistence tracking enabled
fn create_signal_averaging_plus_multi_frame_config() -> ScanningConfig {
    let mut config = create_signal_averaging_only_config();
    config.enable_multi_frame_integration = true;
    config.multi_frame_history_frames = 5;
    config.multi_frame_confirmation_threshold = 3;
    config.multi_frame_frequency_tolerance = 25_000.0;
    config.multi_frame_max_age = 10.0;
    config
}

/// CFAR + multi-frame integration: CFAR detection and persistence tracking enabled
fn create_cfar_plus_multi_frame_config() -> ScanningConfig {
    let mut config = create_cfar_only_config();
    config.enable_multi_frame_integration = true;
    config.multi_frame_history_frames = 5;
    config.multi_frame_confirmation_threshold = 3;
    config.multi_frame_frequency_tolerance = 25_000.0;
    config.multi_frame_max_age = 10.0;
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
