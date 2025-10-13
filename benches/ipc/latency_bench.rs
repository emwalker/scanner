use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use scanner::hardware::mock::MockDevice;
use scanner::hardware::pool::{Pool, TaskPriority, TaskRequirements, TunerActivity};
use std::time::Instant;

fn bench_control_message_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipc_control_roundtrip");
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(5));

    group.bench_function("configure_and_start_roundtrip", |b| {
        let pool = Pool::new_unfiltered();
        let device = Box::new(MockDevice::new("mock", "bench001", false));
        pool.add_device(device, scanner::hardware::types::Backend::Mock);

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        b.iter(|| {
            let tuner = pool
                .try_acquire(&requirements, TunerActivity::Scanning)
                .expect("Failed to acquire tuner");

            let start = Instant::now();

            let mut graph = rustradio::graph::Graph::new();
            let _stream = tuner
                .add_source_to_graph(&mut graph, 100.0e6, 2.4e6, 30.0)
                .expect("Failed to add source");

            let elapsed = start.elapsed();

            black_box(elapsed);

            let _ = tuner.stop_stream();
            drop(tuner);
        });

        pool.shutdown();
    });

    group.finish();
}

fn bench_stop_stream_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipc_stop_roundtrip");
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(5));

    group.bench_function("stop_stream_roundtrip", |b| {
        let pool = Pool::new_unfiltered();
        let device = Box::new(MockDevice::new("mock", "bench002", false));
        pool.add_device(device, scanner::hardware::types::Backend::Mock);

        let requirements = TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 200e3,
            required_sample_rate: 2.4e6,
            priority: TaskPriority::Normal,
        };

        b.iter(|| {
            let tuner = pool
                .try_acquire(&requirements, TunerActivity::Scanning)
                .expect("Failed to acquire tuner");

            let mut graph = rustradio::graph::Graph::new();
            let _stream = tuner
                .add_source_to_graph(&mut graph, 100.0e6, 2.4e6, 30.0)
                .expect("Failed to add source");

            let start = Instant::now();
            let _ = tuner.stop_stream();
            let elapsed = start.elapsed();

            black_box(elapsed);
            drop(tuner);
        });

        pool.shutdown();
    });

    group.finish();
}

fn bench_multiple_roundtrips(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipc_multiple_roundtrips");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(20));

    for &count in &[5, 10] {
        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &count,
            |b, &count| {
                let pool = Pool::new_unfiltered();
                let device = Box::new(MockDevice::new("mock", "bench003", false));
                pool.add_device(device, scanner::hardware::types::Backend::Mock);

                let requirements = TaskRequirements {
                    frequency_hz: 88.9e6,
                    bandwidth_hz: 200e3,
                    required_sample_rate: 2.4e6,
                    priority: TaskPriority::Normal,
                };

                b.iter(|| {
                    for _ in 0..count {
                        let tuner = pool
                            .try_acquire(&requirements, TunerActivity::Scanning)
                            .expect("Failed to acquire tuner");

                        let mut graph = rustradio::graph::Graph::new();
                        let _stream = tuner
                            .add_source_to_graph(&mut graph, 100.0e6, 2.4e6, 30.0)
                            .expect("Failed to add source");

                        let _ = tuner.stop_stream();
                        drop(tuner);
                    }
                });

                pool.shutdown();
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_control_message_roundtrip,
    bench_stop_stream_roundtrip,
    bench_multiple_roundtrips
);
criterion_main!(benches);
