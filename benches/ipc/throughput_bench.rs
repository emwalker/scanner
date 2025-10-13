use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rustradio::blocks::NullSink;
use rustradio::graph::GraphRunner;
use scanner::hardware::mock::MockDevice;
use scanner::hardware::pool::{Pool, TaskPriority, TaskRequirements, TunerActivity};
use std::time::Duration;

fn bench_data_streaming_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipc_data_throughput");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for &duration_ms in &[100, 250] {
        group.throughput(Throughput::Elements(duration_ms));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}ms", duration_ms)),
            &duration_ms,
            |b, &duration_ms| {
                b.iter(|| {
                    let pool = Pool::new_unfiltered();
                    let device = Box::new(MockDevice::new("mock", "throughput001", false));
                    pool.add_device(device, scanner::hardware::types::Backend::Mock);

                    let requirements = TaskRequirements {
                        frequency_hz: 88.9e6,
                        bandwidth_hz: 200e3,
                        required_sample_rate: 2.4e6,
                        priority: TaskPriority::Normal,
                    };

                    let tuner = pool
                        .try_acquire(&requirements, TunerActivity::Scanning)
                        .expect("Failed to acquire tuner");

                    let mut graph = rustradio::graph::Graph::new();
                    let stream = tuner
                        .add_source_to_graph(&mut graph, 100.0e6, 2.4e6, 30.0)
                        .expect("Failed to add source");

                    graph.add(Box::new(NullSink::new(stream)));

                    let cancel_token = graph.cancel_token();
                    let graph_handle = std::thread::spawn(move || {
                        graph.run().ok();
                    });

                    std::thread::sleep(Duration::from_millis(duration_ms));
                    cancel_token.cancel();
                    graph_handle.join().ok();

                    let _ = tuner.stop_stream();
                    drop(tuner);
                    pool.shutdown();
                });
            },
        );
    }

    group.finish();
}

fn bench_sample_rate_handling(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipc_sample_rates");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for &sample_rate in &[2.0e6, 2.4e6] {
        group.throughput(Throughput::Elements(sample_rate as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}MHz", sample_rate / 1e6)),
            &sample_rate,
            |b, &sample_rate| {
                b.iter(|| {
                    let pool = Pool::new_unfiltered();
                    let device = Box::new(MockDevice::new("mock", "samplerate001", false));
                    pool.add_device(device, scanner::hardware::types::Backend::Mock);

                    let requirements = TaskRequirements {
                        frequency_hz: 88.9e6,
                        bandwidth_hz: 200e3,
                        required_sample_rate: sample_rate,
                        priority: TaskPriority::Normal,
                    };

                    let tuner = pool
                        .try_acquire(&requirements, TunerActivity::Scanning)
                        .expect("Failed to acquire tuner");

                    let mut graph = rustradio::graph::Graph::new();
                    let stream = tuner
                        .add_source_to_graph(&mut graph, 100.0e6, sample_rate, 30.0)
                        .expect("Failed to add source");

                    graph.add(Box::new(NullSink::new(stream)));

                    let cancel_token = graph.cancel_token();
                    let graph_handle = std::thread::spawn(move || {
                        graph.run().ok();
                    });

                    std::thread::sleep(Duration::from_millis(250));
                    cancel_token.cancel();
                    graph_handle.join().ok();

                    let _ = tuner.stop_stream();
                    drop(tuner);
                    pool.shutdown();
                });
            },
        );
    }

    group.finish();
}

fn bench_concurrent_channels(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipc_concurrent_channels");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("two_channels_concurrent", |b| {
        b.iter(|| {
            let pool = Pool::new_unfiltered();

            let mut caps = scanner::hardware::Capabilities::for_mock("mock", "dual001");
            caps.channels = 2;

            pool.add_device_metadata(
                caps.device_id.clone(),
                caps,
                scanner::hardware::types::Backend::Mock,
            );

            let requirements = TaskRequirements {
                frequency_hz: 88.9e6,
                bandwidth_hz: 200e3,
                required_sample_rate: 2.4e6,
                priority: TaskPriority::Normal,
            };

            let tuner1 = pool
                .try_acquire(&requirements, TunerActivity::Scanning)
                .expect("Failed to acquire tuner 1");

            let tuner2 = pool
                .try_acquire(&requirements, TunerActivity::Listening)
                .expect("Failed to acquire tuner 2");

            let mut graph1 = rustradio::graph::Graph::new();
            let stream1 = tuner1
                .add_source_to_graph(&mut graph1, 88.9e6, 2.4e6, 30.0)
                .expect("Failed to add source 1");

            let mut graph2 = rustradio::graph::Graph::new();
            let stream2 = tuner2
                .add_source_to_graph(&mut graph2, 91.5e6, 2.4e6, 30.0)
                .expect("Failed to add source 2");

            graph1.add(Box::new(NullSink::new(stream1)));
            graph2.add(Box::new(NullSink::new(stream2)));

            let cancel1 = graph1.cancel_token();
            let cancel2 = graph2.cancel_token();

            let handle1 = std::thread::spawn(move || {
                graph1.run().ok();
            });

            let handle2 = std::thread::spawn(move || {
                graph2.run().ok();
            });

            std::thread::sleep(Duration::from_millis(250));

            cancel1.cancel();
            cancel2.cancel();

            handle1.join().ok();
            handle2.join().ok();

            let _ = tuner1.stop_stream();
            let _ = tuner2.stop_stream();

            drop(tuner1);
            drop(tuner2);

            pool.shutdown();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_data_streaming_throughput,
    bench_sample_rate_handling,
    bench_concurrent_channels
);
criterion_main!(benches);
