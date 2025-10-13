use criterion::{Criterion, black_box, criterion_group, criterion_main};
use num::Complex;
use scanner::ipc::{ControlMessage, IQPacket};

fn bench_control_message_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("control_messages");
    group.noise_threshold(0.15);

    group.bench_function("serialize ConfigureAndStart", |b| {
        let msg = ControlMessage::ConfigureAndStart {
            channel: 0,
            freq_hz: 88.9e6,
            gain_db: 24.0,
            sample_rate: 2_000_000.0,
        };
        b.iter(|| {
            let bytes = postcard::to_allocvec(black_box(&msg)).unwrap();
            black_box(bytes);
        });
    });

    group.bench_function("deserialize ConfigureAndStart", |b| {
        let msg = ControlMessage::ConfigureAndStart {
            channel: 0,
            freq_hz: 88.9e6,
            gain_db: 24.0,
            sample_rate: 2_000_000.0,
        };
        let bytes = postcard::to_allocvec(&msg).unwrap();
        b.iter(|| {
            let msg: ControlMessage = postcard::from_bytes(black_box(&bytes)).unwrap();
            black_box(msg);
        });
    });

    group.finish();
}

fn bench_iq_packet_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("iq_packets");
    group.noise_threshold(0.30);

    group.bench_function("serialize IQPacket (1024 samples)", |b| {
        let samples: Vec<Complex<f32>> = (0..1024)
            .map(|i| Complex::new(i as f32 / 1024.0, (1024 - i) as f32 / 1024.0))
            .collect();

        let packet = IQPacket {
            channel: 0,
            sequence: 42,
            timestamp: 123456789,
            samples,
        };

        b.iter(|| {
            let bytes = postcard::to_allocvec(black_box(&packet)).unwrap();
            black_box(bytes);
        });
    });

    group.bench_function("deserialize IQPacket (1024 samples)", |b| {
        let samples: Vec<Complex<f32>> = (0..1024)
            .map(|i| Complex::new(i as f32 / 1024.0, (1024 - i) as f32 / 1024.0))
            .collect();

        let packet = IQPacket {
            channel: 0,
            sequence: 42,
            timestamp: 123456789,
            samples,
        };

        let bytes = postcard::to_allocvec(&packet).unwrap();

        b.iter(|| {
            let packet: IQPacket = postcard::from_bytes(black_box(&bytes)).unwrap();
            black_box(packet);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_control_message_serialization,
    bench_iq_packet_serialization
);
criterion_main!(benches);
