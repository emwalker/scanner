use std::{thread, time::Duration};

use scanner::{hardware::DeviceId, ipc::ControlMessage};

#[test]
fn test_protocol_subprocess_control() {
    let mut fixture = super::common::SubprocessTestFixture::new();

    let device_id = DeviceId::from_serial("mock", "protocol001");

    fixture
        .spawn_worker(device_id)
        .expect("Failed to spawn worker");

    fixture.connect().expect("Failed to connect to worker");

    let config_msg = ControlMessage::ConfigureAndStart {
        channel: 0,
        freq_hz: 88.9e6,
        gain_db: 24.0,
        sample_rate: 2_000_000.0,
    };

    let ready = fixture
        .recv_control()
        .expect("Failed to receive Ready message");

    match ready {
        ControlMessage::Ready { .. } => {}
        other => panic!("Expected Ready, got {:?}", other),
    }

    fixture
        .send_control(&config_msg)
        .expect("Failed to send configure message");

    let response = fixture.recv_control().expect("Failed to receive response");

    match response {
        ControlMessage::StreamStarted { .. } => {}
        other => panic!("Expected StreamStarted, got {:?}", other),
    }

    thread::sleep(Duration::from_millis(100));

    let packet = fixture.recv_data().expect("Failed to receive data packet");

    assert_eq!(packet.channel, 0);
    assert!(!packet.samples.is_empty());

    let stop_msg = ControlMessage::StopStream { channel: 0 };
    fixture
        .send_control(&stop_msg)
        .expect("Failed to send stop message");

    fixture.shutdown().expect("Failed to shutdown worker");
}

#[test]
fn test_protocol_multiple_data_packets() {
    let mut fixture = super::common::SubprocessTestFixture::new();

    let device_id = DeviceId::from_serial("mock", "protocol002");

    fixture
        .spawn_worker(device_id)
        .expect("Failed to spawn worker");

    fixture.connect().expect("Failed to connect to worker");

    let _ready = fixture
        .recv_control()
        .expect("Failed to receive Ready message");

    let config_msg = ControlMessage::ConfigureAndStart {
        channel: 0,
        freq_hz: 101.5e6,
        gain_db: 30.0,
        sample_rate: 2_400_000.0,
    };

    fixture
        .send_control(&config_msg)
        .expect("Failed to send configure message");

    let _response = fixture.recv_control().expect("Failed to receive response");

    let mut packets_received = 0;
    for _ in 0..5 {
        match fixture.recv_data() {
            Ok(packet) => {
                assert_eq!(packet.channel, 0);
                packets_received += 1;
            }
            Err(_) => break,
        }
    }

    assert!(packets_received > 0, "Should receive at least one packet");

    fixture.shutdown().expect("Failed to shutdown worker");
}

#[test]
fn test_protocol_stop_and_restart() {
    let mut fixture = super::common::SubprocessTestFixture::new();

    let device_id = DeviceId::from_serial("mock", "protocol003");

    fixture
        .spawn_worker(device_id)
        .expect("Failed to spawn worker");

    fixture.connect().expect("Failed to connect to worker");

    let _ready = fixture
        .recv_control()
        .expect("Failed to receive Ready message");

    let config_msg = ControlMessage::ConfigureAndStart {
        channel: 0,
        freq_hz: 88.9e6,
        gain_db: 24.0,
        sample_rate: 2_000_000.0,
    };

    fixture
        .send_control(&config_msg)
        .expect("Failed to send configure message");

    let _start1 = fixture
        .recv_control()
        .expect("Failed to receive StreamStarted");

    let stop_msg = ControlMessage::StopStream { channel: 0 };
    fixture
        .send_control(&stop_msg)
        .expect("Failed to send stop message");

    let _stopped = fixture
        .recv_control()
        .expect("Failed to receive StreamStopped");

    let config_msg2 = ControlMessage::ConfigureAndStart {
        channel: 0,
        freq_hz: 101.5e6,
        gain_db: 30.0,
        sample_rate: 2_400_000.0,
    };

    fixture
        .send_control(&config_msg2)
        .expect("Failed to send second configure message");

    let response = fixture.recv_control().expect("Failed to receive response");

    match response {
        ControlMessage::StreamStarted { actual_freq, .. } => {
            assert!((actual_freq - 101.5e6).abs() < 1.0);
        }
        other => panic!("Expected StreamStarted, got {:?}", other),
    }

    fixture.shutdown().expect("Failed to shutdown worker");
}
