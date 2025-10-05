use scanner::discovery::{self, DiscoveryMode, Event};
use scanner::sdr::{Backend, Mock};
use scanner::shutdown::ShutdownCoordinator;
use std::sync::mpsc;
use std::time::Duration;

#[test]
fn test_polling_discovery_integration() {
    let _ = tracing_subscriber::fmt::try_init();

    let coordinator = ShutdownCoordinator::new();
    let backends: Vec<Box<dyn Backend>> = vec![Box::new(Mock)];

    let mut service = discovery::create_for_testing(
        backends,
        DiscoveryMode::ForcePolling(Duration::from_millis(100)),
    );
    let (event_tx, event_rx) = mpsc::channel();

    coordinator
        .spawn_sdr_thread(move |cancel| {
            service.run(event_tx, cancel);
        })
        .unwrap();

    let mut events = Vec::new();
    let timeout = Duration::from_millis(500);
    let start = std::time::Instant::now();

    while start.elapsed() < timeout {
        if let Ok(event) = event_rx.recv_timeout(Duration::from_millis(50)) {
            events.push(event);
        }
    }

    assert!(!events.is_empty(), "Should detect existing devices");

    let added_count = events
        .iter()
        .filter(|e| matches!(e, Event::Added(_)))
        .count();
    assert_eq!(added_count, 2, "Should detect 2 mock devices");

    coordinator.shutdown();
    coordinator.wait().unwrap();
}

#[test]
#[cfg(target_os = "linux")]
fn test_udev_discovery_integration() {
    let _ = tracing_subscriber::fmt::try_init();

    let coordinator = ShutdownCoordinator::new();
    let backends: Vec<Box<dyn Backend>> = vec![Box::new(Mock)];

    let mut service = discovery::create_for_testing(backends, DiscoveryMode::Auto);
    let (event_tx, event_rx) = mpsc::channel();

    coordinator
        .spawn_sdr_thread(move |cancel| {
            service.run(event_tx, cancel);
        })
        .unwrap();

    let mut events = Vec::new();
    let timeout = Duration::from_millis(500);
    let start = std::time::Instant::now();

    while start.elapsed() < timeout {
        if let Ok(event) = event_rx.recv_timeout(Duration::from_millis(50)) {
            events.push(event);
        }
    }

    assert!(
        !events.is_empty(),
        "Should detect existing devices via udev"
    );

    coordinator.shutdown();
    coordinator.wait().unwrap();
}

#[test]
fn test_discovery_shutdown_responsiveness() {
    let _ = tracing_subscriber::fmt::try_init();

    let coordinator = ShutdownCoordinator::new();
    let backends: Vec<Box<dyn Backend>> = vec![Box::new(Mock)];

    let mut service = discovery::create_for_testing(
        backends,
        DiscoveryMode::ForcePolling(Duration::from_secs(10)),
    );
    let (event_tx, _event_rx) = mpsc::channel();

    coordinator
        .spawn_sdr_thread(move |cancel| {
            service.run(event_tx, cancel);
        })
        .unwrap();

    std::thread::sleep(Duration::from_millis(100));

    let shutdown_start = std::time::Instant::now();
    coordinator.shutdown();
    coordinator.wait().unwrap();
    let shutdown_duration = shutdown_start.elapsed();

    assert!(
        shutdown_duration < Duration::from_secs(1),
        "Shutdown should be quick even with long poll interval, took {:?}",
        shutdown_duration
    );
}
