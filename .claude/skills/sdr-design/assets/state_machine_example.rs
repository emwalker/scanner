// Example: ECS-based State Machine for Audio Thread Lifecycle
//
// This demonstrates how to manage thread lifecycle using ECS components
// and state machines in the context of an SDR application.
//
// Key patterns:
// - State enum with explicit transitions
// - Components hold thread handles and control channels
// - Systems implement state transitions
// - Atomic flags for shutdown coordination

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use crossbeam::channel::{self, Sender, Receiver};

// ============================================================================
// State Machine Definition
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioThreadState {
    Idle,              // No thread exists
    Starting,          // Thread spawn initiated
    Running,           // Thread actively processing
    Stopping,          // Shutdown requested
    Failed(ErrorCode), // Thread exited with error
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    SpawnFailed,
    UnexpectedExit,
    HardwareError,
}

// ============================================================================
// Control Messages
// ============================================================================

#[derive(Debug, Clone)]
pub enum AudioControl {
    Start { station_id: u64 },
    UpdateVolume(f32),
    Shutdown,
}

#[derive(Debug, Clone)]
pub enum AudioEvent {
    Started { station_id: u64 },
    QualityUpdate { snr_db: f32 },
    Error(String),
    Stopped,
}

// ============================================================================
// ECS Component
// ============================================================================

pub struct AudioThreadComponent {
    pub state: AudioThreadState,
    pub handle: Option<JoinHandle<Result<(), String>>>,
    pub control_tx: Option<Sender<AudioControl>>,
    pub event_rx: Option<Receiver<AudioEvent>>,
    pub shutdown_flag: Arc<AtomicBool>,
}

impl AudioThreadComponent {
    pub fn new() -> Self {
        Self {
            state: AudioThreadState::Idle,
            handle: None,
            control_tx: None,
            event_rx: None,
            shutdown_flag: Arc::new(AtomicBool::new(false)),
        }
    }
}

// ============================================================================
// Thread Worker Function
// ============================================================================

fn audio_worker_thread(
    control_rx: Receiver<AudioControl>,
    event_tx: Sender<AudioEvent>,
    shutdown_flag: Arc<AtomicBool>,
) -> Result<(), String> {
    // Send ready signal
    event_tx.send(AudioEvent::Started { station_id: 0 })
        .map_err(|_| "Failed to send started event")?;

    // Main loop
    while !shutdown_flag.load(Ordering::SeqCst) {
        // Check for control messages (non-blocking with timeout)
        match control_rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(AudioControl::Start { station_id }) => {
                // Start processing for station
                println!("Audio thread: Starting station {}", station_id);
            }
            Ok(AudioControl::UpdateVolume(volume)) => {
                println!("Audio thread: Updating volume to {}", volume);
            }
            Ok(AudioControl::Shutdown) => {
                println!("Audio thread: Shutdown requested");
                break;
            }
            Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                // No message, continue
            }
            Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                // Control channel closed, exit
                break;
            }
        }

        // Simulate audio processing work
        // In real implementation: read from broadcast channel, demodulate, play audio
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    // Send stopped event
    let _ = event_tx.send(AudioEvent::Stopped);

    Ok(())
}

// ============================================================================
// ECS Systems
// ============================================================================

/// System 1: Spawn audio thread when needed
pub fn spawn_audio_thread_system(component: &mut AudioThreadComponent) {
    if component.state == AudioThreadState::Idle {
        // Check if we should spawn (in real app, check for StationTuned event)
        let should_spawn = true; // Placeholder

        if should_spawn {
            // Create channels
            let (control_tx, control_rx) = channel::unbounded();
            let (event_tx, event_rx) = channel::unbounded();

            // Reset shutdown flag
            component.shutdown_flag.store(false, Ordering::SeqCst);

            // Spawn thread
            let shutdown_flag = component.shutdown_flag.clone();
            match thread::Builder::new()
                .name("audio-worker".to_string())
                .spawn(move || audio_worker_thread(control_rx, event_tx, shutdown_flag))
            {
                Ok(handle) => {
                    component.state = AudioThreadState::Starting;
                    component.handle = Some(handle);
                    component.control_tx = Some(control_tx);
                    component.event_rx = Some(event_rx);
                    println!("Audio thread spawned");
                }
                Err(e) => {
                    eprintln!("Failed to spawn audio thread: {}", e);
                    component.state = AudioThreadState::Failed(ErrorCode::SpawnFailed);
                }
            }
        }
    }
}

/// System 2: Monitor audio thread state transitions
pub fn monitor_audio_thread_system(component: &mut AudioThreadComponent) {
    match component.state {
        AudioThreadState::Starting => {
            // Check for ready signal from thread
            if let Some(event_rx) = &component.event_rx {
                match event_rx.try_recv() {
                    Ok(AudioEvent::Started { .. }) => {
                        component.state = AudioThreadState::Running;
                        println!("Audio thread transitioned to Running");
                    }
                    Ok(AudioEvent::Error(e)) => {
                        eprintln!("Audio thread error during startup: {}", e);
                        component.state = AudioThreadState::Failed(ErrorCode::UnexpectedExit);
                    }
                    Err(_) => {
                        // No message yet, keep waiting
                    }
                }
            }
        }

        AudioThreadState::Running => {
            // Check if thread is still alive
            if let Some(handle) = &component.handle {
                if handle.is_finished() {
                    eprintln!("Audio thread exited unexpectedly");
                    component.state = AudioThreadState::Failed(ErrorCode::UnexpectedExit);
                }
            }

            // Process events from thread
            if let Some(event_rx) = &component.event_rx {
                while let Ok(event) = event_rx.try_recv() {
                    match event {
                        AudioEvent::QualityUpdate { snr_db } => {
                            println!("Audio quality update: SNR = {} dB", snr_db);
                        }
                        AudioEvent::Error(e) => {
                            eprintln!("Audio thread error: {}", e);
                        }
                        AudioEvent::Stopped => {
                            component.state = AudioThreadState::Stopping;
                        }
                        _ => {}
                    }
                }
            }
        }

        AudioThreadState::Stopping => {
            // Wait for thread to exit
            if let Some(handle) = component.handle.take() {
                match handle.join() {
                    Ok(Ok(())) => {
                        component.state = AudioThreadState::Idle;
                        component.control_tx = None;
                        component.event_rx = None;
                        println!("Audio thread stopped cleanly");
                    }
                    Ok(Err(e)) => {
                        eprintln!("Audio thread exited with error: {}", e);
                        component.state = AudioThreadState::Failed(ErrorCode::UnexpectedExit);
                    }
                    Err(_) => {
                        eprintln!("Audio thread panicked");
                        component.state = AudioThreadState::Failed(ErrorCode::UnexpectedExit);
                    }
                }
            }
        }

        AudioThreadState::Failed(_) => {
            // Cleanup resources
            if let Some(handle) = component.handle.take() {
                let _ = handle.join();
            }
            component.control_tx = None;
            component.event_rx = None;
        }

        AudioThreadState::Idle => {
            // Nothing to monitor
        }
    }
}

/// System 3: Shutdown audio thread when needed
pub fn shutdown_audio_thread_system(component: &mut AudioThreadComponent) {
    if component.state == AudioThreadState::Running {
        // Check if we should shutdown (in real app, check for TuneAway event or app exit)
        let should_shutdown = false; // Placeholder

        if should_shutdown {
            // Set shutdown flag (non-blocking)
            component.shutdown_flag.store(true, Ordering::SeqCst);

            // Send shutdown message on control channel
            if let Some(control_tx) = &component.control_tx {
                let _ = control_tx.send(AudioControl::Shutdown);
            }

            component.state = AudioThreadState::Stopping;
            println!("Audio thread shutdown initiated");
        }
    }
}

// ============================================================================
// Example Usage
// ============================================================================

fn main() {
    let mut audio_component = AudioThreadComponent::new();

    // Frame 1: Spawn thread
    spawn_audio_thread_system(&mut audio_component);

    // Frame 2-3: Wait for thread to start
    for _ in 0..3 {
        std::thread::sleep(std::time::Duration::from_millis(50));
        monitor_audio_thread_system(&mut audio_component);
    }

    // Frame 4-10: Thread running
    for _ in 0..7 {
        std::thread::sleep(std::time::Duration::from_millis(50));
        monitor_audio_thread_system(&mut audio_component);
        shutdown_audio_thread_system(&mut audio_component);
    }

    // Trigger shutdown
    audio_component.shutdown_flag.store(true, Ordering::SeqCst);
    if let Some(control_tx) = &audio_component.control_tx {
        let _ = control_tx.send(AudioControl::Shutdown);
    }
    audio_component.state = AudioThreadState::Stopping;

    // Frame 11: Wait for shutdown
    std::thread::sleep(std::time::Duration::from_millis(200));
    monitor_audio_thread_system(&mut audio_component);

    println!("Final state: {:?}", audio_component.state);
}
