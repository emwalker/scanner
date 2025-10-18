//! Messages for coordinator-worker communication

use crate::ecs::ScanId;
use crate::ecs::components::Priority;
use crate::hardware::pool::TunerId;

#[derive(Debug, Clone)]
pub enum WorkerEvent {
    ScanStarted {
        scan_id: ScanId,
    },
    WindowCompleted {
        scan_id: ScanId,
        window_num: usize,
        candidates_found: usize,
    },
    ScanPaused {
        scan_id: ScanId,
        window_num: usize,
    },
    ScanResumed {
        scan_id: ScanId,
        window_num: usize,
    },
    TunerAllocated {
        scan_id: ScanId,
        tuner_id: TunerId,
    },
    TunerReleased {
        scan_id: ScanId,
        tuner_id: TunerId,
    },
    StationDiscovered {
        scan_id: ScanId,
        frequency: f64,
        signal_strength: f64,
    },
}

#[derive(Debug, Clone)]
pub enum WorkerCommand {
    ProcessNextWindow {
        window_num: usize,
    },
    PauseScan,
    ResumeScan,
    CompleteScan,
    AllocateTuner {
        frequency_hz: f64,
        bandwidth_hz: f64,
        for_audio: bool,
    },
    ReleaseTuner {
        tuner_id: TunerId,
    },
    ChangePriority {
        priority: Priority,
    },
}

#[derive(Debug)]
pub enum CommandError {
    CoordinatorNotResponding,
    CoordinatorShutdown,
    InvalidCommand,
}
