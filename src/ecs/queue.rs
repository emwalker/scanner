//! FIFO queues for ECS resources

use std::collections::VecDeque;

use crate::{
    ecs::{StationId, TaskId, components::window::WindowId},
    hardware::pool::{TaskRequirements, TunerActivity},
};

/// Request to tune a station for playback
#[derive(Debug, Clone)]
pub struct TunerRequest {
    pub station_id: StationId,
    pub frequency: f64,
    pub window_id: WindowId,
    pub center_frequency: f64,
}

/// FIFO queue for tuner acquisition requests
///
/// Stations request tuner access for playback. The AudioPlaybackSystem
/// processes requests from the front of the queue, acquiring tuners
/// when available. If a tuner isn't available, the request stays in
/// the queue and is retried next tick.
pub type TunerRequestQueue = VecDeque<TunerRequest>;

/// Request to pause a scan and optionally tune to a station
#[derive(Debug, Clone)]
pub struct PauseAndTuneRequest {
    pub task_id: TaskId,
    pub window_num: usize,
    pub station_frequency_hz: Option<f64>,
    pub window_center_frequency_hz: Option<f64>,
}

impl PauseAndTuneRequest {
    pub fn new(task_id: TaskId, window_num: usize) -> Self {
        Self {
            task_id,
            window_num,
            station_frequency_hz: None,
            window_center_frequency_hz: None,
        }
    }

    pub fn with_station(
        task_id: TaskId,
        window_num: usize,
        station_frequency_hz: f64,
        window_center_frequency_hz: f64,
    ) -> Self {
        Self {
            task_id,
            window_num,
            station_frequency_hz: Some(station_frequency_hz),
            window_center_frequency_hz: Some(window_center_frequency_hz),
        }
    }
}

/// FIFO queue for pause requests
///
/// The TUI and other input sources push pause requests to this queue
/// when the user wants to pause scanning and tune to a station.
/// The ScanRequestProcessorSystem processes requests from the front
/// of the queue, setting the pause_request component on the target TaskEntity.
pub type PauseRequestQueue = VecDeque<PauseAndTuneRequest>;

/// Requester type for unified tuner allocation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TunerRequester {
    Window(WindowId),
    Station(StationId),
}

/// Unified tuner allocation request
///
/// All entities requesting tuners (windows for scanning, stations for playback)
/// go through this unified queue. The TunerAllocationSystem processes requests
/// in FIFO order, allocating tuners when available.
#[derive(Debug, Clone)]
pub struct TunerAllocationRequest {
    pub requester: TunerRequester,
    pub requirements: TaskRequirements,
    pub activity: TunerActivity,
    pub requester_id: String,
}

/// FIFO queue for unified tuner allocation
///
/// All tuner allocation requests (windows, stations) share this queue.
/// The TunerAllocationSystem processes requests from the front of the queue,
/// acquiring tuners when available. If a tuner isn't available, the request
/// stays in the queue and is retried next tick.
///
/// FIFO ordering ensures fair allocation across competing uses.
pub type TunerAllocationQueue = VecDeque<TunerAllocationRequest>;
