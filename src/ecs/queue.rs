//! FIFO queues for ECS resources

use crate::ecs::{ScanId, StationId};
use std::collections::VecDeque;

/// Request to tune a station for playback
#[derive(Debug, Clone)]
pub struct TunerRequest {
    pub station_id: StationId,
    pub frequency: f64,
    pub window_id: usize,
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
pub struct PauseRequest {
    pub scan_id: ScanId,
    pub window_num: usize,
    pub station_frequency_hz: Option<f64>,
    pub window_center_frequency_hz: Option<f64>,
}

impl PauseRequest {
    pub fn new(scan_id: ScanId, window_num: usize) -> Self {
        Self {
            scan_id,
            window_num,
            station_frequency_hz: None,
            window_center_frequency_hz: None,
        }
    }

    pub fn with_station(
        scan_id: ScanId,
        window_num: usize,
        station_frequency_hz: f64,
        window_center_frequency_hz: f64,
    ) -> Self {
        Self {
            scan_id,
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
/// of the queue, setting the pause_request component on the target ScanEntity.
pub type PauseRequestQueue = VecDeque<PauseRequest>;
