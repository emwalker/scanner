use crate::core::types::{Result, ScannerError};
use crate::hardware::pool::{Pool, PoolFilter, TuningMode};
use crate::shutdown::ShutdownCoordinator;
use crate::task::TaskScheduler;
use crate::ui::tui::themes::ThemeName;
use std::sync::Arc;

use super::args::ScanArgs;
use super::config::build_scanning_config;
use super::discovery::start_discovery_service;
use super::signals::setup_signal_handler;
use super::tui_mode::{TuiRunContext, create_logger, run_with_tui, setup_tui_channels, start_tui};

fn parse_stations(stations_str: &str) -> Result<Vec<f64>> {
    stations_str
        .split(',')
        .map(|s| s.trim().parse::<f64>().map_err(ScannerError::from))
        .collect()
}

pub fn handle_scan_command(args: ScanArgs) -> Result<()> {
    let config = build_scanning_config(&args)?;

    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());
    setup_signal_handler(shutdown_coordinator.clone())?;

    run_tui_mode(&args, config, shutdown_coordinator)
}

fn run_tui_mode(
    args: &ScanArgs,
    config: crate::core::types::ScanningConfig,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
) -> Result<()> {
    let (tui_context, tui_event_receiver) = setup_tui_channels();

    let theme_name = args
        .theme
        .parse::<ThemeName>()
        .map_err(|e| ScannerError::InvalidTheme {
            theme: args.theme.clone(),
            reason: e.to_string(),
        })?;

    let filter = PoolFilter::new()
        .with_driver("sdrplay")
        .with_mode(TuningMode::SingleTuner);
    let shared_pool = Arc::new(Pool::new(filter, args.log_file.clone()));

    let scheduler = Arc::new(TaskScheduler::new(
        shared_pool.clone(),
        shutdown_coordinator.clone(),
    ));

    let discovery_setup = start_discovery_service(
        tui_context.tui_event_sender.clone(),
        shutdown_coordinator.clone(),
        scheduler.clone(),
        shared_pool.clone(),
    )?;

    use crate::ecs::components::scan::{ScanConfigComponent, ScanType};
    use crate::ecs::{AudioEntity, CandidateEntity, EntityWorld, ScanEntity, StationEntity};
    use std::sync::RwLock;

    let scan_entities = Arc::new(RwLock::new(EntityWorld::<ScanEntity>::new()));
    let station_entities = Arc::new(RwLock::new(EntityWorld::<StationEntity>::new()));
    let audio_entities = Arc::new(RwLock::new(EntityWorld::<AudioEntity>::new()));
    let candidate_entities = Arc::new(RwLock::new(EntityWorld::<CandidateEntity>::new()));

    if let Some(ref stations_str) = args.stations {
        let stations = parse_stations(stations_str)?;
        let freq_min = stations.iter().cloned().fold(f64::INFINITY, f64::min);
        let freq_max = stations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let scan_config = ScanConfigComponent::new(
            ScanType::Stations,
            freq_min,
            freq_max,
            config.samp_rate,
            config.samp_rate,
            config.sdr_gain,
            config.duration as f64,
            1,
        )
        .with_stations(stations);

        let scan_entity = ScanEntity::new(scan_config);
        scan_entities.write().unwrap().insert(scan_entity);
    } else {
        let (freq_min, freq_max) = config.band.frequency_range();
        let scan_config = ScanConfigComponent::new(
            ScanType::Band,
            freq_min,
            freq_max,
            config.samp_rate,
            config.samp_rate,
            config.sdr_gain,
            config.duration as f64,
            config.scanning_windows.unwrap_or(1),
        );

        let scan_entity = ScanEntity::new(scan_config);
        scan_entities.write().unwrap().insert(scan_entity);
    }

    let tui_handle = start_tui(
        tui_event_receiver,
        shutdown_coordinator.clone(),
        theme_name,
        scan_entities.clone(),
        station_entities.clone(),
        audio_entities.clone(),
        candidate_entities.clone(),
    );

    let logger = create_logger(args);
    crate::logging::init(logger.as_ref(), args.verbose)?;

    let run_context = TuiRunContext {
        config,
        stations: args.stations.clone(),
        shutdown_coordinator: shutdown_coordinator.clone(),
        pool: shared_pool.clone(),
        scheduler: scheduler.clone(),
        logger,
        scan_entities,
        station_entities,
        audio_entities,
        candidate_entities,
    };

    let result = run_with_tui(run_context, tui_context, tui_handle);

    let _ = discovery_setup.discovery_handle.join();
    let _ = discovery_setup.discovery_forwarder.join();

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_stations() {
        let stations = parse_stations("88.9e6,101.5e6,107.3e6").unwrap();
        assert_eq!(stations, vec![88.9e6, 101.5e6, 107.3e6]);
    }

    #[test]
    fn test_parse_stations_with_whitespace() {
        let stations = parse_stations("88.9e6, 101.5e6 , 107.3e6").unwrap();
        assert_eq!(stations, vec![88.9e6, 101.5e6, 107.3e6]);
    }

    #[test]
    fn test_parse_stations_invalid() {
        let result = parse_stations("88.9e6,invalid,107.3e6");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_stations_single() {
        let stations = parse_stations("88.9e6").unwrap();
        assert_eq!(stations, vec![88.9e6]);
    }
}
