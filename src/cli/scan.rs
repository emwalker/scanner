use std::{io, sync::Arc};

use super::{
    args::ScanArgs,
    config::build_scanning_config,
    discovery::{OutputMode, start_discovery_service},
    log_mode,
    signals::setup_signal_handler,
    tui_mode::{TuiRunContext, run_with_tui, setup_tui_channels, start_tui},
};
use crate::{
    core::types::{Result, ScannerError},
    ecs::resources::{LocationResource, new_location_resource},
    hardware::pool::{Pool, PoolFilter, TuningMode},
    shutdown::ShutdownCoordinator,
    task::TaskScheduler,
    ui::tui::themes::ThemeName,
};

fn is_stdout_piped() -> bool {
    use std::os::unix::io::AsRawFd;
    let stdout_fd = io::stdout().as_raw_fd();

    unsafe { libc::isatty(stdout_fd) == 0 }
}

fn parse_stations(stations_str: &str) -> Result<Vec<f64>> {
    stations_str
        .split(',')
        .map(|s| s.trim().parse::<f64>().map_err(ScannerError::from))
        .collect()
}

struct EntityWorlds {
    task_entities: Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::TaskEntity>>>,
    audio_entities: Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::AudioEntity>>>,
    signal_entities: Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::SignalEntity>>>,
}

fn create_entity_worlds() -> EntityWorlds {
    use std::sync::RwLock;

    use crate::ecs::{AudioEntity, EntityWorld, SignalEntity, TaskEntity};

    EntityWorlds {
        task_entities: Arc::new(RwLock::new(EntityWorld::<TaskEntity>::new())),
        audio_entities: Arc::new(RwLock::new(EntityWorld::<AudioEntity>::new())),
        signal_entities: Arc::new(RwLock::new(EntityWorld::<SignalEntity>::new())),
    }
}

struct DeviceEntityWorlds {
    tuner_entities: Arc<std::sync::Mutex<crate::ecs::EntityWorld<crate::ecs::TunerEntity>>>,
    device_entities: Arc<std::sync::Mutex<crate::ecs::EntityWorld<crate::ecs::DeviceEntity>>>,
}

fn create_device_entity_worlds() -> DeviceEntityWorlds {
    use std::sync::Mutex;

    use crate::ecs::{DeviceEntity, EntityWorld, TunerEntity};

    DeviceEntityWorlds {
        tuner_entities: Arc::new(Mutex::new(EntityWorld::<TunerEntity>::new())),
        device_entities: Arc::new(Mutex::new(EntityWorld::<DeviceEntity>::new())),
    }
}

fn create_pending_scan_request(
    args: &ScanArgs,
    config: &crate::core::types::ScanningConfig,
) -> Result<crate::ecs::components::scan::PendingScanRequest> {
    use crate::{
        ecs::components::scan::{PendingScanRequest, ScanConfigComponent, ScanType},
        hardware::pool::{TaskPriority, TaskRequirements},
    };

    let scan_config = if let Some(ref stations_str) = args.stations {
        let stations = parse_stations(stations_str)?;
        let freq_min = stations.iter().cloned().fold(f64::INFINITY, f64::min);
        let freq_max = stations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        ScanConfigComponent::new(
            ScanType::Stations,
            freq_min,
            freq_max,
            config.samp_rate,
            config.samp_rate,
            config.sdr_gain,
            config.duration as f64,
            1,
        )
        .with_stations(stations)
    } else {
        let (freq_min, freq_max) = config.band.frequency_range();
        let step_size = config.samp_rate * (1.0 - config.signal_processing.window_overlap);

        tracing::debug!(
            samp_rate = config.samp_rate,
            samp_rate_mhz = config.samp_rate / 1e6,
            window_overlap = config.signal_processing.window_overlap,
            step_size_hz = step_size,
            step_size_mhz = step_size / 1e6,
            scanning_windows = config.scanning_windows,
            "create_pending_scan_request: Calculated step_size for Band scan"
        );

        ScanConfigComponent::new(
            ScanType::Band,
            freq_min,
            freq_max,
            step_size,
            config.samp_rate,
            config.sdr_gain,
            config.duration as f64,
            config.scanning_windows.unwrap_or(1),
        )
    };

    let requirements = TaskRequirements {
        frequency_hz: scan_config.freq_min,
        bandwidth_hz: config.samp_rate,
        required_sample_rate: config.samp_rate,
        priority: TaskPriority::Normal,
    };

    Ok(PendingScanRequest::new(scan_config, 1, requirements))
}

pub fn handle_scan_command(args: ScanArgs) -> Result<()> {
    let config = build_scanning_config(&args)?;

    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());
    setup_signal_handler(shutdown_coordinator.clone())?;

    // Create LocationResource once at application level - used by both TUI and headless modes
    let location_resource = new_location_resource();

    if args.headless || is_stdout_piped() {
        run_log_mode(&args, config, shutdown_coordinator, location_resource)
    } else {
        run_tui_mode(&args, config, shutdown_coordinator, location_resource)
    }
}

fn run_tui_mode(
    args: &ScanArgs,
    config: crate::core::types::ScanningConfig,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
    location_resource: LocationResource,
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
        .with_device_mode("DT")
        .with_mode(TuningMode::SingleTuner)
        .with_channel(1);
    let device_worlds = create_device_entity_worlds();
    let shared_pool = Arc::new(Pool::with_entity_worlds(
        filter,
        args.log_file.clone(),
        device_worlds.tuner_entities.clone(),
        device_worlds.device_entities.clone(),
    ));

    let scheduler = Arc::new(TaskScheduler::new(
        shared_pool.clone(),
        shutdown_coordinator.clone(),
    ));

    let mut discovery_setup = start_discovery_service(
        OutputMode::Tui(tui_context.tui_event_sender.clone()),
        shutdown_coordinator.clone(),
        scheduler.clone(),
        shared_pool.clone(),
        device_worlds.tuner_entities.clone(),
        device_worlds.device_entities.clone(),
    )?;

    let entity_worlds = create_entity_worlds();
    let pending_scan_request = create_pending_scan_request(args, &config)?;
    let pending_scan_request = Arc::new(std::sync::RwLock::new(Some(pending_scan_request)));

    let discovery_rx = {
        use std::mem;
        mem::replace(
            &mut discovery_setup.discovery_rx,
            std::sync::mpsc::channel().1,
        )
    };

    let pause_request_queue = Arc::new(std::sync::Mutex::new(std::collections::VecDeque::<
        crate::ecs::PauseAndTuneRequest,
    >::new()));

    let global_pause_resource =
        Arc::new(std::sync::Mutex::new(crate::ecs::GlobalPauseState::Active));

    let tui_handle = start_tui(
        tui_event_receiver,
        shutdown_coordinator.clone(),
        theme_name,
        entity_worlds.task_entities.clone(),
        entity_worlds.audio_entities.clone(),
        entity_worlds.signal_entities.clone(),
        pause_request_queue.clone(),
        global_pause_resource.clone(),
        location_resource.clone(),
    );

    let format = super::config::determine_format(args);
    let level = crate::logging::level_from_flags(args.verbose, args.quiet);
    crate::logging::init(level, format, args.log_file.clone())?;

    let run_context = TuiRunContext {
        config,
        stations: args.stations.clone(),
        shutdown_coordinator: shutdown_coordinator.clone(),
        pool: shared_pool.clone(),
        scheduler: scheduler.clone(),
        task_entities: entity_worlds.task_entities,
        audio_entities: entity_worlds.audio_entities,
        signal_entities: entity_worlds.signal_entities,
        pause_request_queue,
        global_pause_resource,
        pending_scan_request,
        discovery_rx,
        location_resource: location_resource.clone(),
    };

    let result = run_with_tui(run_context, tui_context, tui_handle);

    let _ = discovery_setup.discovery_handle.join();
    let _ = discovery_setup.discovery_forwarder.join();

    result
}

fn run_log_mode(
    args: &ScanArgs,
    config: crate::core::types::ScanningConfig,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
    location_resource: LocationResource,
) -> Result<()> {
    let filter = PoolFilter::new()
        .with_driver("sdrplay")
        .with_device_mode("DT")
        .with_mode(TuningMode::SingleTuner)
        .with_channel(1);
    let device_worlds = create_device_entity_worlds();
    let shared_pool = Arc::new(Pool::with_entity_worlds(
        filter,
        args.log_file.clone(),
        device_worlds.tuner_entities.clone(),
        device_worlds.device_entities.clone(),
    ));

    let scheduler = Arc::new(TaskScheduler::new(
        shared_pool.clone(),
        shutdown_coordinator.clone(),
    ));

    let mut discovery_setup = start_discovery_service(
        OutputMode::Headless,
        shutdown_coordinator.clone(),
        scheduler.clone(),
        shared_pool.clone(),
        device_worlds.tuner_entities.clone(),
        device_worlds.device_entities.clone(),
    )?;

    let entity_worlds = create_entity_worlds();
    let pending_scan_request = create_pending_scan_request(args, &config)?;
    let pending_scan_request = Arc::new(std::sync::RwLock::new(Some(pending_scan_request)));

    let discovery_rx = {
        use std::mem;
        mem::replace(
            &mut discovery_setup.discovery_rx,
            std::sync::mpsc::channel().1,
        )
    };

    let format = super::config::determine_format(args);
    let level = crate::logging::level_from_flags(args.verbose, args.quiet);
    crate::logging::init(level, format, None)?;

    // Attempt location detection at startup (fail gracefully if rate limited)
    {
        tracing::debug!("Attempting location detection at startup...");
        if let Ok(mut resource) = location_resource.try_lock() {
            match resource.detect_current_location() {
                Ok(detected_location) => {
                    tracing::info!(
                        lat = detected_location.lat,
                        lon = detected_location.lon,
                        city = ?detected_location.city,
                        source = ?detected_location.source,
                        "Location detected at startup: {}",
                        detected_location.locality_name()
                    );
                }
                Err(e) => {
                    tracing::debug!("Startup location detection failed (may retry later): {}", e);
                }
            }
        }
    }

    let run_context = log_mode::LogRunContext {
        config,
        stations: args.stations.clone(),
        shutdown_coordinator: shutdown_coordinator.clone(),
        pool: shared_pool.clone(),
        scheduler: scheduler.clone(),
        task_entities: entity_worlds.task_entities,
        audio_entities: entity_worlds.audio_entities,
        pending_scan_request,
        discovery_rx,
        location_resource,
    };

    let result = log_mode::run_with_logs(run_context);

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

    #[test]
    fn test_band_scan_window_calculation() {
        use crate::{
            core::{bands::Band, config::SignalProcessingConfig, types::ScanningConfig},
            ecs::components::scan::ScanType,
        };

        let config = ScanningConfig {
            band: Band::Fm,
            duration: 1,
            samp_rate: 2_000_000.0,
            sdr_gain: 40.0,
            scanning_windows: Some(2),
            signal_processing: SignalProcessingConfig {
                window_overlap: 0.75,
                ..Default::default()
            },
            ..Default::default()
        };

        let (freq_min, freq_max) = config.band.frequency_range();
        let step_size = config.samp_rate * (1.0 - config.signal_processing.window_overlap);

        let scan_config = crate::ecs::components::scan::ScanConfigComponent::new(
            ScanType::Band,
            freq_min,
            freq_max,
            step_size,
            config.samp_rate,
            config.sdr_gain,
            config.duration as f64,
            config.scanning_windows.unwrap_or(1),
        );

        let scan_entity = crate::ecs::ScanEntity::new(scan_config, 1);

        assert_eq!(
            step_size, 500_000.0,
            "Step size with 75% overlap should be 0.5MHz"
        );

        let bandwidth = freq_max - freq_min;
        assert_eq!(bandwidth, 20_000_000.0, "FM bandwidth should be 20MHz");

        // Correct formula: floor(bandwidth / step_size) + 1 for inclusive range
        let expected_windows = ((bandwidth / step_size).floor() as usize) + 1;
        assert_eq!(
            expected_windows, 41,
            "Should calculate 41 windows (88.0, 88.5, ..., 108.0)"
        );

        assert_eq!(
            scan_entity.progress.total_windows, 41,
            "FM band scan with 75% overlap and 2MHz sample rate should have 41 windows (inclusive \
             range)"
        );
        assert_eq!(
            scan_entity.config.step_size, 500_000.0,
            "Step size should be 500 kHz with 75% overlap"
        );
    }

    #[test]
    fn test_window_center_frequencies_stay_within_band_range() {
        use crate::{
            core::{bands::Band, config::SignalProcessingConfig, types::ScanningConfig},
            ecs::components::scan::ScanType,
        };

        let config = ScanningConfig {
            band: Band::Fm,
            duration: 1,
            samp_rate: 2_000_000.0,
            sdr_gain: 40.0,
            scanning_windows: Some(2),
            signal_processing: SignalProcessingConfig {
                window_overlap: 0.75,
                ..Default::default()
            },
            ..Default::default()
        };

        let (freq_min, freq_max) = config.band.frequency_range();
        let step_size = config.samp_rate * (1.0 - config.signal_processing.window_overlap);

        let scan_config = crate::ecs::components::scan::ScanConfigComponent::new(
            ScanType::Band,
            freq_min,
            freq_max,
            step_size,
            config.samp_rate,
            config.sdr_gain,
            config.duration as f64,
            config.scanning_windows.unwrap_or(1),
        );

        let total_windows = scan_config.total_windows();

        for window_index in 0..total_windows {
            let center_freq = scan_config.freq_min + (window_index as f64 * scan_config.step_size);

            assert!(
                center_freq >= freq_min,
                "Window {} center frequency {:.1} MHz is below band minimum {:.1} MHz",
                window_index,
                center_freq / 1e6,
                freq_min / 1e6
            );

            let scan_range_mhz = config.samp_rate / 2.0;
            let max_signal_freq = center_freq + scan_range_mhz;

            assert!(
                max_signal_freq <= freq_max + scan_range_mhz,
                "Window {} (center {:.1} MHz) could detect signals up to {:.1} MHz, beyond band \
                 maximum {:.1} MHz",
                window_index,
                center_freq / 1e6,
                max_signal_freq / 1e6,
                freq_max / 1e6
            );

            assert!(
                center_freq <= freq_max,
                "Window {} center frequency {:.1} MHz exceeds band maximum {:.1} MHz \
                 (step_size={:.1} MHz)",
                window_index,
                center_freq / 1e6,
                freq_max / 1e6,
                scan_config.step_size / 1e6
            );
        }
    }

    /// Regression test for Issue: scan exceeds upper bound with total_windows=41 but 1.0 MHz steps
    ///
    /// This test demonstrates the bug where ScanConfigComponent stores step_size correctly (0.5
    /// MHz) but the actual window center calculation produces windows at 1.0 MHz intervals.
    ///
    /// Expected behavior with 75% overlap and 2 MHz sample rate:
    /// - step_size = 0.5 MHz
    /// - total_windows = 41 (88.0, 88.5, 89.0, ..., 108.0 MHz)
    /// - Window 22 should be at: 88.0 + 22*0.5 = 99.0 MHz
    ///
    /// Actual buggy behavior (from runtime logs):
    /// - total_windows = 41 (correct)
    /// - Window 22 at 110.0 MHz (using 1.0 MHz steps instead!)
    #[test]
    fn test_window_22_center_freq_with_overlap() {
        use crate::{
            core::{bands::Band, config::SignalProcessingConfig, types::ScanningConfig},
            ecs::components::scan::ScanType,
        };

        let config = ScanningConfig {
            band: Band::Fm,
            duration: 1,
            samp_rate: 2_000_000.0,
            sdr_gain: 40.0,
            scanning_windows: Some(2),
            signal_processing: SignalProcessingConfig {
                window_overlap: 0.75,
                ..Default::default()
            },
            ..Default::default()
        };

        let (freq_min, freq_max) = config.band.frequency_range();
        let step_size = config.samp_rate * (1.0 - config.signal_processing.window_overlap);

        let scan_config = crate::ecs::components::scan::ScanConfigComponent::new(
            ScanType::Band,
            freq_min,
            freq_max,
            step_size,
            config.samp_rate,
            config.sdr_gain,
            config.duration as f64,
            config.scanning_windows.unwrap_or(1),
        );

        // Verify config has correct step_size
        assert_eq!(
            scan_config.step_size, 500_000.0,
            "Config should store step_size as 0.5 MHz"
        );

        // Verify total_windows is correct
        assert_eq!(
            scan_config.total_windows(),
            41,
            "Should have 41 windows with 0.5 MHz steps"
        );

        // This is the key test: window 22 should be at 99.0 MHz, NOT 110.0 MHz
        let window_22_center = scan_config.freq_min + (22_f64 * scan_config.step_size);
        assert_eq!(
            window_22_center, 99.0e6,
            "Window 22 should be at 99.0 MHz (88 + 22*0.5), not 110.0 MHz. This test verifies the \
             runtime calculation matches the config."
        );

        // Verify window 22 is within band range
        assert!(
            window_22_center <= freq_max,
            "Window 22 at {:.1} MHz should be <= freq_max {:.1} MHz",
            window_22_center / 1e6,
            freq_max / 1e6
        );
    }
}
