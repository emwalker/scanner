use crate::core::types::{Result, ScannerError};
use crate::shutdown::ShutdownCoordinator;
use crate::ui::tui::themes::ThemeName;
use std::sync::Arc;
use tracing::debug;

use super::args::ScanArgs;
use super::config::build_scanning_config;
use super::discovery::{initialize_pool_with_device, start_discovery_service};
use super::headless_mode::run_headless;
use super::signals::setup_signal_handler;
use super::tui_mode::{create_logger, run_with_tui, setup_tui_channels, start_tui};

const DEFAULT_DRIVER: &str = "driver=sdrplay";

pub fn handle_scan_command(args: ScanArgs) -> Result<()> {
    let config = build_scanning_config(&args)?;

    let backends = vec![crate::hardware::types::Backend::Soapy];
    let driver_filter = args.device_args.as_deref().or(Some(DEFAULT_DRIVER));
    let discovered_devices = crate::discovery::enumerate_once_subprocess(
        &backends,
        driver_filter,
        args.log_file.clone(),
    )?;

    debug!(
        device_count = discovered_devices.len(),
        "Device enumeration complete"
    );
    for (idx, device) in discovered_devices.iter().enumerate() {
        debug!(
            index = idx,
            device_id = ?device.id,
            label = %device.label,
            "Discovered device"
        );
    }

    if discovered_devices.is_empty() {
        return Err(ScannerError::NoSdrDevicesFound {
            backends: vec!["sdrplay".to_string()],
        });
    }

    let selected_device = &discovered_devices[0];
    // Select first tuner (channel_index 0) of the first device
    let selected_tuner_id = crate::hardware::pool::TunerId::new(selected_device.id.clone(), 0);

    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());
    setup_signal_handler(shutdown_coordinator.clone())?;

    if !args.headless {
        run_tui_mode(
            &args,
            config,
            shutdown_coordinator,
            selected_tuner_id,
            discovered_devices,
        )
    } else {
        run_headless(
            config,
            args.stations.clone(),
            args.verbose,
            args.json,
            args.log,
            args.log_file.clone(),
            shutdown_coordinator,
        )
    }
}

fn run_tui_mode(
    args: &ScanArgs,
    config: crate::core::types::ScanningConfig,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
    selected_tuner_id: crate::hardware::pool::TunerId,
    discovered_devices: Vec<crate::hardware::DeviceInfo>,
) -> Result<()> {
    let (mut tui_context, tui_event_receiver) = setup_tui_channels();

    let theme_name = args
        .theme
        .parse::<ThemeName>()
        .map_err(|e| ScannerError::InvalidTheme {
            theme: args.theme.clone(),
            reason: e.to_string(),
        })?;

    let (command_sender, command_receiver) = std::sync::mpsc::channel();
    tui_context.command_receiver = command_receiver;

    let backend = crate::hardware::types::Backend::Soapy;
    let shared_pool =
        initialize_pool_with_device(&selected_tuner_id, backend, args.log_file.clone())?;

    // Send cached devices to TUI immediately
    for device in &discovered_devices {
        let _ = tui_context
            .tui_event_sender
            .send(crate::ui::TuiEvent::TunerAdded(device.clone()));
    }

    let discovery_setup = start_discovery_service(
        tui_context.tui_event_sender.clone(),
        shutdown_coordinator.clone(),
        discovered_devices.clone(),
        args.log_file.clone(),
    )?;

    debug!(
        cached_device_count = discovered_devices.len(),
        "Passing cached devices to TUI"
    );
    let tui_handle = start_tui(
        tui_event_receiver,
        shutdown_coordinator.clone(),
        theme_name,
        command_sender,
        discovered_devices.clone(),
    );

    let logger = create_logger(args);
    crate::logging::init(logger.as_ref(), args.verbose)?;

    let result = run_with_tui(
        config,
        args.stations.clone(),
        shutdown_coordinator.clone(),
        shared_pool.clone(),
        tui_context,
        tui_handle,
        logger,
        discovered_devices,
    );

    let _ = discovery_setup.discovery_handle.join();
    let _ = discovery_setup.discovery_forwarder.join();

    result
}
