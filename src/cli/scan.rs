use crate::core::types::{Result, ScannerError};
use crate::hardware::Backend;
use crate::hardware::soapy;
use crate::shutdown::ShutdownCoordinator;
use crate::ui::tui::themes::ThemeName;
use std::sync::Arc;

use super::args::ScanArgs;
use super::config::build_scanning_config;
use super::discovery::{initialize_pool_with_device, start_discovery_service};
use super::headless_mode::run_headless;
use super::signals::setup_signal_handler;
use super::tui_mode::{create_logger, run_with_tui, setup_tui_channels, start_tui};

const DEFAULT_DRIVER: &str = "driver=sdrplay";

pub fn handle_scan_command(args: ScanArgs) -> Result<()> {
    let allow_cpp_output = args.verbose && args.headless;
    crate::logging::set_soapysdr_log_level(!allow_cpp_output);

    let config = build_scanning_config(&args)?;

    soapy::reset_soapysdr_state();

    let backends: Vec<Box<dyn Backend>> = vec![Box::new(crate::hardware::Soapy)];
    let driver_filter = args.device_args.as_deref().or(Some(DEFAULT_DRIVER));
    let discovered_devices = crate::discovery::enumerate_once(&backends, driver_filter)?;

    if discovered_devices.is_empty() {
        return Err(ScannerError::HardwareNotAvailable(
            "No SDR devices found".to_string(),
        ));
    }

    let selected_device = &discovered_devices[0];
    let selected_tuner_id = selected_device.id.clone();

    let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());
    setup_signal_handler(shutdown_coordinator.clone());

    if !args.headless {
        run_tui_mode(&args, config, shutdown_coordinator, selected_tuner_id)
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
    selected_tuner_id: crate::hardware::DeviceId,
) -> Result<()> {
    let (mut tui_context, tui_event_receiver) = setup_tui_channels();

    let theme_name = args
        .theme
        .parse::<ThemeName>()
        .map_err(|e| ScannerError::ConfigurationError(format!("Invalid theme: {}", e)))?;

    let (command_sender, command_receiver) = std::sync::mpsc::channel();
    tui_context.command_receiver = command_receiver;

    let backend = crate::hardware::Soapy;
    let shared_pool = initialize_pool_with_device(&selected_tuner_id, &backend)?;

    let discovery_setup = start_discovery_service(
        tui_context.tui_event_sender.clone(),
        shutdown_coordinator.clone(),
    )?;

    let tui_handle = start_tui(
        tui_event_receiver,
        shutdown_coordinator.clone(),
        theme_name,
        command_sender,
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
    );

    let _ = discovery_setup.discovery_handle.join();
    let _ = discovery_setup.discovery_forwarder.join();

    result
}
