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

    let tui_handle = start_tui(
        tui_event_receiver,
        shutdown_coordinator.clone(),
        theme_name,
        command_sender,
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
    };

    let result = run_with_tui(run_context, tui_context, tui_handle);

    let _ = discovery_setup.discovery_handle.join();
    let _ = discovery_setup.discovery_forwarder.join();

    result
}
