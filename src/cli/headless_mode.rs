use crate::core::types::{Format, Result, ScannerError, ScanningConfig};
use crate::logging::DefaultLogger;
use crate::main_thread::{DefaultConsoleWriter, MainThread};
use crate::shutdown::ShutdownCoordinator;
use std::sync::Arc;
use std::thread;

pub fn run_headless(
    config: ScanningConfig,
    stations: Option<String>,
    verbose: bool,
    json: bool,
    log: bool,
    log_file: Option<String>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
) -> Result<()> {
    let format = if json {
        Format::Json
    } else if log {
        Format::Log
    } else {
        Format::Text
    };

    let logger = Arc::new(DefaultLogger::new(verbose, format).with_log_file(log_file));

    crate::logging::init(logger.as_ref(), verbose)?;

    let console_writer = Arc::new(DefaultConsoleWriter);
    let backend = Arc::new(crate::hardware::Soapy);

    let main_thread = MainThread::new(
        Arc::new(config),
        console_writer,
        logger,
        backend,
        shutdown_coordinator.clone(),
    )?;

    let main_handle = thread::spawn(move || main_thread.run(stations));

    let result = main_handle.join();
    match result {
        Ok(r) => {
            shutdown_coordinator.shutdown();
            r?
        }
        Err(e) => return Err(ScannerError::ThreadJoin(e)),
    }

    Ok(())
}
