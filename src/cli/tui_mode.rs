use crate::core::types::{Logger, Result, ScannerError, ScanningConfig};
use crate::hardware::pool::Pool;
use crate::logging::DefaultLogger;
use crate::main_thread::{DefaultConsoleWriter, MainThread};
use crate::shutdown::ShutdownCoordinator;
use crate::ui::tui::TuiProgressDisplay;
use crate::ui::tui::themes::{ThemeName, create_theme};
use crate::ui::{ChannelProgressReporter, ScannerCommand, TuiEvent};
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;

use super::args::ScanArgs;
use super::config::determine_format;

/// No-op logger that doesn't initialize tracing to suppress all log output
#[derive(Debug)]
struct NoOpLogger;

unsafe impl Send for NoOpLogger {}
unsafe impl Sync for NoOpLogger {}

impl Logger for NoOpLogger {
    fn init(&self) -> Result<()> {
        Ok(())
    }
}

pub struct TuiContext {
    pub tui_event_sender: mpsc::Sender<TuiEvent>,
    pub command_receiver: mpsc::Receiver<ScannerCommand>,
    pub progress_reporter: Arc<ChannelProgressReporter>,
}

pub fn setup_tui_channels() -> (TuiContext, mpsc::Receiver<TuiEvent>) {
    let (tui_event_sender, tui_event_receiver) = mpsc::channel();
    let (progress_sender, progress_receiver) = mpsc::channel();
    let progress_reporter = Arc::new(ChannelProgressReporter::new(progress_sender));

    let tui_event_sender_clone = tui_event_sender.clone();
    thread::spawn(move || {
        while let Ok(event) = progress_receiver.recv() {
            if tui_event_sender_clone
                .send(TuiEvent::Progress(event))
                .is_err()
            {
                break;
            }
        }
    });

    let (_command_sender, command_receiver) = mpsc::channel();

    (
        TuiContext {
            tui_event_sender,
            command_receiver,
            progress_reporter,
        },
        tui_event_receiver,
    )
}

pub fn start_tui(
    tui_event_receiver: mpsc::Receiver<TuiEvent>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
    theme_name: ThemeName,
    command_sender: mpsc::Sender<ScannerCommand>,
    cached_devices: Vec<crate::hardware::DeviceInfo>,
) -> thread::JoinHandle<std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>> {
    let theme = create_theme(&theme_name);

    thread::spawn(move || {
        let mut tui_display = TuiProgressDisplay::new_with_theme(
            tui_event_receiver,
            shutdown_coordinator.token(),
            theme,
            theme_name,
        )
        .with_cached_devices(cached_devices)
        .with_command_sender(command_sender);
        tui_display.run()
    })
}

pub fn create_logger(args: &ScanArgs) -> Arc<dyn Logger + Send + Sync> {
    let format = determine_format(args);

    if args.log_file.is_some() {
        Arc::new(DefaultLogger::new(true, format).with_log_file(args.log_file.clone()))
    } else {
        Arc::new(NoOpLogger)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_with_tui(
    config: ScanningConfig,
    stations: Option<String>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
    shared_pool: Arc<Pool>,
    tui_context: TuiContext,
    tui_handle: thread::JoinHandle<
        std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>,
    >,
    logger: Arc<dyn Logger + Send + Sync>,
    discovered_devices: Vec<crate::hardware::DeviceInfo>,
) -> Result<()> {
    let console_writer = Arc::new(DefaultConsoleWriter);
    let backend = Arc::new(crate::hardware::Soapy);

    let main_thread = MainThread::new_with_progress(
        Arc::new(config),
        console_writer,
        logger,
        backend,
        tui_context.progress_reporter,
        shutdown_coordinator.clone(),
        shared_pool.clone(),
        discovered_devices,
    )?
    .with_command_receiver(tui_context.command_receiver)
    .with_tui_event_sender(tui_context.tui_event_sender);

    let main_handle = thread::spawn(move || main_thread.run(stations));

    let _ = tui_handle.join();
    shutdown_coordinator.shutdown();
    shared_pool.shutdown();

    match main_handle.join() {
        Ok(r) => r?,
        Err(e) => return Err(ScannerError::ThreadJoin(e)),
    }

    Ok(())
}
