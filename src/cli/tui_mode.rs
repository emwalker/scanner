use crate::core::types::{Logger, Result, ScannerError, ScanningConfig};
use crate::hardware::pool::Pool;
use crate::logging::DefaultLogger;
use crate::main_thread::{DefaultConsoleWriter, MainThread};
use crate::shutdown::ShutdownCoordinator;
use crate::task::TaskScheduler;
use crate::ui::TuiEvent;
use crate::ui::tui::TuiProgressDisplay;
use crate::ui::tui::themes::{ThemeName, create_theme};
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
}

pub fn setup_tui_channels() -> (TuiContext, mpsc::Receiver<TuiEvent>) {
    let (tui_event_sender, tui_event_receiver) = mpsc::channel();

    (TuiContext { tui_event_sender }, tui_event_receiver)
}

pub fn start_tui(
    tui_event_receiver: mpsc::Receiver<TuiEvent>,
    shutdown_coordinator: Arc<ShutdownCoordinator>,
    theme_name: ThemeName,
    scan_entities: Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::ScanEntity>>>,
    station_entities: Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::StationEntity>>>,
    audio_entities: Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::AudioEntity>>>,
    candidate_entities: Arc<
        std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::CandidateEntity>>,
    >,
) -> thread::JoinHandle<std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>> {
    let theme = create_theme(&theme_name);

    thread::spawn(move || {
        let mut tui_display = TuiProgressDisplay::new_with_theme(
            tui_event_receiver,
            shutdown_coordinator.token(),
            theme,
            theme_name,
        )
        .with_entities(
            scan_entities,
            station_entities,
            audio_entities,
            candidate_entities,
        );
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

pub struct TuiRunContext {
    pub config: ScanningConfig,
    pub stations: Option<String>,
    pub shutdown_coordinator: Arc<ShutdownCoordinator>,
    pub pool: Arc<Pool>,
    pub scheduler: Arc<TaskScheduler>,
    pub logger: Arc<dyn Logger + Send + Sync>,
    pub scan_entities: Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::ScanEntity>>>,
    pub station_entities:
        Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::StationEntity>>>,
    pub audio_entities: Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::AudioEntity>>>,
    pub candidate_entities:
        Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::CandidateEntity>>>,
}

pub fn run_with_tui(
    context: TuiRunContext,
    tui_context: TuiContext,
    tui_handle: thread::JoinHandle<
        std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>,
    >,
) -> Result<()> {
    let console_writer = Arc::new(DefaultConsoleWriter);
    let backend = Arc::new(crate::hardware::Soapy);

    let main_thread = MainThread::new_with_entities(
        Arc::new(context.config),
        console_writer,
        context.logger,
        backend,
        context.shutdown_coordinator.clone(),
        context.pool.clone(),
        context.scheduler,
        Vec::new(),
        context.scan_entities,
        context.station_entities,
        context.audio_entities,
        context.candidate_entities,
    )?
    .with_tui_event_sender(tui_context.tui_event_sender);

    let main_handle = thread::spawn(move || main_thread.run(context.stations));

    let _ = tui_handle.join();
    context.shutdown_coordinator.shutdown();
    context.pool.shutdown();

    match main_handle.join() {
        Ok(r) => r?,
        Err(e) => return Err(ScannerError::ThreadJoin(e)),
    }

    Ok(())
}
