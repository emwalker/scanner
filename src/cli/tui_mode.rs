use crate::core::types::{Result, ScannerError, ScanningConfig};
use crate::hardware::pool::Pool;
use crate::main_thread::MainThread;
use crate::shutdown::ShutdownCoordinator;
use crate::task::TaskScheduler;
use crate::ui::TuiEvent;
use crate::ui::tui::TuiProgressDisplay;
use crate::ui::tui::themes::{ThemeName, create_theme};
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;

pub struct TuiContext {
    pub tui_event_sender: mpsc::Sender<TuiEvent>,
}

pub fn setup_tui_channels() -> (TuiContext, mpsc::Receiver<TuiEvent>) {
    let (tui_event_sender, tui_event_receiver) = mpsc::channel();

    (TuiContext { tui_event_sender }, tui_event_receiver)
}

#[allow(clippy::too_many_arguments)]
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
    pause_request_queue: crate::ecs::Resource<crate::ecs::PauseRequestQueue>,
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
        )
        .with_pause_request_queue(pause_request_queue);
        tui_display.run()
    })
}

pub struct TuiRunContext {
    pub config: ScanningConfig,
    pub stations: Option<String>,
    pub shutdown_coordinator: Arc<ShutdownCoordinator>,
    pub pool: Arc<Pool>,
    pub scheduler: Arc<TaskScheduler>,
    pub scan_entities: Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::ScanEntity>>>,
    pub station_entities:
        Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::StationEntity>>>,
    pub audio_entities: Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::AudioEntity>>>,
    pub candidate_entities:
        Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::CandidateEntity>>>,
    pub pause_request_queue: crate::ecs::Resource<crate::ecs::PauseRequestQueue>,
}

pub fn run_with_tui(
    context: TuiRunContext,
    tui_context: TuiContext,
    tui_handle: thread::JoinHandle<
        std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>,
    >,
) -> Result<()> {
    let backend = Arc::new(crate::hardware::Soapy);

    let window_entities = Arc::new(std::sync::RwLock::new(crate::ecs::EntityWorld::new()));

    let main_thread = MainThread::new_with_entities(
        Arc::new(context.config),
        backend,
        context.shutdown_coordinator.clone(),
        context.pool.clone(),
        context.scheduler,
        Vec::new(),
        context.scan_entities,
        window_entities,
        context.station_entities,
        context.audio_entities,
        context.candidate_entities,
        context.pause_request_queue,
    )?
    .with_tui_event_sender(tui_context.tui_event_sender)
    .start();

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
