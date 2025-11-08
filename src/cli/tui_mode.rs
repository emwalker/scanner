use std::{
    sync::{Arc, mpsc},
    thread,
};

use crate::{
    core::types::{Result, ScannerError, ScanningConfig},
    hardware::pool::Pool,
    main_thread::MainThread,
    shutdown::ShutdownCoordinator,
    task::TaskScheduler,
    ui::{
        TuiEvent,
        tui::{
            TuiProgressDisplay,
            themes::{ThemeName, create_theme},
        },
    },
};

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
    task_entities: Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::TaskEntity>>>,
    audio_entities: Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::AudioEntity>>>,
    signal_entities: Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::SignalEntity>>>,
    pause_request_queue: crate::ecs::Resource<crate::ecs::PauseRequestQueue>,
    global_pause_resource: crate::ecs::GlobalPauseResource,
) -> thread::JoinHandle<std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>> {
    let theme = create_theme(&theme_name);

    thread::spawn(move || {
        let mut tui_display = TuiProgressDisplay::new_with_theme(
            tui_event_receiver,
            shutdown_coordinator.token(),
            theme,
            theme_name,
        )
        .with_entities(task_entities, audio_entities, signal_entities)
        .with_pause_request_queue(pause_request_queue)
        .with_global_pause_resource(global_pause_resource)
        .with_persistence();
        tui_display.run()
    })
}

pub struct TuiRunContext {
    pub config: ScanningConfig,
    pub stations: Option<String>,
    pub shutdown_coordinator: Arc<ShutdownCoordinator>,
    pub pool: Arc<Pool>,
    pub scheduler: Arc<TaskScheduler>,
    pub task_entities: Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::TaskEntity>>>,
    pub audio_entities: Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::AudioEntity>>>,
    pub signal_entities: Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::SignalEntity>>>,
    pub pause_request_queue: crate::ecs::Resource<crate::ecs::PauseRequestQueue>,
    pub global_pause_resource: crate::ecs::GlobalPauseResource,
    pub pending_scan_request:
        Arc<std::sync::RwLock<Option<crate::ecs::components::scan::PendingScanRequest>>>,
    pub discovery_rx: std::sync::mpsc::Receiver<crate::discovery::Event>,
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
        context.task_entities,
        window_entities,
        context.audio_entities,
        context.signal_entities,
        context.pause_request_queue,
        context.global_pause_resource,
        context.pending_scan_request,
        context.discovery_rx,
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
