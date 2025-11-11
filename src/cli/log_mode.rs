use std::{sync::Arc, thread};

use crate::{
    core::types::{Result, ScannerError, ScanningConfig},
    ecs::resources::LocationResource,
    hardware::pool::Pool,
    main_thread::MainThread,
    shutdown::ShutdownCoordinator,
    task::TaskScheduler,
};

pub struct LogRunContext {
    pub config: ScanningConfig,
    pub stations: Option<String>,
    pub shutdown_coordinator: Arc<ShutdownCoordinator>,
    pub pool: Arc<Pool>,
    pub scheduler: Arc<TaskScheduler>,
    pub task_entities: Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::TaskEntity>>>,
    pub audio_entities: Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::AudioEntity>>>,
    pub pending_scan_request:
        Arc<std::sync::RwLock<Option<crate::ecs::components::scan::PendingScanRequest>>>,
    pub discovery_rx: std::sync::mpsc::Receiver<crate::discovery::Event>,
    pub location_resource: LocationResource,
}

pub fn run_with_logs(context: LogRunContext) -> Result<()> {
    let backend = Arc::new(crate::hardware::Soapy);

    let window_entities = Arc::new(std::sync::RwLock::new(crate::ecs::EntityWorld::new()));
    let signal_entities = Arc::new(std::sync::RwLock::new(crate::ecs::EntityWorld::new()));

    let pause_request_queue = Arc::new(std::sync::Mutex::new(std::collections::VecDeque::<
        crate::ecs::PauseAndTuneRequest,
    >::new()));

    let global_pause_resource =
        Arc::new(std::sync::Mutex::new(crate::ecs::GlobalPauseState::Active));

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
        signal_entities,
        pause_request_queue,
        global_pause_resource,
        context.pending_scan_request,
        context.discovery_rx,
        context.location_resource,
    )?
    .start();

    let main_handle = thread::spawn(move || main_thread.run(context.stations));

    match main_handle.join() {
        Ok(r) => r?,
        Err(e) => return Err(ScannerError::ThreadJoin(e)),
    }

    context.shutdown_coordinator.shutdown();
    context.pool.shutdown();

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{sync::RwLock, time::Duration};

    use super::*;
    use crate::{
        audio::quality::AudioAnalyzer,
        core::types::ScanningConfig,
        ecs::{EntityWorld, test_helpers::create_test_pool_with_entities},
        hardware::pool::{PoolFilter, TuningMode},
    };

    #[test]
    fn test_log_mode_starts_coordinator() {
        let mut config = ScanningConfig::default();
        config.audio.buffer_size = 8192;
        config.audio.analyzer = AudioAnalyzer::mock();
        config.scanning_windows = Some(1);
        config.duration = 1;

        let shutdown_coordinator = Arc::new(ShutdownCoordinator::new());

        let filter = PoolFilter::new()
            .with_driver("mock")
            .with_mode(TuningMode::SingleTuner);
        let (pool, _tuner_entities, _device_entities) =
            create_test_pool_with_entities(filter, None);
        let scheduler = Arc::new(TaskScheduler::new(
            pool.clone(),
            shutdown_coordinator.clone(),
        ));

        let task_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let audio_entities = Arc::new(RwLock::new(EntityWorld::new()));

        let scan_config = crate::ecs::components::scan::ScanConfigComponent::new(
            crate::ecs::components::scan::ScanType::Stations,
            88.9e6,
            88.9e6,
            2_000_000.0,
            2_000_000.0,
            24.0,
            1.0,
            1,
        )
        .with_stations(vec![88.9e6]);

        let requirements = crate::hardware::pool::TaskRequirements {
            frequency_hz: 88.9e6,
            bandwidth_hz: 2_000_000.0,
            required_sample_rate: 2_000_000.0,
            priority: crate::hardware::pool::TaskPriority::Normal,
        };

        let pending_scan_request = Arc::new(std::sync::RwLock::new(Some(
            crate::ecs::components::scan::PendingScanRequest::new(scan_config, 1, requirements),
        )));

        let (_discovery_tx, discovery_rx) = std::sync::mpsc::channel();

        let location_resource = crate::ecs::resources::new_location_resource();

        let context = LogRunContext {
            config,
            stations: Some("88.9e6".to_string()),
            shutdown_coordinator: shutdown_coordinator.clone(),
            pool: pool.clone(),
            scheduler,
            task_entities: task_entities.clone(),
            audio_entities,
            pending_scan_request,
            discovery_rx,
            location_resource,
        };

        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            shutdown_coordinator.shutdown();
        });

        let result = run_with_logs(context);
        assert!(result.is_ok(), "Log mode should complete successfully");
    }
}
