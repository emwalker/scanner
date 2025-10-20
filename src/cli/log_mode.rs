use crate::core::types::{Result, ScannerError, ScanningConfig};
use crate::hardware::pool::Pool;
use crate::main_thread::MainThread;
use crate::shutdown::ShutdownCoordinator;
use crate::task::TaskScheduler;
use std::sync::Arc;
use std::thread;

pub struct LogRunContext {
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
}

pub fn run_with_logs(context: LogRunContext) -> Result<()> {
    let backend = Arc::new(crate::hardware::Soapy);

    let window_entities = Arc::new(std::sync::RwLock::new(crate::ecs::EntityWorld::new()));

    let pause_request_queue = Arc::new(std::sync::Mutex::new(std::collections::VecDeque::<
        crate::ecs::PauseRequest,
    >::new()));

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
        pause_request_queue,
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
    use super::*;
    use crate::audio::quality::AudioAnalyzer;
    use crate::core::types::ScanningConfig;
    use crate::ecs::components::scan::{ScanConfigComponent, ScanType};
    use crate::ecs::{EntityWorld, ScanEntity};
    use crate::hardware::pool::{PoolFilter, TuningMode};
    use std::sync::RwLock;
    use std::time::Duration;

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
        let pool = Arc::new(Pool::new(filter, None));
        let scheduler = Arc::new(TaskScheduler::new(
            pool.clone(),
            shutdown_coordinator.clone(),
        ));

        let scan_entities = Arc::new(RwLock::new(EntityWorld::<ScanEntity>::new()));
        let station_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let audio_entities = Arc::new(RwLock::new(EntityWorld::new()));
        let candidate_entities = Arc::new(RwLock::new(EntityWorld::new()));

        let scan_config = ScanConfigComponent::new(
            ScanType::Stations,
            88.9e6,
            88.9e6,
            2_000_000.0,
            2_000_000.0,
            24.0,
            1.0,
            1,
        )
        .with_stations(vec![88.9e6]);

        let scan_entity = ScanEntity::new(scan_config);
        scan_entities.write().unwrap().insert(scan_entity);

        let context = LogRunContext {
            config,
            stations: Some("88.9e6".to_string()),
            shutdown_coordinator: shutdown_coordinator.clone(),
            pool: pool.clone(),
            scheduler,
            scan_entities: scan_entities.clone(),
            station_entities,
            audio_entities,
            candidate_entities,
        };

        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            shutdown_coordinator.shutdown();
        });

        let result = run_with_logs(context);
        assert!(result.is_ok(), "Log mode should complete successfully");

        let entities = scan_entities.read().unwrap();
        let scan = entities.iter().next().unwrap();
        assert!(
            !scan.is_pending(),
            "Scan should have been processed by coordinator (not Pending)"
        );
    }
}
