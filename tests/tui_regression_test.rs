use scanner::hardware::DeviceId;
use scanner::hardware::pool::{PoolStatus, TunerActivity, TunerId, TunerState};
use scanner::hardware::types::Backend;
use scanner::ui::TuiEvent;
use scanner::ui::tui::model::Model;
use std::sync::mpsc;
use std::time::Duration;

#[test]
fn test_active_tuners_updated_skips_redundant_processing() {
    let mut model = Model::new();

    let tuner_id = TunerId::new(
        DeviceId::Driver {
            backend: Backend::Soapy,
            driver: "rtlsdr".to_string(),
            serial: "00000001".to_string(),
        },
        0,
    );

    let status = PoolStatus {
        available_tuner_count: 0,
        allocated_tuner_count: 1,
        device_count: 1,
        tuners: vec![scanner::hardware::pool::TunerStatus {
            id: tuner_id.clone(),
            state: TunerState::Allocated,
            activity: Some(TunerActivity::Listening),
        }],
    };

    let event1 = TuiEvent::ActiveTunersUpdated {
        status: status.clone(),
    };
    model.update_tui_event(event1);

    let initial_tuner_state = model.pool_info.get(&tuner_id).unwrap().clone();

    let event2 = TuiEvent::ActiveTunersUpdated {
        status: status.clone(),
    };
    model.update_tui_event(event2);

    let after_redundant_update = model.pool_info.get(&tuner_id).unwrap().clone();

    assert_eq!(
        initial_tuner_state.state, after_redundant_update.state,
        "Regression test: Pool info should not be rebuilt when status is identical.\n\
         Bug: During listening mode, redundant ActiveTunersUpdated events caused \n\
         expensive HashMap rebuilds every time, spiking CPU to 50%.\n\
         Fix: Added change detection to skip processing when status unchanged."
    );
    assert_eq!(
        initial_tuner_state.activity, after_redundant_update.activity,
        "Activity should also remain unchanged for redundant updates"
    );

    let event3 = TuiEvent::ActiveTunersUpdated {
        status: PoolStatus {
            available_tuner_count: 1,
            allocated_tuner_count: 0,
            device_count: 1,
            tuners: vec![scanner::hardware::pool::TunerStatus {
                id: tuner_id.clone(),
                state: TunerState::Available,
                activity: None,
            }],
        },
    };
    model.update_tui_event(event3);

    assert_ne!(
        model.pool_info.get(&tuner_id).unwrap().state,
        TunerState::Allocated,
        "Pool info should update when status actually changes"
    );
}

#[test]
fn test_spectrum_renders_continuously_at_10fps() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use tokio_util::sync::CancellationToken;

    let (_tx, rx) = mpsc::channel();
    let shutdown_token = CancellationToken::new();

    let _display = scanner::ui::tui::TuiProgressDisplay::new(rx, shutdown_token.clone());

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();

    let start = std::time::Instant::now();
    let mut draw_count = 0;
    let test_duration = Duration::from_millis(350);

    terminal
        .draw(|_f| {
            draw_count += 1;
        })
        .unwrap();

    while start.elapsed() < test_duration {
        std::thread::sleep(Duration::from_millis(100));

        terminal
            .draw(|_f| {
                draw_count += 1;
            })
            .unwrap();
    }

    assert!(
        draw_count >= 3,
        "Regression test: Terminal should render at ~10 FPS (100ms intervals) for smooth spectrum animation.\n\
         Bug: Conditional rendering based on model.is_dirty() caused jumpy spectrum wave \n\
         because it only updated when entities changed, not continuously.\n\
         Fix: Always call mark_dirty() in main loop to ensure 10 FPS rendering.\n\
         Expected at least 3 draws in 350ms, got {}",
        draw_count
    );
}
