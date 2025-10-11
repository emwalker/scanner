use super::helpers::create_test_pool_status;
use crate::hardware::{DeviceId, DeviceInfo};
use crate::ui::tui::model::{Model, TunerState, UiMode};

#[test]
fn test_only_used_tuner_shows_scanning_state() {
    let mut model = Model::default();

    // Discovery service finds RTL-SDR first (alphabetically or by enumeration order)
    let rtlsdr_tuner = DeviceInfo {
        id: DeviceId::from_serial("rtlsdr", "00000001"),
        label: "Generic RTL-SDR".to_string(),
    };
    model.add_device(rtlsdr_tuner.clone());

    // RTL-SDR should be Available, not Scanning
    assert_eq!(
        model.tuner_states.get(&rtlsdr_tuner.id),
        Some(&TunerState::Available),
        "First discovered tuner should be Available, not auto-set to Scanning"
    );

    // Discovery service then finds SDRplay
    let sdrplay_tuner = crate::hardware::DeviceInfo {
        id: DeviceId::from_serial("sdrplay", "2301034E34:ST"),
        label: "SDRplay RSPduo".to_string(),
    };
    model.add_device(sdrplay_tuner.clone());

    // Both should be Available
    assert_eq!(
        model.tuner_states.get(&sdrplay_tuner.id),
        Some(&TunerState::Available)
    );
    assert_eq!(
        model.tuner_states.get(&rtlsdr_tuner.id),
        Some(&TunerState::Available)
    );

    // MainThread starts scan with SDRplay - sends ActiveTunersUpdated event
    model.update_tui_event(crate::ui::TuiEvent::ActiveTunersUpdated {
        status: create_test_pool_status(
            vec![sdrplay_tuner.id.clone()],
            vec![sdrplay_tuner.id.clone()],
            vec![],
        ),
    });

    // SDRplay should now be Scanning
    assert_eq!(
        model.tuner_state(&sdrplay_tuner.id),
        TunerState::Scanning,
        "SDRplay should be Scanning when MainThread allocated it for scanning"
    );

    // RTL-SDR should still be Available (regression test for incorrect auto-scanning)
    assert_eq!(
        model.tuner_state(&rtlsdr_tuner.id),
        TunerState::Available,
        "RTL-SDR should remain Available since it's not in active tuners"
    );

    // Scan continues - active tuners remain unchanged
    // Progress events no longer affect tuner state

    // SDRplay should still be Scanning
    assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Scanning);

    // RTL-SDR should STILL be Available
    assert_eq!(
        model.tuner_state(&rtlsdr_tuner.id),
        TunerState::Available,
        "RTL-SDR should never transition to Scanning since it's not in active tuners"
    );
}

#[test]
fn test_only_used_tuner_shows_listening_state() {
    use crate::hardware::DeviceId;

    let mut model = Model::default();

    // Discovery finds both tuners
    let rtlsdr_tuner = crate::hardware::DeviceInfo {
        id: DeviceId::from_serial("rtlsdr", "00000001"),
        label: "Generic RTL-SDR".to_string(),
    };
    model.add_device(rtlsdr_tuner.clone());

    let sdrplay_tuner = crate::hardware::DeviceInfo {
        id: DeviceId::from_serial("sdrplay", "2301034E34:ST"),
        label: "SDRplay RSPduo".to_string(),
    };
    model.add_device(sdrplay_tuner.clone());

    // Both should be Available initially
    assert_eq!(model.tuner_state(&rtlsdr_tuner.id), TunerState::Available);
    assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Available);

    // MainThread starts scan with SDRplay
    model.update_tui_event(crate::ui::TuiEvent::ActiveTunersUpdated {
        status: create_test_pool_status(
            vec![sdrplay_tuner.id.clone()],
            vec![sdrplay_tuner.id.clone()],
            vec![],
        ),
    });

    // SDRplay is now Scanning
    assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Scanning);

    // User presses Enter to tune to the candidate - MainThread moves tuner to listening
    model.update_tui_event(crate::ui::TuiEvent::ActiveTunersUpdated {
        status: create_test_pool_status(
            vec![sdrplay_tuner.id.clone()],
            vec![],
            vec![sdrplay_tuner.id.clone()],
        ),
    });

    // SDRplay should transition to Listening
    assert_eq!(
        model.tuner_state(&sdrplay_tuner.id),
        TunerState::Listening,
        "SDRplay should be Listening when MainThread allocated it to listening"
    );

    // RTL-SDR should still be Available (regression test for incorrect listening state)
    // The bug was: update_candidate() set self.tuners.first() to Listening
    // instead of using event.tuner_id
    assert_eq!(
        model.tuner_state(&rtlsdr_tuner.id),
        TunerState::Available,
        "RTL-SDR should remain Available since it's not in active tuners"
    );

    // Stop listening doesn't change active tuners
    // (MainThread would send new ActiveTunersUpdated when user presses Escape)
    // For this test, we're just verifying state stays as-is

    // SDRplay remains in Listening state (still allocated to listening)
    assert_eq!(
        model.tuner_state(&sdrplay_tuner.id),
        TunerState::Listening,
        "SDRplay remains Listening until MainThread reallocates it"
    );

    // RTL-SDR should STILL be Available throughout
    assert_eq!(
        model.tuner_state(&rtlsdr_tuner.id),
        TunerState::Available,
        "RTL-SDR should never transition to Listening since it's not in active tuners"
    );
}

#[test]
fn test_tuner_stays_scanning_during_automatic_audio_playback() {
    use crate::hardware::DeviceId;

    let mut model = Model::default();

    // Discovery finds SDRplay tuner
    let sdrplay_tuner = crate::hardware::DeviceInfo {
        id: DeviceId::from_serial("sdrplay", "2301034E34:ST"),
        label: "SDRplay RSPduo".to_string(),
    };
    model.add_device(sdrplay_tuner.clone());

    // Should be Available initially
    assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Available);

    // MainThread allocates SDRplay for scanning
    model.update_tui_event(crate::ui::TuiEvent::ActiveTunersUpdated {
        status: create_test_pool_status(
            vec![sdrplay_tuner.id.clone()],
            vec![sdrplay_tuner.id.clone()],
            vec![],
        ),
    });

    // SDRplay is now Scanning
    assert_eq!(
        model.tuner_state(&sdrplay_tuner.id),
        TunerState::Scanning,
        "Tuner should be Scanning when MainThread allocated it for scanning"
    );

    // Model is still in Idle mode (not AwaitingTune) - user has NOT pressed Enter
    assert!(matches!(model.ui_mode, UiMode::Idle));

    // During scanning, audio playback starts automatically for quality analysis
    // Even though audio is playing, MainThread keeps the tuner in scanning list
    // because user has not pressed Enter (no TuneToCandidate command sent)

    // MainThread continues to report tuner as scanning during automatic playback
    model.update_tui_event(crate::ui::TuiEvent::ActiveTunersUpdated {
        status: create_test_pool_status(
            vec![sdrplay_tuner.id.clone()],
            vec![sdrplay_tuner.id.clone()],
            vec![],
        ),
    });

    // The tuner should remain in Scanning state during automatic audio playback
    // Only when user presses Enter (sends TuneToCandidate) should it go to Listening
    assert_eq!(
        model.tuner_state(&sdrplay_tuner.id),
        TunerState::Scanning,
        "Tuner should remain Scanning during automatic audio playback (user has not pressed Enter)"
    );

    // Audio playback completes automatically, tuner still scanning
    model.update_tui_event(crate::ui::TuiEvent::ActiveTunersUpdated {
        status: create_test_pool_status(
            vec![sdrplay_tuner.id.clone()],
            vec![sdrplay_tuner.id.clone()],
            vec![],
        ),
    });

    // Should still be Scanning after automatic playback completes
    assert_eq!(
        model.tuner_state(&sdrplay_tuner.id),
        TunerState::Scanning,
        "Tuner should remain Scanning after automatic audio playback completes"
    );
}

#[test]
fn test_correct_tuner_shows_scanning_when_returning_from_listening() {
    use crate::hardware::DeviceId;

    let mut model = Model::default();

    // Discovery finds both tuners (RTL-SDR first, SDRplay second)
    let rtlsdr_tuner = crate::hardware::DeviceInfo {
        id: DeviceId::from_serial("rtlsdr", "00000001"),
        label: "Generic RTL-SDR".to_string(),
    };
    model.add_device(rtlsdr_tuner.clone());

    let sdrplay_tuner = crate::hardware::DeviceInfo {
        id: DeviceId::from_serial("sdrplay", "2301034E34:ST"),
        label: "SDRplay RSPduo".to_string(),
    };
    model.add_device(sdrplay_tuner.clone());

    // Both should be Available initially
    assert_eq!(model.tuner_state(&rtlsdr_tuner.id), TunerState::Available);
    assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Available);

    // MainThread allocates SDRplay for scanning (not RTL-SDR)
    model.update_tui_event(crate::ui::TuiEvent::ActiveTunersUpdated {
        status: create_test_pool_status(
            vec![sdrplay_tuner.id.clone()],
            vec![sdrplay_tuner.id.clone()],
            vec![],
        ),
    });

    // SDRplay should be Scanning, RTL-SDR should remain Available
    assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Scanning);
    assert_eq!(model.tuner_state(&rtlsdr_tuner.id), TunerState::Available);

    // User presses Enter to listen - MainThread moves SDRplay to listening list
    model.update_tui_event(crate::ui::TuiEvent::ActiveTunersUpdated {
        status: create_test_pool_status(
            vec![sdrplay_tuner.id.clone()],
            vec![],
            vec![sdrplay_tuner.id.clone()],
        ),
    });

    // SDRplay should be Listening, RTL-SDR should remain Available
    assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Listening);
    assert_eq!(model.tuner_state(&rtlsdr_tuner.id), TunerState::Available);

    // User presses Escape to go back to scanning
    // MainThread moves SDRplay back to scanning list
    model.update_tui_event(crate::ui::TuiEvent::ActiveTunersUpdated {
        status: create_test_pool_status(
            vec![sdrplay_tuner.id.clone()],
            vec![sdrplay_tuner.id.clone()],
            vec![],
        ),
    });

    // After returning from listening to scanning, only SDRplay should be Scanning
    // RTL-SDR should remain Available (never used)
    assert_eq!(
        model.tuner_state(&rtlsdr_tuner.id),
        TunerState::Available,
        "RTL-SDR should remain Available since it's not being used"
    );

    assert_eq!(
        model.tuner_state(&sdrplay_tuner.id),
        TunerState::Scanning,
        "SDRplay should transition back to Scanning when MainThread returns it to scanning list"
    );

    // Verify that exactly one tuner is in Scanning state by checking active_tuners
    if let Some(ref status) = model.pool_status {
        assert_eq!(
            status
                .tuners
                .iter()
                .filter(|t| t.activity == Some(crate::hardware::pool::TunerActivity::Scanning))
                .count(),
            1,
            "Exactly one tuner should be in scanning list"
        );
        assert_eq!(
            status
                .tuners
                .iter()
                .find(|t| t.activity == Some(crate::hardware::pool::TunerActivity::Scanning))
                .unwrap()
                .id
                .device_id
                .clone(),
            sdrplay_tuner.id,
            "Only SDRplay should be in scanning list"
        );
    } else {
        panic!("pool_status should be set");
    }
}

// View model regression tests - verify correct labels are produced for each state

#[test]
fn test_scanning_tuner_displays_scanning_label() {
    use crate::hardware::DeviceId;

    let mut model = Model::default();

    let sdrplay_tuner = crate::hardware::DeviceInfo {
        id: DeviceId::from_serial("sdrplay", "2301034E34:ST"),
        label: "SDRplay RSPduo".to_string(),
    };
    model.add_device(sdrplay_tuner.clone());

    // Allocate tuner for scanning
    model.update_tui_event(crate::ui::TuiEvent::ActiveTunersUpdated {
        status: create_test_pool_status(
            vec![sdrplay_tuner.id.clone()],
            vec![sdrplay_tuner.id.clone()],
            vec![],
        ),
    });

    // Verify tuner state shows Scanning
    assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Scanning);

    // Verify view model produces "Scanning" label
    let display_states = model.tuner_display_states();
    assert_eq!(display_states.len(), 1);
    assert_eq!(display_states[0].device_id, sdrplay_tuner.id);
    assert_eq!(
        display_states[0].status_label, "Scanning",
        "Scanning tuner must display 'Scanning' label"
    );
}

#[test]
fn test_listening_tuner_displays_listening_label() {
    use crate::hardware::DeviceId;

    let mut model = Model::default();

    let sdrplay_tuner = crate::hardware::DeviceInfo {
        id: DeviceId::from_serial("sdrplay", "2301034E34:ST"),
        label: "SDRplay RSPduo".to_string(),
    };
    model.add_device(sdrplay_tuner.clone());

    // Allocate tuner for listening
    model.update_tui_event(crate::ui::TuiEvent::ActiveTunersUpdated {
        status: create_test_pool_status(
            vec![sdrplay_tuner.id.clone()],
            vec![],
            vec![sdrplay_tuner.id.clone()],
        ),
    });

    // Verify tuner state shows Listening
    assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Listening);

    // Verify view model produces "Listening" label
    let display_states = model.tuner_display_states();
    assert_eq!(display_states.len(), 1);
    assert_eq!(display_states[0].device_id, sdrplay_tuner.id);
    assert_eq!(
        display_states[0].status_label, "Listening",
        "Listening tuner must display 'Listening' label"
    );
}

#[test]
fn test_available_tuner_displays_available_label() {
    use crate::hardware::DeviceId;

    let mut model = Model::default();

    let sdrplay_tuner = crate::hardware::DeviceInfo {
        id: DeviceId::from_serial("sdrplay", "2301034E34:ST"),
        label: "SDRplay RSPduo".to_string(),
    };
    model.add_device(sdrplay_tuner.clone());

    // Tuner is discovered but not allocated
    assert_eq!(model.tuner_state(&sdrplay_tuner.id), TunerState::Available);

    // Verify view model produces "Available" label
    let display_states = model.tuner_display_states();
    assert_eq!(display_states.len(), 1);
    assert_eq!(display_states[0].device_id, sdrplay_tuner.id);
    assert_eq!(
        display_states[0].status_label, "Available",
        "Available tuner must display 'Available' label"
    );
}

#[test]
fn test_state_transition_updates_label() {
    use crate::hardware::DeviceId;

    let mut model = Model::default();

    let sdrplay_tuner = crate::hardware::DeviceInfo {
        id: DeviceId::from_serial("sdrplay", "2301034E34:ST"),
        label: "SDRplay RSPduo".to_string(),
    };
    model.add_device(sdrplay_tuner.clone());

    // Initially Available
    let display_states = model.tuner_display_states();
    assert_eq!(display_states[0].status_label, "Available");

    // Transition to Scanning
    model.update_tui_event(crate::ui::TuiEvent::ActiveTunersUpdated {
        status: create_test_pool_status(
            vec![sdrplay_tuner.id.clone()],
            vec![sdrplay_tuner.id.clone()],
            vec![],
        ),
    });

    let display_states = model.tuner_display_states();
    assert_eq!(
        display_states[0].status_label, "Scanning",
        "Label must update to 'Scanning' after transition"
    );

    // Transition to Listening
    model.update_tui_event(crate::ui::TuiEvent::ActiveTunersUpdated {
        status: create_test_pool_status(
            vec![sdrplay_tuner.id.clone()],
            vec![],
            vec![sdrplay_tuner.id.clone()],
        ),
    });

    let display_states = model.tuner_display_states();
    assert_eq!(
        display_states[0].status_label, "Listening",
        "Label must update to 'Listening' after transition"
    );

    // Transition back to Available
    model.update_tui_event(crate::ui::TuiEvent::ActiveTunersUpdated {
        status: create_test_pool_status(vec![sdrplay_tuner.id.clone()], vec![], vec![]),
    });

    let display_states = model.tuner_display_states();
    assert_eq!(
        display_states[0].status_label, "Available",
        "Label must update to 'Available' after transition"
    );
}

#[test]
fn test_multiple_tuners_show_correct_individual_labels() {
    use crate::hardware::DeviceId;

    let mut model = Model::default();

    let rtlsdr_tuner = crate::hardware::DeviceInfo {
        id: DeviceId::from_serial("rtlsdr", "00000001"),
        label: "Generic RTL-SDR".to_string(),
    };
    model.add_device(rtlsdr_tuner.clone());

    let sdrplay_tuner = crate::hardware::DeviceInfo {
        id: DeviceId::from_serial("sdrplay", "2301034E34:ST"),
        label: "SDRplay RSPduo".to_string(),
    };
    model.add_device(sdrplay_tuner.clone());

    // SDRplay is scanning, RTL-SDR is available
    model.update_tui_event(crate::ui::TuiEvent::ActiveTunersUpdated {
        status: create_test_pool_status(
            vec![sdrplay_tuner.id.clone()],
            vec![sdrplay_tuner.id.clone()],
            vec![],
        ),
    });

    let display_states = model.tuner_display_states();
    assert_eq!(display_states.len(), 2);

    // Find each tuner's display state
    let rtlsdr_display = display_states
        .iter()
        .find(|d| d.device_id == rtlsdr_tuner.id)
        .expect("RTL-SDR should be in display states");
    let sdrplay_display = display_states
        .iter()
        .find(|d| d.device_id == sdrplay_tuner.id)
        .expect("SDRplay should be in display states");

    assert_eq!(
        rtlsdr_display.status_label, "Available",
        "RTL-SDR should show Available"
    );
    assert_eq!(
        sdrplay_display.status_label, "Scanning",
        "SDRplay should show Scanning"
    );
}
