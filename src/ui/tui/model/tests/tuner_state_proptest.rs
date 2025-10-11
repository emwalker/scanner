//! Property-based state machine testing for tuner states
//!
//! Uses proptest-state-machine to generate random sequences of tuner state transitions
//! and verify invariants hold across all possible event sequences.

use super::helpers::create_test_pool_status;
use crate::{
    hardware::{DeviceId, DeviceInfo},
    ui::{
        TuiEvent,
        tui::model::{Model, TunerState},
    },
};
use proptest::prelude::*;
use proptest_state_machine::{ReferenceStateMachine, StateMachineTest};
use std::collections::HashMap;

/// Reference state tracking expected tuner states
#[derive(Clone, Debug)]
struct RefState {
    /// Map of device_id -> expected activity
    /// None means tuner hasn't been added yet
    tuners: HashMap<String, Option<RefActivity>>,
}

/// Simplified activity enum for reference state
#[derive(Clone, Debug, PartialEq, Eq)]
enum RefActivity {
    Available,
    Scanning,
    Listening,
}

impl RefActivity {
    fn expected_label(&self) -> &'static str {
        match self {
            RefActivity::Available => "Available",
            RefActivity::Scanning => "Scanning",
            RefActivity::Listening => "Listening",
        }
    }
}

/// State transitions that can occur
#[derive(Clone, Debug)]
enum Transition {
    /// Add a tuner to the model
    AddTuner(String),
    /// Allocate an available tuner for scanning
    AllocateForScanning(String),
    /// Allocate an available tuner for listening
    AllocateForListening(String),
    /// Return an allocated tuner to available state
    ReturnToAvailable(String),
}

/// Reference state machine implementation
struct TunerStateReference;

impl ReferenceStateMachine for TunerStateReference {
    type State = RefState;
    type Transition = Transition;

    fn init_state() -> BoxedStrategy<Self::State> {
        // Start with 1-3 tuners, all added but available
        prop::collection::hash_map(
            "[a-c]", // Generate tuner IDs: "a", "b", "c"
            Just(Some(RefActivity::Available)),
            1..=3,
        )
        .prop_map(|tuners| RefState { tuners })
        .boxed()
    }

    fn transitions(state: &Self::State) -> BoxedStrategy<Self::Transition> {
        let available_tuners: Vec<_> = state
            .tuners
            .iter()
            .filter_map(|(id, activity)| {
                if let Some(RefActivity::Available) = activity {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect();

        let allocated_tuners: Vec<_> = state
            .tuners
            .iter()
            .filter_map(|(id, activity)| match activity {
                Some(RefActivity::Scanning) | Some(RefActivity::Listening) => Some(id.clone()),
                _ => None,
            })
            .collect();

        // Build transition strategies based on current state
        let mut strategies: Vec<BoxedStrategy<Transition>> = Vec::new();

        // Can always add new tuners (up to "f")
        if state.tuners.len() < 6 {
            let next_id = (b'a' + state.tuners.len() as u8) as char;
            strategies.push(Just(Transition::AddTuner(next_id.to_string())).boxed());
        }

        // Can allocate available tuners for scanning
        if !available_tuners.is_empty() {
            strategies.push(
                prop::sample::select(available_tuners.clone())
                    .prop_map(Transition::AllocateForScanning)
                    .boxed(),
            );
        }

        // Can allocate available tuners for listening
        if !available_tuners.is_empty() {
            strategies.push(
                prop::sample::select(available_tuners)
                    .prop_map(Transition::AllocateForListening)
                    .boxed(),
            );
        }

        // Can return allocated tuners to available
        if !allocated_tuners.is_empty() {
            strategies.push(
                prop::sample::select(allocated_tuners)
                    .prop_map(Transition::ReturnToAvailable)
                    .boxed(),
            );
        }

        // If no valid transitions, just add a tuner
        if strategies.is_empty() {
            strategies.push(Just(Transition::AddTuner("x".to_string())).boxed());
        }

        prop::strategy::Union::new(strategies).boxed()
    }

    fn apply(mut state: Self::State, transition: &Self::Transition) -> Self::State {
        match transition {
            Transition::AddTuner(id) => {
                state
                    .tuners
                    .insert(id.clone(), Some(RefActivity::Available));
            }
            Transition::AllocateForScanning(id) => {
                if let Some(activity) = state.tuners.get_mut(id) {
                    *activity = Some(RefActivity::Scanning);
                }
            }
            Transition::AllocateForListening(id) => {
                if let Some(activity) = state.tuners.get_mut(id) {
                    *activity = Some(RefActivity::Listening);
                }
            }
            Transition::ReturnToAvailable(id) => {
                if let Some(activity) = state.tuners.get_mut(id) {
                    *activity = Some(RefActivity::Available);
                }
            }
        }
        state
    }

    fn preconditions(state: &Self::State, transition: &Self::Transition) -> bool {
        match transition {
            Transition::AddTuner(id) => !state.tuners.contains_key(id),
            Transition::AllocateForScanning(id) | Transition::AllocateForListening(id) => {
                matches!(state.tuners.get(id), Some(Some(RefActivity::Available)))
            }
            Transition::ReturnToAvailable(id) => matches!(
                state.tuners.get(id),
                Some(Some(RefActivity::Scanning)) | Some(Some(RefActivity::Listening))
            ),
        }
    }
}

/// System under test implementation
struct TunerStateMachineTest;

impl StateMachineTest for TunerStateMachineTest {
    type SystemUnderTest = Model;
    type Reference = TunerStateReference;

    fn init_test(ref_state: &RefState) -> Self::SystemUnderTest {
        let mut model = Model::default();

        // Add all tuners that exist in reference state
        for (id, activity) in &ref_state.tuners {
            if activity.is_some() {
                let device_id = DeviceId::from_serial("test", id);
                let device_info = DeviceInfo {
                    id: device_id,
                    label: format!("Test Device {}", id),
                };
                model.add_device(device_info);
            }
        }

        // Apply initial allocations
        let (scanning, listening) = extract_allocations(ref_state);
        if !scanning.is_empty() || !listening.is_empty() {
            let all_devices: Vec<_> = ref_state
                .tuners
                .keys()
                .map(|id| DeviceId::from_serial("test", id))
                .collect();

            model.update_tui_event(TuiEvent::ActiveTunersUpdated {
                status: create_test_pool_status(all_devices, scanning, listening),
            });
        }

        model
    }

    fn apply(
        mut state: Self::SystemUnderTest,
        ref_state: &RefState,
        transition: Transition,
    ) -> Self::SystemUnderTest {
        match transition {
            Transition::AddTuner(id) => {
                let device_id = DeviceId::from_serial("test", &id);
                let device_info = DeviceInfo {
                    id: device_id,
                    label: format!("Test Device {}", id),
                };
                state.add_device(device_info);
            }
            Transition::AllocateForScanning(_)
            | Transition::AllocateForListening(_)
            | Transition::ReturnToAvailable(_) => {
                // Apply the full pool status reflecting the new reference state
                let (scanning, listening) = extract_allocations(ref_state);
                let all_devices: Vec<_> = ref_state
                    .tuners
                    .keys()
                    .map(|id| DeviceId::from_serial("test", id))
                    .collect();

                state.update_tui_event(TuiEvent::ActiveTunersUpdated {
                    status: create_test_pool_status(all_devices, scanning, listening),
                });
            }
        }

        state
    }

    fn check_invariants(state: &Self::SystemUnderTest, ref_state: &RefState) {
        let display_states = state.tuner_display_states();

        // Check that every tuner in reference state has correct label in model
        for (id, ref_activity) in &ref_state.tuners {
            if let Some(expected_activity) = ref_activity {
                let device_id = DeviceId::from_serial("test", id);

                // Find this tuner's display state
                let display_state = display_states
                    .iter()
                    .find(|d| d.device_id == device_id)
                    .unwrap_or_else(|| {
                        panic!(
                            "Tuner {} present in reference state but not in model display states",
                            id
                        )
                    });

                // Check label matches expected activity
                let expected_label = expected_activity.expected_label();
                assert_eq!(
                    display_state.status_label,
                    expected_label,
                    "Tuner {} should have label '{}' but has '{}'. Reference: {:?}, Model state: {:?}",
                    id,
                    expected_label,
                    display_state.status_label,
                    ref_activity,
                    state.tuner_state(&device_id)
                );

                // Also check the underlying tuner state enum
                let tuner_state = state.tuner_state(&device_id);
                match expected_activity {
                    RefActivity::Available => assert_eq!(tuner_state, TunerState::Available),
                    RefActivity::Scanning => assert_eq!(tuner_state, TunerState::Scanning),
                    RefActivity::Listening => assert_eq!(tuner_state, TunerState::Listening),
                }
            }
        }

        // Check that model doesn't have extra tuners
        assert_eq!(
            display_states.len(),
            ref_state.tuners.values().filter(|a| a.is_some()).count(),
            "Model has different number of tuners than reference state"
        );
    }
}

/// Extract scanning and listening tuners from reference state
fn extract_allocations(ref_state: &RefState) -> (Vec<DeviceId>, Vec<DeviceId>) {
    let mut scanning = Vec::new();
    let mut listening = Vec::new();

    for (id, activity) in &ref_state.tuners {
        let device_id = DeviceId::from_serial("test", id);
        match activity {
            Some(RefActivity::Scanning) => scanning.push(device_id),
            Some(RefActivity::Listening) => listening.push(device_id),
            _ => {}
        }
    }

    (scanning, listening)
}

proptest_state_machine::prop_state_machine! {
    #[test]
    fn tuner_state_machine_test(sequential 5..=20 => TunerStateMachineTest);
}
