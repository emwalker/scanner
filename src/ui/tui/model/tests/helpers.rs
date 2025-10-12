use crate::hardware::pool::{PoolStatus, TunerActivity, TunerId, TunerState, TunerStatus};

pub fn create_test_pool_status(
    available: Vec<crate::hardware::DeviceId>,
    scanning: Vec<crate::hardware::DeviceId>,
    listening: Vec<crate::hardware::DeviceId>,
) -> PoolStatus {
    let mut tuners = Vec::new();

    for device_id in available.iter() {
        let is_scanning = scanning.contains(device_id);
        let is_listening = listening.contains(device_id);

        let (state, activity) = if is_scanning {
            (TunerState::Allocated, Some(TunerActivity::Scanning))
        } else if is_listening {
            (TunerState::Allocated, Some(TunerActivity::Listening))
        } else {
            (TunerState::Available, None)
        };

        tuners.push(TunerStatus {
            id: TunerId {
                device_id: device_id.clone(),
                channel_index: 0,
            },
            state,
            activity,
        });
    }

    let available_count = available
        .iter()
        .filter(|id| !scanning.contains(id) && !listening.contains(id))
        .count();
    let allocated_count = scanning.len() + listening.len();

    PoolStatus {
        tuners,
        available_tuner_count: available_count,
        allocated_tuner_count: allocated_count,
        device_count: available.len(),
    }
}
