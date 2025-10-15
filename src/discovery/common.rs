#![allow(dead_code)]

use crate::hardware;
use std::collections::HashMap;

pub fn detect_changes(
    known: &HashMap<hardware::DeviceId, hardware::DeviceInfo>,
    current: &HashMap<hardware::DeviceId, hardware::DeviceInfo>,
) -> (Vec<hardware::DeviceInfo>, Vec<hardware::DeviceId>) {
    let mut added: Vec<_> = current
        .iter()
        .filter(|(id, _)| !known.contains_key(id))
        .map(|(_, device)| device.clone())
        .collect();
    added.sort_by(|a, b| a.id.cmp(&b.id));

    let mut removed: Vec<_> = known
        .keys()
        .filter(|id| !current.contains_key(id))
        .cloned()
        .collect();
    removed.sort();

    (added, removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{DeviceId, DeviceInfo};

    #[test]
    fn test_detect_no_changes() {
        let mut devices = HashMap::new();
        devices.insert(
            DeviceId::from_serial("test", "001"),
            DeviceInfo {
                id: DeviceId::from_serial("test", "001"),
                label: "Test Device".to_string(),
                tuners: vec![crate::hardware::types::TunerInfo {
                    label: "Test Device".to_string(),
                    mode: String::new(),
                    id: crate::hardware::pool::TunerId::new(
                        DeviceId::from_serial("test", "001"),
                        0,
                    ),
                }],
            },
        );

        let (added, removed) = detect_changes(&devices, &devices);
        assert!(added.is_empty());
        assert!(removed.is_empty());
    }

    #[test]
    fn test_detect_added_devices() {
        let known = HashMap::new();
        let mut current = HashMap::new();
        current.insert(
            DeviceId::from_serial("test", "001"),
            DeviceInfo {
                id: DeviceId::from_serial("test", "001"),
                label: "Test Device 1".to_string(),
                tuners: vec![crate::hardware::types::TunerInfo {
                    label: "Test Device 1".to_string(),
                    mode: String::new(),
                    id: crate::hardware::pool::TunerId::new(
                        DeviceId::from_serial("test", "001"),
                        0,
                    ),
                }],
            },
        );
        current.insert(
            DeviceId::from_serial("test", "002"),
            DeviceInfo {
                id: DeviceId::from_serial("test", "002"),
                label: "Test Device 2".to_string(),
                tuners: vec![crate::hardware::types::TunerInfo {
                    label: "Test Device 2".to_string(),
                    mode: String::new(),
                    id: crate::hardware::pool::TunerId::new(
                        DeviceId::from_serial("test", "002"),
                        0,
                    ),
                }],
            },
        );

        let (added, removed) = detect_changes(&known, &current);
        assert_eq!(added.len(), 2);
        assert!(removed.is_empty());
        assert_eq!(added[0].id, DeviceId::from_serial("test", "001"));
        assert_eq!(added[1].id, DeviceId::from_serial("test", "002"));
    }

    #[test]
    fn test_detect_removed_devices() {
        let mut known = HashMap::new();
        known.insert(
            DeviceId::from_serial("test", "001"),
            DeviceInfo {
                id: DeviceId::from_serial("test", "001"),
                label: "Test Device 1".to_string(),
                tuners: vec![crate::hardware::types::TunerInfo {
                    label: "Test Device 1".to_string(),
                    mode: String::new(),
                    id: crate::hardware::pool::TunerId::new(
                        DeviceId::from_serial("test", "001"),
                        0,
                    ),
                }],
            },
        );
        known.insert(
            DeviceId::from_serial("test", "002"),
            DeviceInfo {
                id: DeviceId::from_serial("test", "002"),
                label: "Test Device 2".to_string(),
                tuners: vec![crate::hardware::types::TunerInfo {
                    label: "Test Device 2".to_string(),
                    mode: String::new(),
                    id: crate::hardware::pool::TunerId::new(
                        DeviceId::from_serial("test", "002"),
                        0,
                    ),
                }],
            },
        );

        let current = HashMap::new();

        let (added, removed) = detect_changes(&known, &current);
        assert!(added.is_empty());
        assert_eq!(removed.len(), 2);
    }

    #[test]
    fn test_detect_changes_deterministic_ordering() {
        let known = HashMap::new();
        let mut current = HashMap::new();

        for i in (0..10).rev() {
            let id = DeviceId::from_serial("test", &format!("{:03}", i));
            current.insert(
                id.clone(),
                DeviceInfo {
                    id: id.clone(),
                    label: format!("Device {}", i),
                    tuners: vec![crate::hardware::types::TunerInfo {
                        id: crate::hardware::pool::TunerId::new(id.clone(), 0),
                        label: format!("Device {}", i),
                        mode: String::new(),
                    }],
                },
            );
        }

        let (added, _) = detect_changes(&known, &current);

        for i in 0..added.len() - 1 {
            assert!(added[i].id < added[i + 1].id, "Results should be sorted");
        }
    }
}
