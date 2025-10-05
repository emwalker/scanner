use crate::sdr;
use std::collections::HashMap;

pub fn detect_changes(
    known: &HashMap<sdr::TunerId, sdr::TunerInfo>,
    current: &HashMap<sdr::TunerId, sdr::TunerInfo>,
) -> (Vec<sdr::TunerInfo>, Vec<sdr::TunerId>) {
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
    use crate::sdr::{TunerId, TunerInfo};

    #[test]
    fn test_detect_no_changes() {
        let mut devices = HashMap::new();
        devices.insert(
            TunerId::from_serial("test", "001"),
            TunerInfo {
                id: TunerId::from_serial("test", "001"),
                label: "Test Device".to_string(),
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
            TunerId::from_serial("test", "001"),
            TunerInfo {
                id: TunerId::from_serial("test", "001"),
                label: "Test Device 1".to_string(),
            },
        );
        current.insert(
            TunerId::from_serial("test", "002"),
            TunerInfo {
                id: TunerId::from_serial("test", "002"),
                label: "Test Device 2".to_string(),
            },
        );

        let (added, removed) = detect_changes(&known, &current);
        assert_eq!(added.len(), 2);
        assert!(removed.is_empty());
        assert_eq!(added[0].id, TunerId::from_serial("test", "001"));
        assert_eq!(added[1].id, TunerId::from_serial("test", "002"));
    }

    #[test]
    fn test_detect_removed_devices() {
        let mut known = HashMap::new();
        known.insert(
            TunerId::from_serial("test", "001"),
            TunerInfo {
                id: TunerId::from_serial("test", "001"),
                label: "Test Device 1".to_string(),
            },
        );
        known.insert(
            TunerId::from_serial("test", "002"),
            TunerInfo {
                id: TunerId::from_serial("test", "002"),
                label: "Test Device 2".to_string(),
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
            let id = TunerId::from_serial("test", &format!("{:03}", i));
            current.insert(
                id.clone(),
                TunerInfo {
                    id: id.clone(),
                    label: format!("Device {}", i),
                },
            );
        }

        let (added, _) = detect_changes(&known, &current);

        for i in 0..added.len() - 1 {
            assert!(added[i].id < added[i + 1].id, "Results should be sorted");
        }
    }
}
