use std::time::Instant;

use crate::ecs::components::SignalId;

#[derive(Debug, Clone)]
pub struct SignalsTableComponent {
    pub entries: Vec<SignalTableEntry>,
    pub last_update_generation: u64,
}

#[derive(Debug, Clone)]
pub struct SignalTableEntry {
    pub signal_id: SignalId,
    pub sort_order: usize,
    pub last_updated: Instant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignalActivity {
    Playing,
    NotPlaying,
}

impl SignalsTableComponent {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            last_update_generation: 0,
        }
    }
}

impl Default for SignalsTableComponent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::signals::ModulationType;

    fn create_test_signal_id() -> SignalId {
        SignalId::new(88.9e6, ModulationType::WFM)
    }

    #[test]
    fn test_signals_table_component_new() {
        let component = SignalsTableComponent::new();
        assert!(component.entries.is_empty());
        assert_eq!(component.last_update_generation, 0);
    }

    #[test]
    fn test_signal_table_entry_creation() {
        let signal_id = create_test_signal_id();
        let entry = SignalTableEntry {
            signal_id: signal_id.clone(),
            sort_order: 0,
            last_updated: std::time::Instant::now(),
        };

        assert_eq!(entry.signal_id, signal_id);
        assert_eq!(entry.sort_order, 0);
    }
}
