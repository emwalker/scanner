//! Display name component - human-readable label for tuners

/// Component containing the display name for a tuner entity
///
/// This component stores the human-readable name used to identify a tuner
/// in the UI, such as "SDRplay RSPduo" or "RTL2832 Tuner 1".
#[derive(Debug, Clone)]
pub struct DisplayNameComponent {
    /// The human-readable display name for this tuner
    pub name: String,
}

impl DisplayNameComponent {
    /// Create a new display name component
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_display_name() {
        let name = DisplayNameComponent::new("SDRplay RSPduo".to_string());
        assert_eq!(name.name, "SDRplay RSPduo");
    }

    #[test]
    fn test_clone() {
        let name = DisplayNameComponent::new("RTL2832".to_string());
        let cloned = name.clone();
        assert_eq!(cloned.name, "RTL2832");
    }
}
