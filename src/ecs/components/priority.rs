//! Priority component for tuner allocation

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    None = 0,
    Low = 1,
    Medium = 2,
    High = 3,
}

impl Default for Priority {
    fn default() -> Self {
        Self::Medium
    }
}

/// Component tracking tuner allocation priorities
#[derive(Debug, Clone)]
pub struct PriorityComponent {
    /// Priority for audio playback allocation
    pub audio_priority: Priority,

    /// Priority for scanning allocation
    pub scanning_priority: Priority,
}

impl PriorityComponent {
    pub fn new(audio_priority: Priority, scanning_priority: Priority) -> Self {
        Self {
            audio_priority,
            scanning_priority,
        }
    }

    pub fn set_audio_priority(&mut self, priority: Priority) {
        self.audio_priority = priority;
    }

    pub fn set_scanning_priority(&mut self, priority: Priority) {
        self.scanning_priority = priority;
    }

    pub fn allows_audio(&self) -> bool {
        self.audio_priority != Priority::None
    }

    pub fn allows_scanning(&self) -> bool {
        self.scanning_priority != Priority::None
    }
}

impl Default for PriorityComponent {
    fn default() -> Self {
        Self::new(Priority::Medium, Priority::Medium)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_priority() {
        let priority = PriorityComponent::new(Priority::High, Priority::Low);

        assert_eq!(priority.audio_priority, Priority::High);
        assert_eq!(priority.scanning_priority, Priority::Low);
        assert!(priority.allows_audio());
        assert!(priority.allows_scanning());
    }

    #[test]
    fn test_default_priority() {
        let priority = PriorityComponent::default();

        assert_eq!(priority.audio_priority, Priority::Medium);
        assert_eq!(priority.scanning_priority, Priority::Medium);
    }

    #[test]
    fn test_set_priorities() {
        let mut priority = PriorityComponent::default();

        priority.set_audio_priority(Priority::None);
        assert_eq!(priority.audio_priority, Priority::None);
        assert!(!priority.allows_audio());

        priority.set_scanning_priority(Priority::High);
        assert_eq!(priority.scanning_priority, Priority::High);
        assert!(priority.allows_scanning());
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::None < Priority::Low);
        assert!(Priority::Low < Priority::Medium);
        assert!(Priority::Medium < Priority::High);
    }

    #[test]
    fn test_none_priority() {
        let priority = PriorityComponent::new(Priority::None, Priority::None);

        assert!(!priority.allows_audio());
        assert!(!priority.allows_scanning());
    }
}
