//! Constraint component for tuner allocation

use std::ops::Range;

/// Component tracking tuner allocation constraints
#[derive(Debug, Clone)]
pub struct ConstraintComponent {
    /// Allowed frequency range (Hz), None means no restriction
    pub allowed_freq_range: Option<Range<f64>>,

    /// Maximum sample rate (Hz), None means no restriction
    pub max_sample_rate: Option<f64>,

    /// List of blocked frequency ranges (Hz)
    pub blocked_ranges: Vec<Range<f64>>,
}

impl ConstraintComponent {
    pub fn new() -> Self {
        Self {
            allowed_freq_range: None,
            max_sample_rate: None,
            blocked_ranges: Vec::new(),
        }
    }

    pub fn set_allowed_freq_range(&mut self, range: Range<f64>) {
        self.allowed_freq_range = Some(range);
    }

    pub fn clear_allowed_freq_range(&mut self) {
        self.allowed_freq_range = None;
    }

    pub fn set_max_sample_rate(&mut self, rate: f64) {
        self.max_sample_rate = Some(rate);
    }

    pub fn clear_max_sample_rate(&mut self) {
        self.max_sample_rate = None;
    }

    pub fn add_blocked_range(&mut self, range: Range<f64>) {
        self.blocked_ranges.push(range);
    }

    pub fn clear_blocked_ranges(&mut self) {
        self.blocked_ranges.clear();
    }

    pub fn allows_frequency(&self, freq_hz: f64) -> bool {
        // Check if within allowed range (if specified)
        if let Some(ref allowed) = self.allowed_freq_range
            && !allowed.contains(&freq_hz)
        {
            return false;
        }

        // Check if in any blocked range
        for blocked in &self.blocked_ranges {
            if blocked.contains(&freq_hz) {
                return false;
            }
        }

        true
    }

    pub fn allows_sample_rate(&self, rate_hz: f64) -> bool {
        if let Some(max_rate) = self.max_sample_rate {
            rate_hz <= max_rate
        } else {
            true
        }
    }

    pub fn allows_frequency_and_rate(&self, freq_hz: f64, rate_hz: f64) -> bool {
        self.allows_frequency(freq_hz) && self.allows_sample_rate(rate_hz)
    }
}

impl Default for ConstraintComponent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_constraint() {
        let constraint = ConstraintComponent::new();

        assert!(constraint.allowed_freq_range.is_none());
        assert!(constraint.max_sample_rate.is_none());
        assert!(constraint.blocked_ranges.is_empty());
    }

    #[test]
    fn test_default_allows_everything() {
        let constraint = ConstraintComponent::default();

        assert!(constraint.allows_frequency(88.9e6));
        assert!(constraint.allows_sample_rate(2.4e6));
        assert!(constraint.allows_frequency_and_rate(88.9e6, 2.4e6));
    }

    #[test]
    fn test_allowed_freq_range() {
        let mut constraint = ConstraintComponent::new();
        constraint.set_allowed_freq_range(88.0e6..108.0e6);

        assert!(constraint.allows_frequency(88.9e6));
        assert!(constraint.allows_frequency(107.9e6));
        assert!(!constraint.allows_frequency(50.0e6));
        assert!(!constraint.allows_frequency(150.0e6));
    }

    #[test]
    fn test_clear_allowed_range() {
        let mut constraint = ConstraintComponent::new();
        constraint.set_allowed_freq_range(88.0e6..108.0e6);
        assert!(!constraint.allows_frequency(50.0e6));

        constraint.clear_allowed_freq_range();
        assert!(constraint.allows_frequency(50.0e6));
    }

    #[test]
    fn test_blocked_ranges() {
        let mut constraint = ConstraintComponent::new();
        constraint.add_blocked_range(88.0e6..90.0e6);
        constraint.add_blocked_range(100.0e6..102.0e6);

        assert!(!constraint.allows_frequency(88.9e6));
        assert!(!constraint.allows_frequency(101.0e6));
        assert!(constraint.allows_frequency(95.0e6));
    }

    #[test]
    fn test_clear_blocked_ranges() {
        let mut constraint = ConstraintComponent::new();
        constraint.add_blocked_range(88.0e6..90.0e6);
        assert!(!constraint.allows_frequency(88.9e6));

        constraint.clear_blocked_ranges();
        assert!(constraint.allows_frequency(88.9e6));
    }

    #[test]
    fn test_max_sample_rate() {
        let mut constraint = ConstraintComponent::new();
        constraint.set_max_sample_rate(2.0e6);

        assert!(constraint.allows_sample_rate(1.5e6));
        assert!(constraint.allows_sample_rate(2.0e6));
        assert!(!constraint.allows_sample_rate(2.5e6));
    }

    #[test]
    fn test_clear_max_sample_rate() {
        let mut constraint = ConstraintComponent::new();
        constraint.set_max_sample_rate(2.0e6);
        assert!(!constraint.allows_sample_rate(3.0e6));

        constraint.clear_max_sample_rate();
        assert!(constraint.allows_sample_rate(3.0e6));
    }

    #[test]
    fn test_combined_constraints() {
        let mut constraint = ConstraintComponent::new();
        constraint.set_allowed_freq_range(88.0e6..108.0e6);
        constraint.add_blocked_range(100.0e6..102.0e6);
        constraint.set_max_sample_rate(2.0e6);

        // In range, not blocked, good rate
        assert!(constraint.allows_frequency_and_rate(95.0e6, 1.5e6));

        // In range but blocked
        assert!(!constraint.allows_frequency_and_rate(101.0e6, 1.5e6));

        // Out of range
        assert!(!constraint.allows_frequency_and_rate(50.0e6, 1.5e6));

        // Good frequency but bad rate
        assert!(!constraint.allows_frequency_and_rate(95.0e6, 3.0e6));
    }
}
