use std::time::Instant;

use crate::hardware::types::Backend;

#[derive(Debug, Clone)]
pub struct DeviceLifecycleComponent {
    pub added_at: Instant,
    pub backend: Backend,
    pub num_tuners: usize,
}

impl DeviceLifecycleComponent {
    pub fn new(backend: Backend, num_tuners: usize) -> Self {
        Self {
            added_at: Instant::now(),
            backend,
            num_tuners,
        }
    }

    pub fn age(&self) -> std::time::Duration {
        self.added_at.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lifecycle_creation() {
        let lifecycle = DeviceLifecycleComponent::new(Backend::Mock, 2);

        assert_eq!(lifecycle.backend, Backend::Mock);
        assert_eq!(lifecycle.num_tuners, 2);
        assert!(lifecycle.age().as_millis() < 100);
    }

    #[test]
    fn test_lifecycle_age() {
        let lifecycle = DeviceLifecycleComponent::new(Backend::Mock, 1);

        std::thread::sleep(std::time::Duration::from_millis(10));

        assert!(lifecycle.age().as_millis() >= 10);
    }
}
