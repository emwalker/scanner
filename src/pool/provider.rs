use crate::pool::{TaskRequirements, Tuner, TunerActivity};
use crate::types::Result;

/// Trait for objects that can provide tuners on demand
///
/// This abstraction allows Window to acquire tuners without
/// directly coupling to the Pool implementation. This enables:
/// - Dependency injection for testing
/// - Alternative tuner management strategies
/// - Following the dependency inversion principle
pub trait TunerProvider: Send + Sync {
    /// Acquire a tuner matching the given requirements
    ///
    /// Returns an error if no tuner is available or if the pool is shutting down.
    fn acquire(&self, requirements: &TaskRequirements, activity: TunerActivity) -> Result<Tuner>;

    /// Try to acquire a tuner (non-blocking, returns None if unavailable)
    ///
    /// Returns `None` instead of an error if no tuner is available.
    /// This is useful for optional tuner acquisition or polling scenarios.
    fn try_acquire(
        &self,
        requirements: &TaskRequirements,
        activity: TunerActivity,
    ) -> Option<Tuner>;
}

/// Implement TunerProvider for Pool
///
/// This implementation delegates to the existing Pool methods,
/// allowing Pool to be used anywhere a TunerProvider is expected.
impl TunerProvider for crate::pool::Pool {
    fn acquire(&self, requirements: &TaskRequirements, activity: TunerActivity) -> Result<Tuner> {
        // Delegate to Pool's existing acquire method
        crate::pool::Pool::acquire(self, requirements, activity)
    }

    fn try_acquire(
        &self,
        requirements: &TaskRequirements,
        activity: TunerActivity,
    ) -> Option<Tuner> {
        // Delegate to Pool's existing try_acquire method
        crate::pool::Pool::try_acquire(self, requirements, activity)
    }
}
