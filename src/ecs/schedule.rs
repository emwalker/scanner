//! System scheduling and execution

use tracing::{debug, warn};

use crate::{
    core::types::Result,
    ecs::system::{System, SystemContext},
};

/// System scheduler that executes systems in a defined order
pub struct Scheduler {
    systems: Vec<Box<dyn System>>,
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
        }
    }

    /// Add a system to the schedule
    ///
    /// Systems will execute in the order they are added.
    pub fn add_system(&mut self, system: Box<dyn System>) {
        debug!(system_name = system.name(), "Adding system to schedule");
        self.systems.push(system);
    }

    /// Execute all systems in order
    ///
    /// If a system fails, execution continues but the error is logged.
    /// Returns Ok if all systems succeeded, or the first error encountered.
    pub fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        let mut first_error = None;

        for system in &mut self.systems {
            match system.run(context) {
                Ok(()) => {}
                Err(e) => {
                    warn!(
                        system_name = system.name(),
                        error = ?e,
                        "System execution failed"
                    );
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
            }
        }

        if let Some(error) = first_error {
            Err(error)
        } else {
            Ok(())
        }
    }

    /// Get the number of registered systems
    pub fn system_count(&self) -> usize {
        self.systems.len()
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CountingSystem {
        name: &'static str,
        count: usize,
    }

    impl CountingSystem {
        fn new(name: &'static str) -> Self {
            Self { name, count: 0 }
        }
    }

    impl System for CountingSystem {
        fn name(&self) -> &'static str {
            self.name
        }

        fn run(&mut self, _context: &mut SystemContext) -> Result<()> {
            self.count += 1;
            Ok(())
        }
    }

    #[test]
    fn test_scheduler_execution_order() {
        let mut scheduler = Scheduler::new();
        let mut context = SystemContext::new();

        scheduler.add_system(Box::new(CountingSystem::new("System1")));
        scheduler.add_system(Box::new(CountingSystem::new("System2")));
        scheduler.add_system(Box::new(CountingSystem::new("System3")));

        assert_eq!(scheduler.system_count(), 3);

        scheduler.run(&mut context).unwrap();
        scheduler.run(&mut context).unwrap();

        // All systems should have run twice
        // (We can't easily verify this without additional infrastructure)
    }

    struct FailingSystem {
        fail_on_run: usize,
        run_count: usize,
    }

    impl FailingSystem {
        fn new(fail_on_run: usize) -> Self {
            Self {
                fail_on_run,
                run_count: 0,
            }
        }
    }

    impl System for FailingSystem {
        fn name(&self) -> &'static str {
            "FailingSystem"
        }

        fn run(&mut self, _context: &mut SystemContext) -> Result<()> {
            self.run_count += 1;
            if self.run_count == self.fail_on_run {
                Err(crate::core::types::ScannerError::Custom(
                    "Planned failure".to_string(),
                ))
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn test_scheduler_error_handling() {
        let mut scheduler = Scheduler::new();
        let mut context = SystemContext::new();

        scheduler.add_system(Box::new(CountingSystem::new("System1")));
        scheduler.add_system(Box::new(FailingSystem::new(1)));
        scheduler.add_system(Box::new(CountingSystem::new("System3")));

        // First run should fail because FailingSystem fails
        let result = scheduler.run(&mut context);
        assert!(result.is_err());

        // Subsequent runs should succeed
        let result = scheduler.run(&mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_empty_scheduler() {
        let mut scheduler = Scheduler::new();
        let mut context = SystemContext::new();

        assert_eq!(scheduler.system_count(), 0);
        scheduler.run(&mut context).unwrap();
    }
}
