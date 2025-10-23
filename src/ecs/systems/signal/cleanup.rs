use crate::{
    core::types::Result,
    ecs::system::{System, SystemContext},
};

pub struct CandidateCleanupSystem;

impl CandidateCleanupSystem {
    pub fn new() -> Self {
        CandidateCleanupSystem
    }
}

impl System for CandidateCleanupSystem {
    fn name(&self) -> &'static str {
        "CandidateCleanupSystem"
    }

    fn run(&mut self, _context: &mut SystemContext) -> Result<()> {
        Ok(())
    }
}

impl Default for CandidateCleanupSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_name() {
        let system = CandidateCleanupSystem::new();
        assert_eq!(system.name(), "CandidateCleanupSystem");
    }

    #[test]
    fn test_system_runs_with_empty_context() {
        let mut system = CandidateCleanupSystem::new();
        let mut context = SystemContext::new();
        let result = system.run(&mut context);
        assert!(result.is_ok());
    }
}
