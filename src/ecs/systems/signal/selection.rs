use crate::{
    core::types::Result,
    ecs::system::{System, SystemContext},
};

pub struct CandidateSelectionSystem;

impl CandidateSelectionSystem {
    pub fn new() -> Self {
        CandidateSelectionSystem
    }
}

impl System for CandidateSelectionSystem {
    fn name(&self) -> &'static str {
        "CandidateSelectionSystem"
    }

    fn run(&mut self, _context: &mut SystemContext) -> Result<()> {
        Ok(())
    }
}

impl Default for CandidateSelectionSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_name() {
        let system = CandidateSelectionSystem::new();
        assert_eq!(system.name(), "CandidateSelectionSystem");
    }

    #[test]
    fn test_system_runs_with_empty_context() {
        let mut system = CandidateSelectionSystem::new();
        let mut context = SystemContext::new();
        let result = system.run(&mut context);
        assert!(result.is_ok());
    }
}
