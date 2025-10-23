use crate::{
    core::types::Result,
    ecs::system::{System, SystemContext},
};

pub struct CandidateRankingSystem;

impl CandidateRankingSystem {
    pub fn new() -> Self {
        CandidateRankingSystem
    }
}

impl System for CandidateRankingSystem {
    fn name(&self) -> &'static str {
        "CandidateRankingSystem"
    }

    fn run(&mut self, _context: &mut SystemContext) -> Result<()> {
        Ok(())
    }
}

impl Default for CandidateRankingSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_name() {
        let system = CandidateRankingSystem::new();
        assert_eq!(system.name(), "CandidateRankingSystem");
    }

    #[test]
    fn test_system_runs_with_empty_context() {
        let mut system = CandidateRankingSystem::new();
        let mut context = SystemContext::new();
        let result = system.run(&mut context);
        assert!(result.is_ok());
    }
}
