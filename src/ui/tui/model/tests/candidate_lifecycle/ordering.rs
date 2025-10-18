use super::super::helpers::ModelTestContext;
use crate::ecs::CandidateState;

#[test]
fn test_deterministic_candidate_ordering() {
    let mut ctx = ModelTestContext::new();
    let window_id = 1;

    let candidates = vec![
        ("89.1-1", 89_100_000.0),
        ("88.3-1", 88_300_000.0),
        ("90.5-1", 90_500_000.0),
        ("87.9-1", 87_900_000.0),
    ];

    for (_, freq) in &candidates {
        ctx.update_candidate(*freq, window_id, CandidateState::Detected, None, None);
    }
    ctx.sync();

    let window = ctx.model.windows.get(&window_id).unwrap();
    assert_eq!(window.candidates.len(), 4);
    assert_eq!(window.candidates[0].frequency_hz, 89_100_000.0);
    assert_eq!(window.candidates[1].frequency_hz, 88_300_000.0);
    assert_eq!(window.candidates[2].frequency_hz, 90_500_000.0);
    assert_eq!(window.candidates[3].frequency_hz, 87_900_000.0);

    let displayable = window.displayable_candidates(true, false);
    assert_eq!(displayable.len(), 4);
    assert_eq!(displayable[0].frequency_hz, 89_100_000.0);
    assert_eq!(displayable[1].frequency_hz, 88_300_000.0);
    assert_eq!(displayable[2].frequency_hz, 90_500_000.0);
    assert_eq!(displayable[3].frequency_hz, 87_900_000.0);
}
