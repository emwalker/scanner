use crate::ui::tui::model::Model;
use crate::ui::{ProgressEvent, ProgressEventType};
use std::time::Instant;
/// Test that candidates progress through all expected states
/// Test deterministic candidate ordering within windows
#[test]
fn test_deterministic_candidate_ordering() {
    let mut model = Model::new();
    let window_id = 1;
    // Create candidates in specific order
    let candidates = vec![
        ("89.1-1", 89_100_000.0),
        ("88.3-1", 88_300_000.0),
        ("90.5-1", 90_500_000.0),
        ("87.9-1", 87_900_000.0),
    ];
    for (id, freq) in &candidates {
        model.update(ProgressEvent {
            event_type: ProgressEventType::CandidateCreated,
            frequency_hz: *freq,
            metadata: crate::scanning::window::WindowMetadata {
                center_frequency_hz: *freq,
                window_id,
            },
            candidate_id: Some(id.to_string()),
            audio_quality: None,
            signal_strength: None,
            timestamp: Instant::now(),
            tuner_id: None,
        });
    }
    let window = model.windows.get(&window_id).unwrap();
    // Candidates should maintain insertion order
    assert_eq!(window.candidates.len(), 4);
    assert_eq!(window.candidates[0].frequency_hz, 89_100_000.0);
    assert_eq!(window.candidates[1].frequency_hz, 88_300_000.0);
    assert_eq!(window.candidates[2].frequency_hz, 90_500_000.0);
    assert_eq!(window.candidates[3].frequency_hz, 87_900_000.0);
    // displayable_candidates should also maintain this order
    let displayable = window.displayable_candidates(true, false);
    assert_eq!(displayable.len(), 4);
    assert_eq!(displayable[0].frequency_hz, 89_100_000.0);
    assert_eq!(displayable[1].frequency_hz, 88_300_000.0);
    assert_eq!(displayable[2].frequency_hz, 90_500_000.0);
    assert_eq!(displayable[3].frequency_hz, 87_900_000.0);
}
