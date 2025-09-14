use scanner::audio_quality::RandomForestClassifier;

#[test]
fn test_wav_loading() {
    let analyzer = RandomForestClassifier::new(48000.0);

    let wav_path = "tests/data/audio/quality/000.087.700.000Hz-wfm-001.wav";

    println!("Loading WAV file: {}", wav_path);

    let samples = analyzer
        .load_wav_file(wav_path)
        .expect("Failed to load WAV file");

    println!("Loaded {} samples", samples.len());
    println!("Duration: {:.2} seconds", samples.len() as f32 / 48000.0);

    // Check basic properties
    assert!(!samples.is_empty(), "Samples should not be empty");
    assert!(
        samples.len() > 1000,
        "Should have reasonable number of samples"
    );

    // Check sample range (should be normalized to [-1.0, 1.0])
    let min_sample = samples.iter().copied().fold(f32::INFINITY, f32::min);
    let max_sample = samples.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    println!("Sample range: [{:.6}, {:.6}]", min_sample, max_sample);

    assert!(
        min_sample >= -1.0 && min_sample <= 1.0,
        "Min sample out of range"
    );
    assert!(
        max_sample >= -1.0 && max_sample <= 1.0,
        "Max sample out of range"
    );

    println!("✓ WAV loading test passed");
}
