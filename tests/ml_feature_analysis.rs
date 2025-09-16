use scanner::audio_quality::{AudioQuality, HeuristicClassifier, get_training_dataset};
use std::path::PathBuf;

#[test]
fn test_ml_feature_analysis() {
    println!("Analyzing heuristic features to understand classification bias...");

    let classifier = HeuristicClassifier::new(48000.0);
    let training_data = get_training_dataset();

    // Analyze a few samples from each quality category
    let samples_to_analyze = vec![
        ("000.087.700.000Hz-wfm-001.wav", AudioQuality::Static),
        ("000.096.300.000Hz-wfm-001.wav", AudioQuality::NoAudio),
        ("000.088.700.000Hz-wfm-001.wav", AudioQuality::Poor),
        ("000.089.700.000Hz-wfm-001.wav", AudioQuality::Moderate),
        ("000.088.900.000Hz-wfm-001.wav", AudioQuality::Good),
    ];

    for (filename, expected_quality) in samples_to_analyze {
        let wav_path = PathBuf::from("tests/data/audio/quality").join(filename);

        if !wav_path.exists() {
            println!("⚠ Warning: Audio file not found: {}", wav_path.display());
            continue;
        }

        let audio_samples = match scanner::wave::load_file(&wav_path) {
            Ok(samples) => samples,
            Err(e) => {
                println!("⚠ Warning: Failed to load WAV file {}: {}", filename, e);
                continue;
            }
        };

        let features = match classifier.extract_features(&audio_samples) {
            Ok(f) => f,
            Err(e) => {
                println!(
                    "⚠ Warning: Feature extraction failed for {}: {}",
                    filename, e
                );
                continue;
            }
        };

        let result = classifier.classify_quality(&features);

        println!(
            "\n=== {} (Expected: {}) ===",
            filename,
            expected_quality.to_human_string()
        );
        println!(
            "Predicted: {} (conf: {:.1}%)",
            result.quality.to_human_string(),
            result.confidence * 100.0
        );
        println!("Features:");
        println!("  RMS Energy: {:.6}", features.rms_energy);
        println!("  Peak Amplitude: {:.6}", features.peak_amplitude);
        println!("  Dynamic Range: {:.6}", features.dynamic_range);
        println!("  Spectral Centroid: {:.1} Hz", features.spectral_centroid);
        println!("  Spectral Rolloff: {:.1} Hz", features.spectral_rolloff);
        println!("  Spectral Flux: {:.6}", features.spectral_flux);
        println!("  High Freq Energy: {:.6}", features.high_freq_energy);
        println!("  Zero Crossing Rate: {:.6}", features.zero_crossing_rate);
        println!("  Silence Ratio: {:.3}", features.silence_ratio);
        println!("  SNR Estimate: {:.1} dB", features.snr_estimate);
        println!("  Harmonic Ratio: {:.3}", features.harmonic_ratio);
        println!("Reasoning: {}", result.reasoning);
    }

    println!("\n=== Feature Range Analysis ===");
    println!("Analyzing all training samples to understand feature distributions...");

    let mut all_features = Vec::new();
    for (filename, expected_quality) in training_data.iter() {
        let wav_path = PathBuf::from("tests/data/audio/quality").join(filename);

        if !wav_path.exists() {
            continue;
        }

        if let Ok(audio_samples) = scanner::wave::load_file(&wav_path) {
            if let Ok(features) = classifier.extract_features(&audio_samples) {
                all_features.push((features, *expected_quality));
            }
        }
    }

    // Compute statistics by quality level
    for quality in [
        AudioQuality::Static,
        AudioQuality::NoAudio,
        AudioQuality::Poor,
        AudioQuality::Moderate,
        AudioQuality::Good,
    ] {
        let quality_features: Vec<_> = all_features
            .iter()
            .filter(|(_, q)| *q == quality)
            .map(|(f, _)| f)
            .collect();

        if quality_features.is_empty() {
            continue;
        }

        let avg_rms = quality_features.iter().map(|f| f.rms_energy).sum::<f32>()
            / quality_features.len() as f32;
        let avg_snr = quality_features.iter().map(|f| f.snr_estimate).sum::<f32>()
            / quality_features.len() as f32;
        let avg_harmonic = quality_features
            .iter()
            .map(|f| f.harmonic_ratio)
            .sum::<f32>()
            / quality_features.len() as f32;
        let avg_dynamic = quality_features
            .iter()
            .map(|f| f.dynamic_range)
            .sum::<f32>()
            / quality_features.len() as f32;

        println!(
            "\n{} ({} samples):",
            quality.to_human_string(),
            quality_features.len()
        );
        println!("  Avg RMS Energy: {:.6}", avg_rms);
        println!("  Avg SNR: {:.1} dB", avg_snr);
        println!("  Avg Harmonic Ratio: {:.3}", avg_harmonic);
        println!("  Avg Dynamic Range: {:.3}", avg_dynamic);
    }
}
