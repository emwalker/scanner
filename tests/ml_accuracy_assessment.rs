use scanner::audio_quality::{AudioQuality, HeuristicClassifier, get_training_dataset};
use std::collections::HashMap;
use std::path::PathBuf;

#[test]
fn test_ml_accuracy_on_training_data() {
    println!(
        "Testing heuristic rule-based classifier accuracy against human-rated training data..."
    );

    // Create heuristic classifier
    let classifier = HeuristicClassifier::new(48000.0);

    let training_data = get_training_dataset();
    let total_samples = training_data.len();
    let mut correct_predictions = 0;
    let mut confusion_matrix: HashMap<(AudioQuality, AudioQuality), usize> = HashMap::new();

    println!(
        "Testing all {} samples from training data...",
        total_samples
    );

    // Test each sample
    for (i, (filename, expected_quality)) in training_data.iter().enumerate() {
        let wav_path = PathBuf::from("tests/data/audio/quality").join(filename);

        if !wav_path.exists() {
            println!("⚠ Warning: Audio file not found: {}", wav_path.display());
            continue;
        }

        // Load and decode the WAV file
        let audio_samples = match scanner::wave::load_file(&wav_path) {
            Ok(samples) => samples,
            Err(e) => {
                println!("⚠ Warning: Failed to load WAV file {}: {}", filename, e);
                continue;
            }
        };

        // Get prediction from ML classifier
        let quality_result = match classifier.predict(&audio_samples) {
            Ok(result) => result,
            Err(e) => {
                println!("⚠ Warning: Analysis failed for {}: {}", filename, e);
                continue;
            }
        };

        let predicted_quality = quality_result.quality;
        let is_correct = predicted_quality == *expected_quality;
        if is_correct {
            correct_predictions += 1;
        }

        // Update confusion matrix
        *confusion_matrix
            .entry((*expected_quality, predicted_quality))
            .or_insert(0) += 1;

        println!(
            "Test {}: {} -> {} (expected {}) {} [conf: {:.1}%] {}",
            i + 1,
            filename,
            predicted_quality.to_human_string(),
            expected_quality.to_human_string(),
            if is_correct { "✓" } else { "✗" },
            quality_result.confidence * 100.0,
            quality_result.reasoning
        );
    }

    let accuracy = (correct_predictions as f64 / total_samples as f64) * 100.0;

    println!("\n=== Heuristic Rule-Based Audio Quality Accuracy Assessment ===");
    println!("Total samples: {}", total_samples);
    println!("Correct predictions: {}", correct_predictions);
    println!("Overall accuracy: {:.1}%", accuracy);

    // Print confusion matrix
    println!("\n--- Confusion Matrix ---");
    println!("(Human → Predicted)");
    println!("              Static NoAudio    Poor Moderate    Good");

    for expected in &[
        AudioQuality::Static,
        AudioQuality::NoAudio,
        AudioQuality::Poor,
        AudioQuality::Moderate,
        AudioQuality::Good,
    ] {
        print!("{:>12}", expected.to_human_string());
        for predicted in &[
            AudioQuality::Static,
            AudioQuality::NoAudio,
            AudioQuality::Poor,
            AudioQuality::Moderate,
            AudioQuality::Good,
        ] {
            let count = confusion_matrix.get(&(*expected, *predicted)).unwrap_or(&0);
            print!("{:>8}", count);
        }
        println!();
    }

    // Error analysis
    println!("\n--- Error Analysis ---");
    let mut error_counts: HashMap<String, usize> = HashMap::new();
    for ((expected, predicted), count) in &confusion_matrix {
        if expected != predicted {
            let error_key = format!(
                "{} misclassified as {}",
                expected.to_human_string(),
                predicted.to_human_string()
            );
            *error_counts.entry(error_key).or_insert(0) += count;
        }
    }

    let mut error_vec: Vec<_> = error_counts.into_iter().collect();
    error_vec.sort_by(|a, b| b.1.cmp(&a.1));
    for (error_type, count) in error_vec {
        println!("  {} samples: {}", count, error_type);
    }

    // Compare with CNN baseline
    println!("\n--- Comparison with CNN Baseline ---");
    println!("CNN accuracy (Burn framework): 36.6%");
    println!("Heuristic rule-based accuracy: {:.1}%", accuracy);

    if accuracy > 36.6 {
        println!(
            "✓ Heuristic approach outperforms CNN by {:.1} percentage points",
            accuracy - 36.6
        );
    } else {
        println!(
            "✗ Heuristic approach underperforms CNN by {:.1} percentage points",
            36.6 - accuracy
        );
    }

    println!("\nNote: This test evaluates rule-based heuristic classification approach");
    println!("against CNN with mel-spectrograms on the same human-rated training dataset.");
}
