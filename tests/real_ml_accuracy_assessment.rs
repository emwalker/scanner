use scanner::audio_quality::{AudioQuality, RandomForestClassifier, get_training_dataset};
use std::collections::HashMap;
use std::path::PathBuf;

#[test]
fn test_real_ml_accuracy_on_training_data() {
    println!("Testing REAL ML (Random Forest) accuracy against human-rated training data...");

    // Create and train Random Forest classifier
    let mut classifier = RandomForestClassifier::new(48000.0);

    println!("Training Random Forest model on handcrafted features...");
    classifier.train().expect("Failed to train ML model");
    println!("✓ Real ML model trained successfully");

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
        let audio_samples = match classifier.load_wav_file(&wav_path) {
            Ok(samples) => samples,
            Err(e) => {
                println!("⚠ Warning: Failed to load WAV file {}: {}", filename, e);
                continue;
            }
        };

        // Get prediction from real ML classifier
        let ml_result = match classifier.predict(&audio_samples) {
            Ok(result) => result,
            Err(e) => {
                println!("⚠ Warning: ML prediction failed for {}: {}", filename, e);
                continue;
            }
        };

        let predicted_quality = ml_result.quality;
        let is_correct = predicted_quality == *expected_quality;
        if is_correct {
            correct_predictions += 1;
        }

        // Update confusion matrix
        *confusion_matrix
            .entry((*expected_quality, predicted_quality))
            .or_insert(0) += 1;

        println!(
            "Test {}: {} -> {} (expected {}) {} [conf: {:.1}%] [scores: {:?}]",
            i + 1,
            filename,
            predicted_quality.to_human_string(),
            expected_quality.to_human_string(),
            if is_correct { "✓" } else { "✗" },
            ml_result.confidence * 100.0,
            ml_result.model_scores
        );
    }

    let accuracy = (correct_predictions as f64 / total_samples as f64) * 100.0;

    println!("\n=== REAL ML (Random Forest) Audio Quality Accuracy Assessment ===");
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

    // Compare with other approaches
    println!("\n--- Comparison with Other Approaches ---");
    println!("CNN (Burn framework): 36.6%");
    println!("Hand-coded rules: 39.0%");
    println!("REAL ML (Random Forest): {:.1}%", accuracy);

    if accuracy > 39.0 {
        println!(
            "✓ Real ML outperforms hand-coded rules by {:.1} percentage points",
            accuracy - 39.0
        );
    } else if accuracy > 36.6 {
        println!(
            "✓ Real ML outperforms CNN by {:.1} percentage points",
            accuracy - 36.6
        );
        println!(
            "✗ Real ML underperforms hand-coded rules by {:.1} percentage points",
            39.0 - accuracy
        );
    } else {
        println!("✗ Real ML underperforms both CNN and hand-coded rules");
    }

    println!("\nNote: This test evaluates actual machine learning (Random Forest) trained on");
    println!("handcrafted audio features vs hand-coded rules vs CNN with mel-spectrograms.");
}
