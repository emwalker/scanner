use tracing::debug;

use crate::core::types::ScannerError;

/// Test helper function to assert that a classifier correctly classifies audio samples
///
/// # Arguments
/// * `classifier` - An instantiated classifier implementing the Classifier trait
/// * `overrides` - List of (filename, expected_quality) tuples for cases where the classifier is
///   expected to deviate from the training dataset. An empty list means the classifier should have
///   perfect accuracy against the training data.
///
/// # Usage
/// A poor classifier will need many overrides to pass the test, while a good classifier
/// will need minimal or no overrides. This captures the current behavior and protects
/// against regressions.
pub fn assert_classifies_audio(
    classifier: &dyn crate::audio::quality::Classifier,
    overrides: &[(&str, crate::audio::quality::AudioQuality)],
) -> crate::core::types::Result<()> {
    use std::collections::HashMap;

    // Convert overrides to a HashMap for quick lookup
    let override_map: HashMap<&str, crate::audio::quality::AudioQuality> =
        overrides.iter().cloned().collect();

    // Get the training dataset
    let training_data = crate::audio::quality::training_dataset();

    let mut total_tests = 0;
    let mut correct_classifications = 0;
    let mut failed_files = Vec::new();
    let mut unnecessary_overrides = Vec::new();

    for (filename, training_quality) in training_data.iter() {
        // Check for unnecessary overrides (override matches training dataset expectation)
        if let Some(override_quality) = override_map.get(filename)
            && override_quality == training_quality
        {
            unnecessary_overrides.push((filename.to_string(), *training_quality));
        }

        // Check if there's an override for this file
        let expected_quality = override_map.get(filename).unwrap_or(training_quality);

        // Construct the path to the audio file
        let wav_path = std::path::PathBuf::from("tests/data/audio/quality").join(filename);

        // Skip files that don't exist (similar to training logic)
        if !wav_path.exists() {
            debug!(filename = %filename, "Audio file not found, skipping test");
            continue;
        }

        // Load the audio file
        let audio_samples = match crate::file::wave::load_file(&wav_path) {
            Ok(samples) => samples,
            Err(e) => {
                debug!(filename = %filename, error = %e, "Failed to load audio file, skipping");
                continue;
            }
        };

        // Analyze with the classifier
        match classifier.analyze(&audio_samples, 48000.0) {
            Ok(result) => {
                total_tests += 1;

                if result.quality == *expected_quality {
                    correct_classifications += 1;
                    debug!(
                        filename = %filename,
                        expected = %expected_quality.to_human_string(),
                        actual = %result.quality.to_human_string(),
                        confidence = result.confidence,
                        "Classification correct"
                    );
                } else {
                    failed_files.push((
                        filename.to_string(),
                        *expected_quality,
                        result.quality,
                        result.confidence,
                    ));
                    debug!(
                        filename = %filename,
                        expected = %expected_quality.to_human_string(),
                        actual = %result.quality.to_human_string(),
                        confidence = result.confidence,
                        "Classification mismatch"
                    );
                }
            }
            Err(e) => {
                debug!(filename = %filename, error = %e, "Classification failed");
                failed_files.push((
                    filename.to_string(),
                    *expected_quality,
                    crate::audio::quality::AudioQuality::Unknown,
                    0.0,
                ));
            }
        }
    }

    // Report results
    debug!(
        classifier = classifier.name(),
        total_tests = total_tests,
        correct = correct_classifications,
        accuracy_percent = if total_tests > 0 {
            (correct_classifications as f32 / total_tests as f32) * 100.0
        } else {
            0.0
        },
        "Classification test completed"
    );

    // Check for unnecessary overrides first
    if !unnecessary_overrides.is_empty() {
        let mut error_message = format!(
            "Classifier '{}' has {} unnecessary override(s) that match the training dataset:\n",
            classifier.name(),
            unnecessary_overrides.len()
        );

        for (filename, quality) in unnecessary_overrides {
            error_message.push_str(&format!(
                "  {} - Override specifies {}, but training dataset already expects {}\n",
                filename,
                quality.to_human_string(),
                quality.to_human_string()
            ));
        }

        error_message
            .push_str("\nRemove these unnecessary overrides to keep the override list minimal.\n");

        return Err(ScannerError::Custom(error_message));
    }

    // Assert that all classifications were correct
    if !failed_files.is_empty() {
        let mut error_message = format!(
            "Classifier '{}' failed {} out of {} tests:\n",
            classifier.name(),
            failed_files.len(),
            total_tests
        );

        for (filename, expected, actual, confidence) in failed_files {
            error_message.push_str(&format!(
                "  {} - Expected: {}, Got (possibly via an override): {} (confidence: {:.2})\n",
                filename,
                expected.to_human_string(),
                actual.to_human_string(),
                confidence
            ));
        }

        return Err(ScannerError::Custom(error_message));
    }

    Ok(())
}
