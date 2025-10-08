use std::fs;
use std::path::Path;

pub fn generate_versioned_filename(model_version: &str) -> String {
    let date = chrono::Utc::now().format("%Y%m%d").to_string();
    format!("models/audio_quality_rf_v{}_{}.bin", model_version, date)
}

pub fn discover_latest_model() -> Option<String> {
    let models_dir = Path::new("models");
    if !models_dir.exists() {
        return None;
    }

    let mut versioned_models = Vec::new();

    if let Ok(entries) = fs::read_dir(models_dir) {
        for entry in entries.flatten() {
            let filename = entry.file_name().to_string_lossy().to_string();

            if filename.starts_with("audio_quality_rf_v")
                && filename.ends_with(".bin")
                && let Some(version_part) = filename.strip_prefix("audio_quality_rf_v")
                && let Some(base) = version_part.strip_suffix(".bin")
                && let Some(last_underscore) = base.rfind('_')
            {
                let version = &base[..last_underscore];
                let date = &base[last_underscore + 1..];

                versioned_models.push((
                    version.to_string(),
                    date.to_string(),
                    entry.path().to_string_lossy().to_string(),
                ));
            }
        }
    }

    if !versioned_models.is_empty() {
        versioned_models.sort_by(|a, b| match b.0.cmp(&a.0) {
            std::cmp::Ordering::Equal => b.1.cmp(&a.1),
            other => other,
        });

        return Some(versioned_models[0].2.clone());
    }

    let legacy_path = "models/audio_quality_rf.bin";
    if Path::new(legacy_path).exists() {
        Some(legacy_path.to_string())
    } else {
        None
    }
}
