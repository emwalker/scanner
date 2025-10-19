use crate::core::types::Result;
use std::fs;
use std::path::Path;

use super::args::TrainArgs;
use super::model::generate_versioned_filename;

pub fn handle_train_command(args: TrainArgs) -> Result<()> {
    let level = crate::logging::level_from_flags(args.verbose, args.quiet);
    crate::logging::init(level, crate::core::types::Format::Text, None)?;

    let output_model = args
        .output_model
        .unwrap_or_else(|| generate_versioned_filename(&args.model_version));

    tracing::debug!("Training Random Forest audio quality classifier");
    tracing::debug!(training_data_dir = %args.training_data_dir, "Using training data directory");
    tracing::debug!(output_model = %output_model, "Output model path");

    if let Some(parent) = Path::new(&output_model).parent() {
        fs::create_dir_all(parent)?;
    }

    let mut classifier = crate::audio::quality::random_forest::Classifier::new(args.sample_rate);

    tracing::debug!("Loading training data and extracting features");
    classifier.train()?;

    tracing::debug!(output_model = %output_model, "Saving trained model");
    classifier.save_model(&output_model, &args.model_version)?;

    tracing::debug!("Training complete! Model saved successfully");
    tracing::debug!(
        model_path = %output_model,
        "Model ready to use with: scanner scan --model-path {}",
        output_model
    );

    Ok(())
}
