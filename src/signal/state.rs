use std::{
    collections::HashSet,
    sync::{LazyLock, RwLock},
};

use tracing::debug;

/// Global set of processed frequencies (rounded to nearest kHz) to avoid duplicate analysis
pub static PROCESSED_FREQUENCIES: LazyLock<RwLock<HashSet<u64>>> =
    LazyLock::new(|| RwLock::new(HashSet::new()));

/// Clear the processed frequencies set for a new scanning session
pub fn clear_processed_frequencies() {
    match PROCESSED_FREQUENCIES.try_write() {
        Ok(mut processed) => {
            let count = processed.len();
            processed.clear();
            debug!(
                cleared_count = count,
                "Cleared processed frequencies for new scanning session"
            );
        }
        Err(_) => {
            debug!("Could not acquire write lock on PROCESSED_FREQUENCIES, skipping clear");
        }
    }
}
