use tracing::warn;

use super::frequency_tracking::run_frequency_tracking;
use crate::core::types::{Result, ScanningConfig};

pub(crate) fn refine_frequency(
    frequency_hz: f64,
    config: &ScanningConfig,
    sdr_rx: tokio::sync::broadcast::Receiver<crate::broadcast::SamplePacket>,
) -> Result<f64> {
    let refined_frequency = if config.signal_processing.frequency_tracking.disabled {
        tracing::debug!(
            freq_mhz = frequency_hz / 1e6,
            "Frequency tracking disabled, using FFT estimate"
        );
        frequency_hz
    } else {
        match run_frequency_tracking(frequency_hz, config, sdr_rx) {
            Some(freq) => {
                tracing::debug!(
                    original_mhz = frequency_hz / 1e6,
                    refined_mhz = freq / 1e6,
                    error_khz = (freq - frequency_hz) / 1e3,
                    "Frequency tracking successful"
                );
                freq
            }
            None => {
                tracing::debug!(
                    freq_mhz = frequency_hz / 1e6,
                    "Frequency tracking failed, using FFT estimate"
                );
                frequency_hz
            }
        }
    };

    Ok((refined_frequency / 100000.0).round() * 100000.0)
}

pub(crate) fn is_frequency_already_processed(refined_frequency: f64) -> Result<bool> {
    let frequency_khz = (refined_frequency / 1000.0) as u64;

    let already_processed = {
        let processed = match crate::signal::PROCESSED_FREQUENCIES.read() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!("PROCESSED_FREQUENCIES RwLock poisoned - recovering");
                poisoned.into_inner()
            }
        };
        processed.contains(&frequency_khz)
    };

    if already_processed {
        tracing::debug!(
            refined_freq_mhz = refined_frequency / 1e6,
            frequency_khz = frequency_khz,
            "Frequency already processed in another window, skipping analysis"
        );
        Ok(true)
    } else {
        tracing::debug!(
            refined_freq_mhz = refined_frequency / 1e6,
            frequency_khz = frequency_khz,
            "New frequency detected, proceeding with analysis"
        );
        Ok(false)
    }
}

pub(crate) fn mark_frequency_as_processed(frequency_khz: u64) {
    {
        let mut processed = match crate::signal::PROCESSED_FREQUENCIES.write() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!("PROCESSED_FREQUENCIES RwLock poisoned - recovering");
                poisoned.into_inner()
            }
        };
        processed.insert(frequency_khz);
    }
    tracing::debug!(frequency_khz, "Frequency marked as successfully processed");
}
