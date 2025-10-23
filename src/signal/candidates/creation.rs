use tracing::debug;

use super::analysis::analyze_spectral_characteristics;
use crate::{
    core::{
        config::ScanningConfig,
        types::{self, Peak},
    },
    signal::Candidate,
};

fn create_fm_signal(frequency_mhz: f64, peaks: &[Peak], spectral_score: f32) -> types::Candidate {
    let signal_strength = if spectral_score > 0.8 {
        "Strong"
    } else if spectral_score > 0.6 {
        "Medium"
    } else {
        "Weak"
    };

    let tolerance_mhz = 0.1;
    let nearby_peaks: Vec<&Peak> = peaks
        .iter()
        .filter(|peak| {
            let peak_freq_mhz = peak.frequency_hz / 1e6;
            (peak_freq_mhz - frequency_mhz).abs() <= tolerance_mhz
        })
        .collect();

    let peak_count = nearby_peaks.len();
    let max_magnitude = nearby_peaks
        .iter()
        .map(|p| p.magnitude)
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(0.0);
    let avg_magnitude = if peak_count > 0 {
        nearby_peaks.iter().map(|p| p.magnitude).sum::<f32>() / peak_count as f32
    } else {
        0.0
    };

    types::Candidate::Fm(Candidate {
        frequency_hz: frequency_mhz * 1e6,
        peak_count,
        max_magnitude,
        avg_magnitude,
        signal_strength: signal_strength.to_string(),
    })
}

fn next_fm_frequency(current_freq_mhz: f64) -> f64 {
    (current_freq_mhz * 10.0 + 2.0) / 10.0 // Next odd tenth (add 0.2)
}

fn calculate_starting_fm_frequency(freq_start_mhz: f64) -> f64 {
    let mut fm_freq = (freq_start_mhz * 10.0).ceil() / 10.0;
    if (fm_freq * 10.0) as i32 % 2 == 0 {
        fm_freq += 0.1; // Make it an odd tenth
    }
    fm_freq
}

pub(crate) fn find_signals(
    peaks: &[Peak],
    config: &ScanningConfig,
    center_freq: f64,
) -> Vec<types::Candidate> {
    debug!("Using spectral analysis for FM station detection with sidelobe discrimination...");

    let scan_range_mhz = config.samp_rate / 2e6;
    let freq_start_mhz = (center_freq / 1e6) - scan_range_mhz;
    let freq_end_mhz = (center_freq / 1e6) + scan_range_mhz;

    debug!(
        "Analyzing spectral patterns in range: {:.1} - {:.1} MHz",
        freq_start_mhz, freq_end_mhz
    );

    let mut signals = Vec::new();
    let mut fm_freq = calculate_starting_fm_frequency(freq_start_mhz);

    while fm_freq <= freq_end_mhz {
        debug!("Analyzing {:.1} MHz... ", fm_freq);
        let _ = std::io::Write::flush(&mut std::io::stdout());

        let (spectral_score, analysis_summary) =
            analyze_spectral_characteristics(peaks, fm_freq, config.samp_rate, center_freq);

        debug!("score: {:.3} ({})", spectral_score, analysis_summary);

        if spectral_score >= config.peak_detection.spectral_threshold {
            signals.push(create_fm_signal(fm_freq, peaks, spectral_score));
        }

        fm_freq = next_fm_frequency(fm_freq);
    }

    signals
}
