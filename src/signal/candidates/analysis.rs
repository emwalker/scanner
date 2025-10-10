use crate::core::types::Peak;
use tracing::debug;

use super::scoring::{
    calculate_center_proximity_score, calculate_frequency_span_score, calculate_peak_density_score,
    calculate_signal_consistency_score, calculate_signal_strength_score,
};

/// Analyze spectral characteristics around a frequency to determine if it's a main lobe or sidelobe
/// Main lobes are wider and have characteristic spectral patterns compared to sidelobes
pub(crate) fn analyze_spectral_characteristics(
    peaks: &[Peak],
    target_freq_mhz: f64,
    _sample_rate: f64,
    center_freq: f64,
) -> (f32, String) {
    let target_freq_hz = target_freq_mhz * 1e6;

    // Find peaks within ±200 kHz of target frequency (wider than FM channel spacing)
    let analysis_range_hz = 200000.0;
    let nearby_peaks: Vec<&Peak> = peaks
        .iter()
        .filter(|peak| (peak.frequency_hz - target_freq_hz).abs() <= analysis_range_hz)
        .collect();

    // Debug logging for 88.9 MHz specifically
    if (target_freq_mhz - 88.9).abs() < 0.01 {
        debug!(
            "88.9 MHz analysis: found {} peaks within ±200kHz range",
            nearby_peaks.len()
        );
        for (i, peak) in nearby_peaks.iter().take(5).enumerate() {
            debug!(
                "  Peak {}: {:.3} MHz, magnitude {:.3}, offset {:.1} kHz",
                i + 1,
                peak.frequency_hz / 1e6,
                peak.magnitude,
                (peak.frequency_hz - target_freq_hz) / 1e3
            );
        }
    }

    if nearby_peaks.is_empty() {
        return (0.0, "No signal".to_string());
    }

    // Sort peaks by frequency for width analysis
    let mut sorted_peaks = nearby_peaks.clone();
    sorted_peaks.sort_by(|a, b| a.frequency_hz.total_cmp(&b.frequency_hz));

    // Calculate spectral width characteristics
    let peak_count = sorted_peaks.len();
    let freq_span_khz = if peak_count > 1 {
        match (sorted_peaks.last(), sorted_peaks.first()) {
            (Some(last), Some(first)) => (last.frequency_hz - first.frequency_hz) / 1000.0,
            _ => {
                debug_assert!(
                    false,
                    "Invariant violated: sorted_peaks should have elements when peak_count > 1"
                );
                0.0
            }
        }
    } else {
        0.0
    };

    // Find the strongest peak in the group (should be the main signal)
    let max_magnitude = sorted_peaks
        .iter()
        .map(|p| p.magnitude)
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(0.0);

    // Calculate average magnitude
    let avg_magnitude = sorted_peaks.iter().map(|p| p.magnitude).sum::<f32>() / peak_count as f32;

    // Main lobe characteristics analysis
    let mut score: f64 = 0.0;
    let mut analysis_notes = Vec::new();

    // 1. Peak density analysis (main lobes have consistent energy distribution)
    let (density_score, density_note) = calculate_peak_density_score(peak_count, freq_span_khz);
    score += density_score;
    if !density_note.is_empty() {
        analysis_notes.push(density_note);
    }

    // 2. Frequency span analysis (main lobes have characteristic widths)
    let (span_score, span_note) = calculate_frequency_span_score(freq_span_khz);
    score += span_score;
    if !span_note.is_empty() {
        analysis_notes.push(span_note);
    }

    // 3. Signal strength and consistency
    let (consistency_score, consistency_note) =
        calculate_signal_consistency_score(max_magnitude, avg_magnitude);
    score += consistency_score;
    if !consistency_note.is_empty() {
        analysis_notes.push(consistency_note);
    }

    // 4. Distance from center frequency (closer = more likely to be legitimate)
    let center_freq_mhz = center_freq / 1e6;
    let (proximity_score, proximity_note) =
        calculate_center_proximity_score(target_freq_mhz, center_freq_mhz);
    score += proximity_score;
    if !proximity_note.is_empty() {
        analysis_notes.push(proximity_note);
    }

    // 5. Absolute signal strength
    let (strength_score, strength_note) = calculate_signal_strength_score(max_magnitude);
    score += strength_score;
    if !strength_note.is_empty() {
        analysis_notes.push(strength_note);
    }

    let analysis_summary = analysis_notes.join(", ");

    // Additional debug for 88.9 MHz
    if (target_freq_mhz - 88.9).abs() < 0.01 {
        let magnitude_ratio = max_magnitude / avg_magnitude.max(1.0);
        let peak_density = peak_count as f64 / freq_span_khz.max(1.0);
        debug!(
            "88.9 MHz detailed analysis: peak_count={}, freq_span_khz={:.1}, max_mag={:.3}, avg_mag={:.3}, mag_ratio={:.2}, peak_density={:.1}, final_score={:.3}",
            peak_count,
            freq_span_khz,
            max_magnitude,
            avg_magnitude,
            magnitude_ratio,
            peak_density,
            score
        );
    }

    (score.clamp(0.0, 1.0) as f32, analysis_summary)
}
