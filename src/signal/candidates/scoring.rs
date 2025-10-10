pub(crate) fn calculate_peak_density_score(
    peak_count: usize,
    freq_span_khz: f64,
) -> (f64, &'static str) {
    let peak_density = peak_count as f64 / freq_span_khz.max(1.0);
    if peak_density > 20.0 && peak_density < 200.0 {
        (0.3, "Good peak density")
    } else if peak_density > 200.0 {
        (-0.2, "High peak density (interference?)")
    } else {
        (0.0, "")
    }
}

pub(crate) fn calculate_frequency_span_score(freq_span_khz: f64) -> (f64, &'static str) {
    if freq_span_khz > 80.0 && freq_span_khz < 250.0 {
        (0.3, "Appropriate spectral width")
    } else if freq_span_khz < 15.0 {
        (-0.3, "Narrow spectral width (sidelobe?)")
    } else {
        (0.0, "")
    }
}

pub(crate) fn calculate_signal_consistency_score(
    max_magnitude: f32,
    avg_magnitude: f32,
) -> (f64, &'static str) {
    let magnitude_ratio = max_magnitude / avg_magnitude.max(1.0);
    if magnitude_ratio < 3.0 {
        (0.2, "Consistent energy")
    } else if magnitude_ratio > 10.0 {
        (-0.1, "Sharp peak (possible sidelobe)")
    } else {
        (0.0, "")
    }
}

pub(crate) fn calculate_center_proximity_score(
    target_freq_mhz: f64,
    center_freq_mhz: f64,
) -> (f64, &'static str) {
    let dist_from_center_mhz = (target_freq_mhz - center_freq_mhz).abs();
    if dist_from_center_mhz <= 0.1 {
        (0.4, "Near center freq")
    } else if dist_from_center_mhz <= 0.3 {
        (0.1, "")
    } else if dist_from_center_mhz > 0.4 {
        (-0.2, "Far from center")
    } else {
        (0.0, "")
    }
}

pub(crate) fn calculate_signal_strength_score(max_magnitude: f32) -> (f64, &'static str) {
    if max_magnitude > 500.0 {
        (0.2, "Strong signal")
    } else if max_magnitude < 100.0 {
        (-0.1, "Weak signal")
    } else {
        (0.0, "")
    }
}
