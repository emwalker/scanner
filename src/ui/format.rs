/// Format frequency in Hz with dot separators for tabular display: xxx.xxx.xxx.xxx
pub fn frequency_hz_tabular(hz: f64) -> String {
    let hz_u64 = hz as u64;
    let billions = hz_u64 / 1_000_000_000;
    let millions = (hz_u64 / 1_000_000) % 1000;
    let thousands = (hz_u64 / 1_000) % 1000;
    let ones = hz_u64 % 1000;

    format!(
        "{:03}.{:03}.{:03}.{:03}",
        billions, millions, thousands, ones
    )
}

/// Format frequency in Hz as human-readable label (e.g., "88.9 MHz")
pub fn frequency_hz_label(hz: f64) -> String {
    if hz >= 1e9 {
        format!("{:.2} GHz", hz / 1e9)
    } else if hz >= 1e6 {
        format!("{:.1} MHz", hz / 1e6)
    } else if hz >= 1e3 {
        format!("{:.1} kHz", hz / 1e3)
    } else {
        format!("{:.0} Hz", hz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_hz_tabular() {
        assert_eq!(frequency_hz_tabular(88.9e6), "000.088.900.000");
        assert_eq!(frequency_hz_tabular(108.0e6), "000.108.000.000");
        assert_eq!(frequency_hz_tabular(162.55e6), "000.162.550.000");
    }

    #[test]
    fn test_frequency_hz_label() {
        assert_eq!(frequency_hz_label(88.9e6), "88.9 MHz");
        assert_eq!(frequency_hz_label(108.0e6), "108.0 MHz");
        assert_eq!(frequency_hz_label(162.55e6), "162.6 MHz");
        assert_eq!(frequency_hz_label(851.0e6), "851.0 MHz");
        assert_eq!(frequency_hz_label(1.2345e9), "1.23 GHz");
    }
}
