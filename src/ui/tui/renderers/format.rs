//! Central formatting utilities for TUI display

/// Format frequency in Hz to dotted display format
pub fn format_frequency_hz(freq_hz: f64) -> String {
    let freq_int = freq_hz as u64;
    let freq_str = freq_int.to_string();
    let len = freq_str.len();

    let mut result = String::new();
    for (i, ch) in freq_str.chars().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            result.push('.');
        }
        result.push(ch);
    }

    result
}

/// Format frequency with zero-padding and dot separators for SignalId
/// Example: 88900000.0 -> "000.088.900.000"
pub fn format_frequency_with_leading_zeros(freq_hz: f64) -> String {
    let freq_hz = freq_hz as u64;

    // Zero-pad to 12 digits (supports up to 999.999 GHz)
    let padded = format!("{:012}", freq_hz);

    // Insert dots every 3 digits from the right
    let mut result = String::new();
    for (i, ch) in padded.chars().enumerate() {
        if i > 0 && (padded.len() - i) % 3 == 0 {
            result.push('.');
        }
        result.push(ch);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_central_format_frequency_creates_dotted_format() {
        // This should be the same format as the scan progress table uses
        let result = format_frequency_hz(88_900_000.0);
        assert_eq!(result, "88.900.000");
    }

    #[test]
    fn test_central_format_frequency_handles_different_frequencies() {
        assert_eq!(format_frequency_hz(89_100_000.0), "89.100.000");
        assert_eq!(format_frequency_hz(107_900_000.0), "107.900.000");
    }

    #[test]
    fn test_central_format_frequency_handles_edge_cases() {
        assert_eq!(format_frequency_hz(1_000.0), "1.000");
        assert_eq!(format_frequency_hz(1_000_000.0), "1.000.000");
    }

    #[test]
    fn test_format_frequency_with_leading_zeros() {
        // Test the new zero-padded formatter
        assert_eq!(
            format_frequency_with_leading_zeros(88_900_000.0),
            "000.088.900.000"
        );
        assert_eq!(
            format_frequency_with_leading_zeros(107_900_000.0),
            "000.107.900.000"
        );
        assert_eq!(
            format_frequency_with_leading_zeros(1_000.0),
            "000.000.001.000"
        );
    }
}
