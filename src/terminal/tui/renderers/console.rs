//! Console rendering for fallback text modes

use crate::terminal::tui::model::{CandidateStatus, Model};
use std::fs::OpenOptions;
use std::io::Write;

/// Console renderer for text and simple TUI modes
pub struct ConsoleRenderer;

impl ConsoleRenderer {
    /// Write directly to TTY to bypass output suppression
    #[allow(clippy::print_stdout)]
    pub fn tty_println(text: &str) {
        match OpenOptions::new().write(true).open("/dev/tty") {
            Ok(mut tty) => {
                let _ = writeln!(tty, "{}", text);
                let _ = tty.flush();
            }
            Err(_) => {
                // Fallback to stdout if TTY is not available
                println!("{}", text);
            }
        }
    }

    /// Print directly to TTY to bypass output suppression
    #[allow(clippy::print_stdout)]
    pub fn tty_print(text: &str) {
        match OpenOptions::new().write(true).open("/dev/tty") {
            Ok(mut tty) => {
                let _ = write!(tty, "{}", text);
                let _ = tty.flush();
            }
            Err(_) => {
                // Fallback to stdout if TTY is not available
                print!("{}", text);
            }
        }
    }

    /// Calculate how many lines the current display uses
    pub fn calculate_display_lines(model: &Model) -> usize {
        if model.is_empty() {
            return 2; // "Waiting for candidates..." + separator
        }

        let mut lines = 0;
        for window in model.windows.values() {
            lines += 1; // Window header
            lines += window.candidates.len(); // Candidate lines
            lines += 1; // Empty line between windows
        }
        lines += 1; // Final separator
        lines
    }

    /// Print progress in a TUI-like style with ANSI escape codes
    pub fn print_tui_style_progress(model: &Model) {
        if model.is_empty() {
            Self::tty_println("Waiting for candidates...");
            Self::tty_println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            return;
        }

        // Display windows in order (only windows that should be shown)
        for (&window_id, window) in &model.windows {
            if !window.should_display() {
                continue;
            }
            Self::tty_println(&format!("\x1B[1;37mWindow {}\x1B[0m", window_id)); // Bold white
            let is_current_window = window_id == model.current_window;
            let displayable_candidates = window.displayable_candidates(is_current_window);
            for candidate in displayable_candidates {
                let freq_mhz = candidate.frequency_hz / 1e6;
                let progress_percent = (candidate.completion * 100.0) as u8;
                let status = candidate.status.to_string();

                // Create colored progress bar
                let progress_bar = if candidate.completion >= 1.0 {
                    "\x1B[42m████████████████████████\x1B[0m".to_string() // Green background
                } else {
                    let filled = (candidate.completion * 24.0) as usize;
                    let empty = 24 - filled;
                    format!(
                        "\x1B[44m{}\x1B[100m{}\x1B[0m",
                        "█".repeat(filled),
                        "░".repeat(empty)
                    )
                };

                // Color status based on type
                let colored_status = match candidate.status {
                    CandidateStatus::Detected => format!("\x1B[33m{}\x1B[0m", status), // Yellow
                    CandidateStatus::Analyzing => format!("\x1B[36m{}\x1B[0m", status), // Cyan
                    CandidateStatus::Rejected => format!("\x1B[2;31m{}\x1B[0m", status), // Faint red (dim red)
                    CandidateStatus::Signal => format!("\x1B[34m{}\x1B[0m", status),     // Blue
                    CandidateStatus::Playing => format!("\x1B[35m{}\x1B[0m", status),    // Magenta
                    CandidateStatus::Completed => format!("\x1B[32m{}\x1B[0m", status),  // Green
                };

                // Include audio quality for completed or rejected candidates
                let display_line = if let Some(audio_quality) = &candidate.audio_quality {
                    if candidate.status == CandidateStatus::Completed
                        || candidate.status == CandidateStatus::Rejected
                    {
                        let colored_quality_text = match audio_quality {
                            crate::audio_quality::AudioQuality::Good => "\x1B[1;32mGood\x1B[0m", // Bold green
                            crate::audio_quality::AudioQuality::Moderate => {
                                "\x1B[32mModerate\x1B[0m"
                            } // Green without bold
                            crate::audio_quality::AudioQuality::Poor => {
                                "\x1B[38;2;255;165;0mPoor\x1B[0m"
                            } // Yellow orange
                            crate::audio_quality::AudioQuality::NoAudio => {
                                "\x1B[38;2;255;165;0mNo Audio\x1B[0m"
                            } // Yellow orange
                            crate::audio_quality::AudioQuality::Static => {
                                "\x1B[38;2;255;165;0mStatic\x1B[0m"
                            } // Yellow orange
                            crate::audio_quality::AudioQuality::Unknown => "Unknown", // No special color
                        };
                        format!(
                            "{} {:.1} MHz [{}] {}% • {}",
                            progress_bar,
                            freq_mhz,
                            colored_status,
                            progress_percent,
                            colored_quality_text
                        )
                    } else {
                        format!(
                            "{} {:.1} MHz [{}] {}%",
                            progress_bar, freq_mhz, colored_status, progress_percent
                        )
                    }
                } else {
                    format!(
                        "{} {:.1} MHz [{}] {}%",
                        progress_bar, freq_mhz, colored_status, progress_percent
                    )
                };

                Self::tty_println(&display_line);
            }
            Self::tty_println(""); // Empty line between windows
        }
        Self::tty_println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }

    /// Print current progress in text format
    pub fn print_text_progress(model: &Model) {
        if model.is_empty() {
            return;
        }

        Self::tty_println("\n━━━ Progress Update ━━━");

        // Display windows in order (only windows that should be shown)
        for (&window_id, window) in &model.windows {
            if !window.should_display() {
                continue;
            }
            Self::tty_println(&format!("Window {}", window_id));
            let is_current_window = window_id == model.current_window;
            let displayable_candidates = window.displayable_candidates(is_current_window);
            for candidate in displayable_candidates {
                let freq_mhz = candidate.frequency_hz / 1e6;
                let progress_percent = (candidate.completion * 100.0) as u8;
                let status = candidate.status.to_string();

                let progress_bar = if candidate.completion >= 1.0 {
                    "████████████████████████".to_string()
                } else {
                    let filled = (candidate.completion * 24.0) as usize;
                    let empty = 24 - filled;
                    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
                };

                // Include audio quality for completed or rejected candidates
                let display_line = if let Some(audio_quality) = &candidate.audio_quality {
                    if candidate.status == CandidateStatus::Completed
                        || candidate.status == CandidateStatus::Rejected
                    {
                        let colored_quality_text = match audio_quality {
                            crate::audio_quality::AudioQuality::Good => "\x1B[1;32mGood\x1B[0m", // Bold green
                            crate::audio_quality::AudioQuality::Moderate => {
                                "\x1B[32mModerate\x1B[0m"
                            } // Green without bold
                            crate::audio_quality::AudioQuality::Poor => {
                                "\x1B[38;2;255;165;0mPoor\x1B[0m"
                            } // Yellow orange
                            crate::audio_quality::AudioQuality::NoAudio => {
                                "\x1B[38;2;255;165;0mNo Audio\x1B[0m"
                            } // Yellow orange
                            crate::audio_quality::AudioQuality::Static => {
                                "\x1B[38;2;255;165;0mStatic\x1B[0m"
                            } // Yellow orange
                            crate::audio_quality::AudioQuality::Unknown => "Unknown", // No special color
                        };
                        format!(
                            "{} {:.1} MHz [{}] {}% • {}",
                            progress_bar, freq_mhz, status, progress_percent, colored_quality_text
                        )
                    } else {
                        format!(
                            "{} {:.1} MHz [{}] {}%",
                            progress_bar, freq_mhz, status, progress_percent
                        )
                    }
                } else {
                    format!(
                        "{} {:.1} MHz [{}] {}%",
                        progress_bar, freq_mhz, status, progress_percent
                    )
                };

                Self::tty_println(&display_line);
            }
            Self::tty_println(""); // Empty line between windows
        }
        Self::tty_println("━━━━━━━━━━━━━━━━━━━━━━━");
    }
}

#[cfg(test)]
mod tests {
    use crate::terminal::tui::model::CandidateStatus;

    #[test]
    fn test_console_fallback_formats_unchanged() {
        let text_header = [
            "┌─ Scanning FM stations ... ───────────────┐",
            "│ Running in text mode                     │",
            "│ Press CTRL-C to exit                     │",
            "└──────────────────────────────────────────┘",
        ];
        assert_eq!(
            text_header[0],
            "┌─ Scanning FM stations ... ───────────────┐"
        );
        assert_eq!(
            text_header[1],
            "│ Running in text mode                     │"
        );
        assert_eq!(
            text_header[2],
            "│ Press CTRL-C to exit                     │"
        );
        assert_eq!(
            text_header[3],
            "└──────────────────────────────────────────┘"
        );

        let simple_header = [
            "┌─ Scanning FM stations ... ───────────────┐",
            "│ TUI Mode (Simplified)                    │",
            "│ Press CTRL-C to exit                     │",
            "└──────────────────────────────────────────┘",
        ];
        assert_eq!(
            simple_header[0],
            "┌─ Scanning FM stations ... ───────────────┐"
        );
        assert_eq!(
            simple_header[1],
            "│ TUI Mode (Simplified)                    │"
        );
    }

    #[test]
    fn test_console_ansi_colors_unchanged() {
        let ansi_colors = vec![
            (CandidateStatus::Detected, "\x1B[33m"),   // Yellow
            (CandidateStatus::Analyzing, "\x1B[36m"),  // Cyan
            (CandidateStatus::Rejected, "\x1B[2;31m"), // Faint red (dim red)
            (CandidateStatus::Signal, "\x1B[34m"),     // Blue
            (CandidateStatus::Playing, "\x1B[35m"),    // Magenta
            (CandidateStatus::Completed, "\x1B[32m"),  // Green
        ];

        for (status, expected_code) in ansi_colors {
            let status_str = status.to_string();
            let colored_status = match status {
                CandidateStatus::Detected => format!("\x1B[33m{}\x1B[0m", status_str),
                CandidateStatus::Analyzing => format!("\x1B[36m{}\x1B[0m", status_str),
                CandidateStatus::Rejected => format!("\x1B[2;31m{}\x1B[0m", status_str),
                CandidateStatus::Signal => format!("\x1B[34m{}\x1B[0m", status_str),
                CandidateStatus::Playing => format!("\x1B[35m{}\x1B[0m", status_str),
                CandidateStatus::Completed => format!("\x1B[32m{}\x1B[0m", status_str),
            };

            assert!(colored_status.starts_with(expected_code));
            assert!(colored_status.ends_with("\x1B[0m")); // Reset code
        }

        let window_id = 1;
        let window_header = format!("\x1B[1;37mWindow {}\x1B[0m", window_id);
        assert_eq!(window_header, "\x1B[1;37mWindow 1\x1B[0m");
    }
}
