use clap::ValueEnum;

#[derive(ValueEnum, Copy, Clone, Debug)]
pub enum Band {
    /// FM broadcast band (88-108 MHz)
    Fm,
    /// VHF aircraft band (108-137 MHz)
    Aircraft,
    /// 2-meter amateur band (144-148 MHz)
    Ham2m,
    /// NOAA weather radio (162-163 MHz)
    Weather,
    /// Marine VHF band (156-162 MHz)
    Marine,
}

impl std::fmt::Display for Band {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Band::Fm => write!(f, "fm"),
            Band::Aircraft => write!(f, "aircraft"),
            Band::Ham2m => write!(f, "ham2m"),
            Band::Weather => write!(f, "weather"),
            Band::Marine => write!(f, "marine"),
        }
    }
}

impl Band {
    pub fn frequency_range(&self) -> (f64, f64) {
        match self {
            Band::Fm => (88.0e6, 108.0e6),
            Band::Aircraft => (108.0e6, 137.0e6),
            Band::Ham2m => (144.0e6, 148.0e6),
            Band::Weather => (162.0e6, 163.0e6),
            Band::Marine => (156.0e6, 162.0e6),
        }
    }

    pub fn windows(&self, sample_rate: f64, overlap: f64) -> Vec<f64> {
        let (start_freq, end_freq) = self.frequency_range();
        let usable_bandwidth = sample_rate * 0.8; // Use 80% of bandwidth to avoid edge effects
        let step_size = usable_bandwidth * (1.0 - overlap); // Step size based on overlap percentage

        let mut windows = Vec::new();
        let mut center_freq = start_freq + (usable_bandwidth / 2.0);

        while center_freq - (usable_bandwidth / 2.0) < end_freq {
            windows.push(center_freq);
            center_freq += step_size;
        }

        windows
    }
}
