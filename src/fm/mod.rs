use crate::{
    fm::deemph::Deemphasis,
    logging,
    mpsc::{MpscReceiverSource, MpscSenderSink, MpscSink},
    soapy::SdrSource,
    types::{self, Peak, Result},
};
use phf::phf_map;
use rustradio::fir;
use rustradio::graph::{Graph, GraphRunner};
use rustradio::window::WindowType;
use rustradio::{Complex, Float, blockchain};
use rustradio::{
    blocks::{FftFilter, QuadratureDemod, RationalResampler},
    fir::FirFilter,
};
use std::{thread, time::Duration};
use tracing::{debug, info};

pub mod deemph;

static PEAK_SCAN_DURATIONS: phf::Map<&'static str, f64> = phf_map! {
    "driver=sdrplay" => 0.45,
};

const DEFAULT_PEAK_SCAN_DURATION: f64 = 0.5;

#[derive(Debug, Clone)]
pub struct Candidate {
    pub frequency_hz: f64,
    pub peak_count: usize,
    pub max_magnitude: f32,
    pub avg_magnitude: f32,
    pub signal_strength: String,
}

impl Candidate {
    pub fn analyze(
        &self,
        config: &crate::ScanningConfig,
        audio_tx: std::sync::mpsc::SyncSender<f32>,
    ) -> Result<()> {
        info!(
            message = format!("found {:.1} MHz", self.frequency_hz / 1e6),
            frequency_hz = self.frequency_hz
        );
        logging::flush();

        debug!(
            message = "Tuning into station",
            fequency_hz = self.frequency_hz / 1e6,
            signal_strength = self.signal_strength,
            peak_count = self.peak_count,
            self.max_magnitude = self.max_magnitude,
            self.avg_magnitude = self.avg_magnitude
        );
        let lock = config.audo_mutex.lock().unwrap();

        // Create a complete SDR -> DSP -> Audio pipeline for this candidate
        let (sdr_tx, sdr_rx) = std::sync::mpsc::sync_channel::<rustradio::Complex>(16384);
        let station_name = format!("{:.1}FM", self.frequency_hz / 1e6);

        // Create DSP graph to process this candidate's frequency
        let mut dsp_graph = create_station_graph(sdr_rx, config.samp_rate, audio_tx, station_name)?;
        let dsp_cancel_token = dsp_graph.cancel_token();

        // Create SDR graph to capture this frequency
        let mut sdr_graph = Graph::new();
        let sdr_graph_cancel_token = sdr_graph.cancel_token();

        debug!(
            "Frequency {}, sample rate {}",
            self.frequency_hz, config.samp_rate
        );

        let sdr_source = SdrSource::when_ready(config.driver.clone())?;
        let (sdr_source_block, sdr_output_stream) =
            sdr_source.create_source_block(self.frequency_hz, config.samp_rate)?;

        sdr_graph.add(Box::new(sdr_source_block));
        sdr_graph.add(Box::new(MpscSenderSink::new(sdr_output_stream, sdr_tx)));

        // Start DSP graph in a separate thread
        let frequency_hz = self.frequency_hz;
        thread::spawn(move || {
            if let Err(e) = dsp_graph.run() {
                debug!("DSP graph error for {}: {}", frequency_hz / 1e6, e);
            }
        });

        // Set up timer thread
        let duration = config.duration;
        thread::spawn(move || {
            thread::sleep(Duration::from_secs(duration));
            debug!(
                "Ran for {} seconds, continuing on to the next candidate",
                duration
            );
            sdr_graph_cancel_token.cancel();
            dsp_cancel_token.cancel();
        });

        sdr_graph.run()?;
        drop(lock);

        Ok(())
    }
}

/// Collect RF peaks synchronously by directly interfacing with the SDR device
/// This performs FFT analysis to detect spectral peaks above the threshold
pub fn collect_peaks(
    config: &crate::ScanningConfig,
    device_args: &str,
    center_freq: f64,
) -> Result<Vec<Peak>> {
    let mut sample_source =
        crate::soapy::SdrSampleSource::new(device_args.to_string(), center_freq, config.samp_rate)?;

    // Create a modified config with device-specific peak scan duration if not already set
    let mut modified_config = config.clone();
    if modified_config.peak_scan_duration.is_none() {
        modified_config.peak_scan_duration = Some(
            *PEAK_SCAN_DURATIONS
                .get(device_args)
                .unwrap_or(&DEFAULT_PEAK_SCAN_DURATION),
        );
    }

    collect_peaks_from_source(&modified_config, &mut sample_source)
}

/// Collect RF peaks from any SampleSource (for testing and production)
pub fn collect_peaks_from_source(
    config: &crate::ScanningConfig,
    sample_source: &mut dyn crate::types::SampleSource,
) -> Result<Vec<Peak>> {
    let peak_scan_duration = config
        .peak_scan_duration
        .unwrap_or(DEFAULT_PEAK_SCAN_DURATION);
    debug!("Starting peak detection scan for {peak_scan_duration} seconds...",);
    let lock = config.audo_mutex.lock().unwrap();

    // Prepare FFT processing
    let mut peaks = Vec::new();
    let mut fft_buffer = vec![rustfft::num_complex::Complex32::default(); config.fft_size];
    let mut planner = rustfft::FftPlanner::new();
    let fft = planner.plan_fft_forward(config.fft_size);

    // Calculate sampling parameters
    let samples_per_second = sample_source.sample_rate() as usize;
    let total_samples_needed = (samples_per_second as f64 * peak_scan_duration) as usize;
    let mut samples_collected = 0;
    let mut read_buffer = vec![Complex::default(); config.fft_size];

    // Collect samples and perform peak detection
    while samples_collected < total_samples_needed {
        match sample_source.read_samples(&mut read_buffer) {
            Ok(samples_read) => {
                if samples_read == 0 {
                    continue;
                }

                // Copy samples to FFT buffer and process
                for (i, sample) in read_buffer
                    .iter()
                    .take(samples_read.min(config.fft_size))
                    .enumerate()
                {
                    fft_buffer[i] = rustfft::num_complex::Complex32::new(sample.re, sample.im);
                }

                fft.process(&mut fft_buffer);
                let magnitudes: Vec<f32> = fft_buffer.iter().map(|c| c.norm_sqr()).collect();

                // Detect peaks: local maxima above threshold
                for i in 1..magnitudes.len() - 1 {
                    if magnitudes[i] > config.peak_detection_threshold
                        && magnitudes[i] > magnitudes[i - 1]
                        && magnitudes[i] > magnitudes[i + 1]
                    {
                        // Convert FFT bin to frequency
                        let freq_offset =
                            (i as f64 / config.fft_size as f64) * sample_source.sample_rate();
                        let freq_hz = sample_source.center_frequency()
                            - (sample_source.sample_rate() / 2.0)
                            + freq_offset;

                        peaks.push(Peak {
                            frequency_hz: freq_hz,
                            magnitude: magnitudes[i],
                        });
                    }
                }

                samples_collected += samples_read;
            }
            Err(e) => {
                debug!("Error reading from SDR: {}", e);
                break;
            }
        }
    }

    sample_source.deactivate()?;
    drop(lock);
    debug!("Peak detection scan complete. Found {} peaks.", peaks.len());

    Ok(peaks)
}

/// Analyze spectral characteristics around a frequency to determine if it's a main lobe or sidelobe
/// Main lobes are wider and have characteristic spectral patterns compared to sidelobes
fn analyze_spectral_characteristics(
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

    if nearby_peaks.is_empty() {
        return (0.0, "No signal".to_string());
    }

    // Sort peaks by frequency for width analysis
    let mut sorted_peaks = nearby_peaks.clone();
    sorted_peaks.sort_by(|a, b| a.frequency_hz.partial_cmp(&b.frequency_hz).unwrap());

    // Calculate spectral width characteristics
    let peak_count = sorted_peaks.len();
    let freq_span_khz = if peak_count > 1 {
        (sorted_peaks.last().unwrap().frequency_hz - sorted_peaks.first().unwrap().frequency_hz)
            / 1000.0
    } else {
        0.0
    };

    // Find the strongest peak in the group (should be the main signal)
    let max_magnitude = sorted_peaks
        .iter()
        .map(|p| p.magnitude)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    // Calculate average magnitude
    let avg_magnitude = sorted_peaks.iter().map(|p| p.magnitude).sum::<f32>() / peak_count as f32;

    // Main lobe characteristics analysis
    let mut score: f64 = 0.0;
    let mut analysis_notes = Vec::new();

    // 1. Peak density analysis (main lobes have consistent energy distribution)
    let peak_density = peak_count as f64 / freq_span_khz.max(1.0);
    if peak_density > 20.0 && peak_density < 200.0 {
        score += 0.3;
        analysis_notes.push("Good peak density");
    } else if peak_density > 200.0 {
        score -= 0.2; // Too many peaks suggests broadband interference
        analysis_notes.push("High peak density (interference?)");
    }

    // 2. Frequency span analysis (main lobes have characteristic widths)
    // FM broadcast stations typically show energy across 150-200 kHz
    if freq_span_khz > 80.0 && freq_span_khz < 250.0 {
        score += 0.3;
        analysis_notes.push("Appropriate spectral width");
    } else if freq_span_khz < 30.0 {
        score -= 0.3; // Too narrow suggests sidelobe
        analysis_notes.push("Narrow spectral width (sidelobe?)");
    }

    // 3. Signal strength and consistency
    let magnitude_ratio = max_magnitude / avg_magnitude.max(1.0);
    if magnitude_ratio < 3.0 {
        score += 0.2; // Consistent energy suggests main lobe
        analysis_notes.push("Consistent energy");
    } else if magnitude_ratio > 10.0 {
        score -= 0.1; // Single spike suggests sidelobe
        analysis_notes.push("Sharp peak (possible sidelobe)");
    }

    // 4. Distance from center frequency (closer = more likely to be legitimate)
    let center_freq_mhz = center_freq / 1e6;
    let dist_from_center_mhz = (target_freq_mhz - center_freq_mhz).abs();
    if dist_from_center_mhz <= 0.1 {
        score += 0.4; // Strong bonus for center frequency
        analysis_notes.push("Near center freq");
    } else if dist_from_center_mhz <= 0.3 {
        score += 0.1; // Moderate bonus for nearby frequencies
    } else if dist_from_center_mhz > 0.4 {
        score -= 0.2; // Penalty for distant frequencies
        analysis_notes.push("Far from center");
    }

    // 5. Absolute signal strength
    if max_magnitude > 500.0 {
        score += 0.2;
        analysis_notes.push("Strong signal");
    } else if max_magnitude < 100.0 {
        score -= 0.1;
        analysis_notes.push("Weak signal");
    }

    let analysis_summary = analysis_notes.join(", ");
    (score.clamp(0.0, 1.0) as f32, analysis_summary)
}

/// Detect FM radio stations using spectral analysis with main lobe vs sidelobe discrimination
/// This approach analyzes spectral characteristics like peak width, density, and shape
pub fn find_candidates(
    peaks: &[Peak],
    config: &crate::ScanningConfig,
    center_freq: f64,
    candidate_tx: &std::sync::mpsc::SyncSender<types::Candidate>,
) {
    debug!("Using spectral analysis for FM station detection with sidelobe discrimination...");

    // Calculate the frequency range we scanned based on center freq and sample rate
    let scan_range_mhz = config.samp_rate / 2e6; // Half sample rate in MHz (Nyquist)
    let freq_start_mhz = (center_freq / 1e6) - scan_range_mhz;
    let freq_end_mhz = (center_freq / 1e6) + scan_range_mhz;

    debug!(
        "Analyzing spectral patterns in range: {:.1} - {:.1} MHz",
        freq_start_mhz, freq_end_mhz
    );

    // Generate possible FM frequencies within our scan range
    // FM stations are at odd tenth frequencies: 88.1, 88.3, 88.5, etc.
    let mut fm_freq = (freq_start_mhz * 10.0).ceil() / 10.0;
    if (fm_freq * 10.0) as i32 % 2 == 0 {
        fm_freq += 0.1; // Make it an odd tenth
    }

    while fm_freq <= freq_end_mhz {
        debug!("Analyzing {:.1} MHz... ", fm_freq);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let (spectral_score, analysis_summary) =
            analyze_spectral_characteristics(peaks, fm_freq, config.samp_rate, center_freq);

        debug!("score: {:.3} ({})", spectral_score, analysis_summary);

        // Only consider frequencies with significant spectral score
        if spectral_score > 0.3 {
            // Map spectral score to traditional signal strength categories
            let signal_strength = if spectral_score > 0.8 {
                "Strong"
            } else if spectral_score > 0.6 {
                "Medium"
            } else {
                "Weak"
            };

            // Find relevant peaks for this frequency
            let tolerance_mhz = 0.1;
            let nearby_peaks: Vec<&Peak> = peaks
                .iter()
                .filter(|peak| {
                    let peak_freq_mhz = peak.frequency_hz / 1e6;
                    (peak_freq_mhz - fm_freq).abs() <= tolerance_mhz
                })
                .collect();

            let peak_count = nearby_peaks.len();
            let max_magnitude = nearby_peaks
                .iter()
                .map(|p| p.magnitude)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);
            let avg_magnitude = if peak_count > 0 {
                nearby_peaks.iter().map(|p| p.magnitude).sum::<f32>() / peak_count as f32
            } else {
                0.0
            };

            candidate_tx
                .send(types::Candidate::Fm(Candidate {
                    frequency_hz: fm_freq * 1e6,
                    peak_count,
                    max_magnitude,
                    avg_magnitude,
                    signal_strength: signal_strength.to_string(),
                }))
                .expect("Failed to send candidate");
        }

        fm_freq = (fm_freq * 10.0 + 2.0) / 10.0; // Next odd tenth (add 0.2)
    }
}

// Create rustradio graph for a station (matches the scanner example exactly)
pub fn create_station_graph(
    source_receiver: std::sync::mpsc::Receiver<rustradio::Complex>,
    samp_rate: f64,
    audio_sender: std::sync::mpsc::SyncSender<f32>,
    channel_name: String,
) -> rustradio::Result<Graph> {
    let mut graph = Graph::new();

    // MPSC Source
    let (mpsc_source_block, prev) = MpscReceiverSource::new(source_receiver);
    graph.add(Box::new(mpsc_source_block));

    // RF Filter
    let taps = fir::low_pass_complex(
        samp_rate as _,
        120_000.0f32,
        25_000.0f32,
        &WindowType::Hamming,
    );
    let prev = blockchain![graph, prev, FftFilter::new(prev, taps)];

    // Resample to 100kHz
    let quad_rate = 240_000.0;
    let prev = blockchain![
        graph,
        prev,
        RationalResampler::<rustradio::Complex>::builder()
            .deci(samp_rate as usize)
            .interp(quad_rate as usize)
            .build(prev)?
    ];

    // Quadrature demodulation
    let prev = blockchain![graph, prev, QuadratureDemod::new(prev, 1.0)];

    // FM Deemphasis
    let (deemphasis_block, deemphasis_output_stream) = Deemphasis::new(prev, quad_rate, 75.0);
    graph.add(Box::new(deemphasis_block));
    let prev = deemphasis_output_stream;

    // Audio filter
    let taps = fir::low_pass(
        quad_rate,
        15_000.0f32,
        5_000.0f32,
        &WindowType::BlackmanHarris,
    );
    let prev = blockchain![graph, prev, FirFilter::new(prev, &taps)];

    // Resample to 48kHz audio
    let audio_rate = 48_000.0;
    let prev = blockchain![
        graph,
        prev,
        RationalResampler::<Float>::builder()
            .deci(quad_rate as usize)
            .interp(audio_rate as usize)
            .build(prev)?
    ];

    // Our custom MPSC sink
    graph.add(Box::new(MpscSink::new(prev, audio_sender, channel_name)));

    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SampleSource;

    /// Mock sample source for testing that generates a simple sine wave
    struct MockSampleSource {
        sample_rate: f64,
        center_frequency: f64,
        samples_generated: usize,
        max_samples: usize,
        phase: f32,
        frequency_offset: f32, // Hz offset from center frequency
    }

    impl MockSampleSource {
        fn new(
            sample_rate: f64,
            center_frequency: f64,
            max_samples: usize,
            signal_freq_offset: f32,
        ) -> Self {
            Self {
                sample_rate,
                center_frequency,
                samples_generated: 0,
                max_samples,
                phase: 0.0,
                frequency_offset: signal_freq_offset,
            }
        }
    }

    impl SampleSource for MockSampleSource {
        fn read_samples(&mut self, buffer: &mut [Complex]) -> Result<usize> {
            let samples_to_generate = buffer.len().min(self.max_samples - self.samples_generated);
            if samples_to_generate == 0 {
                return Ok(0);
            }

            let angular_freq =
                2.0 * std::f32::consts::PI * self.frequency_offset / self.sample_rate as f32;

            for sample in buffer.iter_mut().take(samples_to_generate) {
                // Generate complex sinusoid at specified frequency offset
                *sample = Complex::new(
                    self.phase.cos() * 0.5, // I component
                    self.phase.sin() * 0.5, // Q component
                );
                self.phase += angular_freq;
                if self.phase > 2.0 * std::f32::consts::PI {
                    self.phase -= 2.0 * std::f32::consts::PI;
                }
            }

            self.samples_generated += samples_to_generate;
            Ok(samples_to_generate)
        }

        fn sample_rate(&self) -> f64 {
            self.sample_rate
        }

        fn center_frequency(&self) -> f64 {
            self.center_frequency
        }

        fn deactivate(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_collect_peaks_from_mock_source() {
        let config = crate::ScanningConfig {
            audio_buffer_size: 4096,
            audio_sample_rate: 48000,
            audo_mutex: std::sync::Arc::new(std::sync::Mutex::new(crate::Audio)),
            band: crate::Band::Fm,
            driver: "test".to_string(),
            duration: 1,
            exit_early: false,
            fft_size: 1024,
            peak_detection_threshold: 0.01, // Low threshold for testing
            peak_scan_duration: Some(0.1),  // Short duration for testing
            print_candidates: false,
            samp_rate: 1000000.0,
        };

        // Create mock source with a signal at +100kHz offset from center
        let mut mock_source = MockSampleSource::new(
            1000000.0,  // 1 MHz sample rate
            88900000.0, // 88.9 MHz center frequency
            100000,     // 100k samples max
            100000.0,   // 100 kHz offset signal
        );

        let peaks = collect_peaks_from_source(&config, &mut mock_source).unwrap();

        // Should detect the peak around 89.0 MHz (88.9 + 0.1)
        assert!(!peaks.is_empty(), "Should detect at least one peak");

        let target_freq = 89000000.0; // 89.0 MHz
        let found_peak = peaks
            .iter()
            .find(|p| (p.frequency_hz - target_freq).abs() < 50000.0);
        assert!(
            found_peak.is_some(),
            "Should find peak near 89.0 MHz, found peaks at: {:?}",
            peaks
                .iter()
                .map(|p| p.frequency_hz / 1e6)
                .collect::<Vec<_>>()
        );
    }
}
