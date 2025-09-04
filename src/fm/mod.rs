use crate::{
    file::AudioCaptureBlock,
    fm::deemph::Deemphasis,
    fm::squelch::SquelchBlock,
    freq_xlating_fir::FreqXlatingFir,
    logging,
    mpsc::{MpscReceiverSource, MpscSenderSink, MpscSink},
    soapy::SdrSource,
    types::{self, Peak, Result},
};
use rustradio::fir;
use rustradio::graph::{Graph, GraphRunner};
use rustradio::window::WindowType;
use rustradio::{Complex, Float, blockchain};
use rustradio::{
    blocks::{QuadratureDemod, RationalResampler},
    fir::FirFilter,
};
use std::{thread, time::Duration};
use tracing::{debug, info};

pub mod deemph;
pub mod squelch;

/// Manages a single SDR source with a single active candidate sink
pub struct SdrManager {
    sdr_source: SdrSource,
    current_center_freq: f64,
    samp_rate: f64,
    current_sink: Option<std::sync::mpsc::SyncSender<rustradio::Complex>>,
    sdr_graph: Option<Graph>,
}

impl SdrManager {
    pub fn new(driver: String, samp_rate: f64) -> Result<Self> {
        let sdr_source = SdrSource::when_ready(driver)?;
        Ok(Self {
            sdr_source,
            current_center_freq: 0.0,
            samp_rate,
            current_sink: None,
            sdr_graph: None,
        })
    }

    pub fn set_center_frequency(&mut self, center_freq: f64) -> Result<()> {
        if (self.current_center_freq - center_freq).abs() > 1000.0 {
            // Only change if different by more than 1kHz
            self.current_center_freq = center_freq;
            self.rebuild_sdr_graph()?;
        }
        Ok(())
    }

    pub fn set_active_candidate(
        &mut self,
    ) -> Result<std::sync::mpsc::Receiver<rustradio::Complex>> {
        let (tx, rx) = std::sync::mpsc::sync_channel::<rustradio::Complex>(16384);
        self.current_sink = Some(tx);
        self.rebuild_sdr_graph()?;
        Ok(rx)
    }

    fn rebuild_sdr_graph(&mut self) -> Result<()> {
        if let Some(graph) = self.sdr_graph.take() {
            graph.cancel_token().cancel();
        }

        if let Some(tx) = &self.current_sink {
            let mut new_graph = Graph::new();
            let (sdr_source_block, sdr_output_stream) = self
                .sdr_source
                .create_source_block(self.current_center_freq, self.samp_rate)?;

            new_graph.add(Box::new(sdr_source_block));
            new_graph.add(Box::new(MpscSenderSink::new(sdr_output_stream, tx.clone())));

            self.sdr_graph = Some(new_graph);
        }
        Ok(())
    }

    pub fn start(&mut self) -> Result<()> {
        if let Some(ref mut graph) = self.sdr_graph {
            let _graph_handle = std::thread::spawn({
                let mut graph = std::mem::replace(graph, Graph::new());
                move || {
                    if let Err(e) = graph.run() {
                        debug!("SDR graph error: {}", e);
                    }
                }
            });
            // Note: We're not storing the handle, which means the graph runs detached
            // This matches the current architecture where SDR graphs run independently
        }
        Ok(())
    }
}

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
        sdr_rx: std::sync::mpsc::Receiver<rustradio::Complex>,
        center_freq: f64,
    ) -> Result<()> {
        debug!(
            message = "Tuning into station",
            fequency_hz = self.frequency_hz / 1e6,
            signal_strength = self.signal_strength,
            peak_count = self.peak_count,
            self.max_magnitude = self.max_magnitude,
            self.avg_magnitude = self.avg_magnitude
        );
        let lock = config.audo_mutex.lock().unwrap();

        let station_name = format!("{:.1}FM", self.frequency_hz / 1e6);

        // Create DSP graph to process this candidate's frequency
        // Use the provided center_freq and this candidate's tune frequency for the FreqXlatingFir
        let (mut dsp_graph, noise_detected, audio_started_playing) = create_station_graph(
            sdr_rx,
            config.samp_rate,
            audio_tx,
            station_name,
            config,
            center_freq,
            self.frequency_hz,
        )?;
        let dsp_cancel_token = dsp_graph.cancel_token();

        debug!(
            "Processing candidate at {:.1} MHz with center freq {:.1} MHz",
            self.frequency_hz / 1e6,
            center_freq / 1e6
        );

        // Start DSP graph in a separate thread
        let frequency_hz = self.frequency_hz;
        thread::spawn(move || {
            if let Err(e) = dsp_graph.run() {
                debug!("DSP graph error for {}: {}", frequency_hz / 1e6, e);
            }
        });

        // Set up timer thread with squelch monitoring
        let duration = config.duration;
        let noise_detected_clone = noise_detected.clone();
        let audio_started_playing_clone = audio_started_playing.clone();
        thread::spawn(move || {
            let check_interval = Duration::from_millis(100);
            let total_checks = (duration * 1000) / 100; // Total checks over duration in 100ms intervals

            let mut message_printed = false;

            for _ in 0..total_checks {
                thread::sleep(check_interval);

                // Check if squelch detected noise
                if noise_detected_clone.load(std::sync::atomic::Ordering::Relaxed) {
                    debug!("Squelch detected noise, exiting early");
                    dsp_cancel_token.cancel();
                    return;
                }

                // If audio has started playing and message hasn't been printed yet
                if audio_started_playing_clone.load(std::sync::atomic::Ordering::Relaxed)
                    && !message_printed
                {
                    info!("found {:.1} MHz", frequency_hz / 1e6);
                    logging::flush();
                    message_printed = true;
                }
            }

            debug!(
                "Ran for {} seconds, continuing on to the next candidate",
                duration
            );
            dsp_cancel_token.cancel();
        });

        // Wait for DSP to complete (the timer thread will cancel it)
        thread::sleep(Duration::from_secs(duration));
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
    let sample_source =
        crate::soapy::SdrSampleSource::new(device_args.to_string(), center_freq, config.samp_rate)?;

    // Wrap with capturing source if requested
    let mut final_source: Box<dyn crate::types::SampleSource> =
        if let Some(ref capture_file) = config.capture_iq {
            Box::new(crate::file::SampleCaptureSink::new(
                Box::new(sample_source),
                capture_file,
                config.capture_duration,
            )?)
        } else {
            Box::new(sample_source)
        };

    collect_peaks_from_source(config, &mut *final_source)
}

/// Extract peaks from FFT magnitudes
fn extract_peaks_from_magnitudes(
    magnitudes: &[f32],
    threshold: f32,
    fft_size: usize,
    sample_source: &dyn crate::types::SampleSource,
) -> Vec<Peak> {
    let mut peaks = Vec::new();

    // Detect peaks: local maxima above threshold
    for i in 1..magnitudes.len() - 1 {
        if magnitudes[i] > threshold
            && magnitudes[i] > magnitudes[i - 1]
            && magnitudes[i] > magnitudes[i + 1]
        {
            // Convert FFT bin to frequency
            let freq_offset = (i as f64 / fft_size as f64) * sample_source.sample_rate();
            let freq_hz = sample_source.center_frequency() - (sample_source.sample_rate() / 2.0)
                + freq_offset;

            // Round to nearest kHz to eliminate floating point precision errors
            let freq_hz_rounded = (freq_hz / 1000.0).round() * 1000.0;

            peaks.push(Peak {
                frequency_hz: freq_hz_rounded,
                magnitude: magnitudes[i],
            });
        }
    }

    peaks
}

/// Process a batch of samples through FFT and extract peaks
fn process_samples_for_peaks(
    read_buffer: &[Complex],
    samples_read: usize,
    fft_buffer: &mut [rustfft::num_complex::Complex32],
    fft: &std::sync::Arc<dyn rustfft::Fft<f32>>,
    config: &crate::ScanningConfig,
    sample_source: &dyn crate::types::SampleSource,
) -> Vec<Peak> {
    // Copy samples to FFT buffer
    for (i, sample) in read_buffer
        .iter()
        .take(samples_read.min(config.fft_size))
        .enumerate()
    {
        fft_buffer[i] = rustfft::num_complex::Complex32::new(sample.re, sample.im);
    }

    fft.process(fft_buffer);
    let magnitudes: Vec<f32> = fft_buffer.iter().map(|c| c.norm_sqr()).collect();

    extract_peaks_from_magnitudes(
        &magnitudes,
        config.peak_detection_threshold,
        config.fft_size,
        sample_source,
    )
}

/// Collect RF peaks from any SampleSource (for testing and production)
pub fn collect_peaks_from_source(
    config: &crate::ScanningConfig,
    sample_source: &mut dyn crate::types::SampleSource,
) -> Result<Vec<Peak>> {
    let peak_scan_duration = sample_source.peak_scan_duration();
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
                    break; // End of file reached
                }

                let batch_peaks = process_samples_for_peaks(
                    &read_buffer,
                    samples_read,
                    &mut fft_buffer,
                    &fft,
                    config,
                    sample_source,
                );
                peaks.extend(batch_peaks);

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

/// Create a candidate from peak analysis results
fn create_fm_candidate(
    frequency_mhz: f64,
    peaks: &[Peak],
    spectral_score: f32,
) -> types::Candidate {
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
            (peak_freq_mhz - frequency_mhz).abs() <= tolerance_mhz
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

    types::Candidate::Fm(Candidate {
        frequency_hz: frequency_mhz * 1e6,
        peak_count,
        max_magnitude,
        avg_magnitude,
        signal_strength: signal_strength.to_string(),
    })
}

/// Generate the next FM frequency (odd tenth increments)
fn next_fm_frequency(current_freq_mhz: f64) -> f64 {
    (current_freq_mhz * 10.0 + 2.0) / 10.0 // Next odd tenth (add 0.2)
}

/// Calculate starting FM frequency for the scan range
fn calculate_starting_fm_frequency(freq_start_mhz: f64) -> f64 {
    let mut fm_freq = (freq_start_mhz * 10.0).ceil() / 10.0;
    if (fm_freq * 10.0) as i32 % 2 == 0 {
        fm_freq += 0.1; // Make it an odd tenth
    }
    fm_freq
}

/// Detect FM radio stations using spectral analysis with main lobe vs sidelobe discrimination
/// This approach analyzes spectral characteristics like peak width, density, and shape
pub fn find_candidates(
    peaks: &[Peak],
    config: &crate::ScanningConfig,
    center_freq: f64,
) -> Vec<types::Candidate> {
    debug!("Using spectral analysis for FM station detection with sidelobe discrimination...");

    // Calculate the frequency range we scanned based on center freq and sample rate
    let scan_range_mhz = config.samp_rate / 2e6; // Half sample rate in MHz (Nyquist)
    let freq_start_mhz = (center_freq / 1e6) - scan_range_mhz;
    let freq_end_mhz = (center_freq / 1e6) + scan_range_mhz;

    debug!(
        "Analyzing spectral patterns in range: {:.1} - {:.1} MHz",
        freq_start_mhz, freq_end_mhz
    );

    let mut candidates = Vec::new();
    let mut fm_freq = calculate_starting_fm_frequency(freq_start_mhz);

    while fm_freq <= freq_end_mhz {
        debug!("Analyzing {:.1} MHz... ", fm_freq);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let (spectral_score, analysis_summary) =
            analyze_spectral_characteristics(peaks, fm_freq, config.samp_rate, center_freq);

        debug!("score: {:.3} ({})", spectral_score, analysis_summary);

        // Only consider frequencies with significant spectral score
        if spectral_score > 0.3 {
            candidates.push(create_fm_candidate(fm_freq, peaks, spectral_score));
        }

        fm_freq = next_fm_frequency(fm_freq);
    }

    candidates
}

// Create rustradio graph for a station with frequency translating filter
pub fn create_station_graph(
    source_receiver: std::sync::mpsc::Receiver<rustradio::Complex>,
    samp_rate: f64,
    audio_sender: std::sync::mpsc::SyncSender<f32>,
    channel_name: String,
    config: &crate::ScanningConfig,
    center_freq: f64,
    tune_freq: f64,
) -> rustradio::Result<(
    Graph,
    std::sync::Arc<std::sync::atomic::AtomicBool>,
    std::sync::Arc<std::sync::atomic::AtomicBool>,
)> {
    let mut graph = Graph::new();

    // MPSC Source
    let (mpsc_source_block, prev) = MpscReceiverSource::new(source_receiver);
    graph.add(Box::new(mpsc_source_block));

    // Calculate frequency offset for translating filter
    let frequency_offset = tune_freq - center_freq;

    // Frequency Translating FIR Filter with reduced computational load
    // Use wider transition width and smaller bandwidth for fewer taps
    let channel_bandwidth = 150_000.0f32; // Reduced from 200 kHz
    let transition_width = 75_000.0f32; // Increased from 50 kHz for fewer taps
    let taps = fir::low_pass(
        samp_rate as f32,
        channel_bandwidth / 2.0,
        transition_width,
        &WindowType::Hamming,
    );

    let decimation = 6; // Reduce decimation to improve quality (was 8)
    let (freq_xlating_block, prev) = FreqXlatingFir::with_real_taps(
        prev,
        &taps,
        frequency_offset as f32,
        samp_rate as f32,
        decimation,
    );
    graph.add(Box::new(freq_xlating_block));

    // Update effective sample rate after decimation
    let decimated_samp_rate = samp_rate / decimation as f64;

    // Skip additional resampling if we're already close to desired quad rate
    let quad_rate = decimated_samp_rate as f32; // Use decimated rate directly to avoid extra resampling

    // Quadrature demodulation with reduced gain to prevent distortion
    // FM deviation for broadcast is 75kHz, so gain should account for sample rate
    let fm_gain = (quad_rate / (2.0 * 75_000.0)) * 0.8; // 0.8 factor to prevent overload
    let prev = blockchain![graph, prev, QuadratureDemod::new(prev, fm_gain)];

    // FM Deemphasis
    let (deemphasis_block, deemphasis_output_stream) = Deemphasis::new(prev, quad_rate, 75.0);
    graph.add(Box::new(deemphasis_block));
    let prev = deemphasis_output_stream;

    // Audio filter - optimized for better quality without more computation
    let taps = fir::low_pass(
        quad_rate,
        12_000.0f32,          // Slightly reduce cutoff to improve stopband rejection
        8_000.0f32,           // Wider transition for fewer taps but better quality
        &WindowType::Hamming, // Hamming has less computation than BlackmanHarris
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

    // Audio capture block (captures samples while passing them through)
    // Create audio capturer if requested
    let audio_capturer = if let Some(ref capture_file) = config.capture_audio {
        match crate::file::AudioCaptureSink::new(
            capture_file,
            audio_rate as f32,
            config.capture_audio_duration,
        ) {
            Ok(capturer) => Some(capturer),
            Err(e) => {
                debug!("Failed to create audio capturer: {}", e);
                None
            }
        }
    } else {
        None
    };

    let (audio_capture_block, audio_capture_output) = AudioCaptureBlock::new(prev, audio_capturer);
    graph.add(Box::new(audio_capture_block));
    let prev = audio_capture_output;

    let (squelch_block, squelch_output, noise_detected, audio_started_playing) =
        SquelchBlock::new(prev, audio_rate as f32, config.squelch_learning_duration);
    graph.add(Box::new(squelch_block));
    let prev = squelch_output;

    // Our custom MPSC sink
    graph.add(Box::new(MpscSink::new(prev, audio_sender, channel_name)));

    Ok((graph, noise_detected, audio_started_playing))
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

        fn device_args(&self) -> &str {
            "test"
        }

        fn peak_scan_duration(&self) -> f64 {
            1.0
        }
    }

    #[test]
    fn test_collect_peaks_from_mock_source() {
        let config = crate::ScanningConfig {
            audio_buffer_size: 4096,
            audio_sample_rate: 48000,
            audo_mutex: std::sync::Arc::new(std::sync::Mutex::new(crate::Audio)),
            band: crate::Band::Fm,
            capture_audio_duration: 3.0,
            capture_audio: None,
            capture_duration: 2.0,
            capture_iq: None,
            driver: "test".to_string(),
            duration: 1,
            exit_early: false,
            fft_size: 1024,
            peak_detection_threshold: 0.01, // Low threshold for testing
            peak_scan_duration: Some(0.1),  // Short duration for testing
            print_candidates: false,
            samp_rate: 1000000.0,
            squelch_learning_duration: 2.0,
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

    #[test]
    fn test_fm_detection_from_iq_file() {
        let (mut file_source, metadata) =
            crate::testing::load_iq_fixture("tests/data/iq/fm_88.9_MHz_1s.iq")
                .expect("Failed to load I/Q fixture");
        let mut config = crate::ScanningConfig::default();
        config.driver = "test".to_string();

        let peaks = collect_peaks_from_source(&config, &mut file_source)
            .expect("Failed to collect peaks from I/Q file");
        assert!(!peaks.is_empty(), "Expected to find peaks in the I/Q file");

        let candidates = find_candidates(&peaks, &config, metadata.center_frequency);
        let candidate_freqs: Vec<f64> = candidates
            .iter()
            .map(|c| (c.frequency_hz() / 1e6 * 10.0).round() / 10.0)
            .collect();

        assert_eq!(
            candidate_freqs.len(),
            2,
            "Expected exactly 2 candidates (88.9 and 89.3 MHz), but found {} candidates at: {:?}",
            candidate_freqs.len(),
            candidate_freqs
        );
        assert!(
            candidate_freqs.contains(&88.9),
            "Expected to find 88.9 MHz station"
        );
        assert!(
            candidate_freqs.contains(&89.3),
            "Expected to find 89.3 MHz station"
        );
    }

    #[test]
    fn test_no_candidates_1() {
        let (mut file_source, metadata) =
            crate::testing::load_iq_fixture("tests/data/iq/fm_88.5_MHz_1s.iq")
                .expect("Failed to load I/Q fixture");
        let mut config = crate::ScanningConfig::default();
        config.driver = "test".to_string();

        let peaks = collect_peaks_from_source(&config, &mut file_source)
            .expect("Failed to collect peaks from I/Q file");

        let candidates: Vec<_> = find_candidates(&peaks, &config, metadata.center_frequency);
        let candidate_freqs: Vec<f64> = candidates
            .iter()
            .map(|c| (c.frequency_hz() / 1e6 * 10.0).round() / 10.0)
            .collect();

        assert_eq!(
            candidates.len(),
            metadata.expected_candidates.len(),
            "Expected {} candidates as specified in metadata, but found {} candidates at: {:?}",
            metadata.expected_candidates.len(),
            candidates.len(),
            candidate_freqs
        );
    }
}
