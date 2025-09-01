use std::{thread, time::Duration};

use crate::{
    fm::deemph::Deemphasis,
    mpsc::{MpscReceiverSource, MpscSenderSink, MpscSink},
    types::{Candidate, Peak, Result},
};
use rustradio::graph::{Graph, GraphRunner};
use rustradio::window::WindowType;
use rustradio::{Complex, Float, blockchain};
use rustradio::{blocks::SoapySdrSource, fir};
use rustradio::{
    blocks::{FftFilter, QuadratureDemod, RationalResampler},
    fir::FirFilter,
};

pub mod deemph;

/// Collect RF peaks synchronously by directly interfacing with the SDR device
/// This performs FFT analysis to detect spectral peaks above the threshold
pub fn collect_peaks(
    config: &crate::ScanningConfig,
    device_args: &str,
    center_freq: f64,
) -> Result<Vec<Peak>> {
    println!(
        "Starting peak detection scan for {} seconds...",
        config.peak_scan_duration
    );
    let lock = config.audo_mutex.lock().unwrap();

    // Initialize SDR device and stream
    let dev = soapysdr::Device::new(device_args)?;
    let mut rxstream = dev.rx_stream::<Complex>(&[0])?;
    rxstream.activate(None)?;

    // Prepare FFT processing
    let mut peaks = Vec::new();
    let mut fft_buffer = vec![rustfft::num_complex::Complex32::default(); config.fft_size];
    let mut planner = rustfft::FftPlanner::new();
    let fft = planner.plan_fft_forward(config.fft_size);

    // Calculate sampling parameters
    let samples_per_second = config.samp_rate as usize;
    let total_samples_needed = samples_per_second * config.peak_scan_duration as usize;
    let mut samples_collected = 0;
    let mut read_buffer = vec![Complex::default(); config.fft_size];

    // Collect samples and perform peak detection
    while samples_collected < total_samples_needed {
        match rxstream.read(&mut [&mut read_buffer], 1000000) {
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
                        let freq_offset = (i as f64 / config.fft_size as f64) * config.samp_rate;
                        let freq_hz = center_freq - (config.samp_rate / 2.0) + freq_offset;

                        peaks.push(Peak {
                            frequency_hz: freq_hz,
                            magnitude: magnitudes[i],
                        });
                    }
                }

                samples_collected += samples_read;
            }
            Err(e) => {
                eprintln!("Error reading from SDR: {}", e);
                break;
            }
        }
    }

    rxstream.deactivate(None)?;
    drop(lock);
    println!("Peak detection scan complete. Found {} peaks.", peaks.len());
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
    candidate_tx: &std::sync::mpsc::SyncSender<Candidate>,
) {
    println!("Using spectral analysis for FM station detection with sidelobe discrimination...");

    // Calculate the frequency range we scanned based on center freq and sample rate
    let scan_range_mhz = config.samp_rate / 2e6; // Half sample rate in MHz (Nyquist)
    let freq_start_mhz = (center_freq / 1e6) - scan_range_mhz;
    let freq_end_mhz = (center_freq / 1e6) + scan_range_mhz;

    println!(
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
        print!("Analyzing {:.1} MHz... ", fm_freq);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let (spectral_score, analysis_summary) =
            analyze_spectral_characteristics(peaks, fm_freq, config.samp_rate, center_freq);

        println!("score: {:.3} ({})", spectral_score, analysis_summary);

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
                .send(Candidate {
                    frequency_hz: fm_freq * 1e6,
                    peak_count,
                    max_magnitude,
                    avg_magnitude,
                    signal_strength: signal_strength.to_string(),
                })
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

pub fn analyze_channel(
    candidate: Candidate,
    config: &crate::ScanningConfig,
    audio_tx: std::sync::mpsc::SyncSender<f32>,
) -> Result<()> {
    println!(
        "Tuning into {:.1} MHz - {} signal ({} peaks, max: {:.1}, avg: {:.1})",
        candidate.frequency_hz / 1e6,
        candidate.signal_strength,
        candidate.peak_count,
        candidate.max_magnitude,
        candidate.avg_magnitude
    );
    let lock = config.audo_mutex.lock().unwrap();

    // Create a complete SDR -> DSP -> Audio pipeline for this candidate
    let (sdr_tx, sdr_rx) = std::sync::mpsc::sync_channel::<rustradio::Complex>(16384);
    let station_name = format!("{:.1}FM", candidate.frequency_hz / 1e6);

    // Create DSP graph to process this candidate's frequency
    let mut dsp_graph = create_station_graph(sdr_rx, config.samp_rate, audio_tx, station_name)?;
    let dsp_cancel_token = dsp_graph.cancel_token();

    // Create SDR graph to capture this frequency
    let mut sdr_graph = Graph::new();
    let sdr_graph_cancel_token = sdr_graph.cancel_token();

    eprintln!(
        "Frequency {}, sample rate {}",
        candidate.frequency_hz, config.samp_rate
    );

    let dev = soapysdr::Device::new(config.driver.as_str())?;
    let (sdr_source_block, sdr_output_stream) =
        SoapySdrSource::builder(&dev, candidate.frequency_hz, config.samp_rate)
            .igain(1 as _)
            .build()?;

    sdr_graph.add(Box::new(sdr_source_block));
    sdr_graph.add(Box::new(MpscSenderSink::new(sdr_output_stream, sdr_tx)));

    // Start DSP graph in a separate thread
    thread::spawn(move || {
        if let Err(e) = dsp_graph.run() {
            eprintln!(
                "DSP graph error for {}: {}",
                candidate.frequency_hz / 1e6,
                e
            );
        }
    });

    // Set up timer thread
    let duration = config.duration;
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(duration));
        println!(
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
