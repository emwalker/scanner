use crate::core::types::{Result, ScanningConfig};
use crate::ecs::{CandidateEntity, Entities, ScanId, StationEntity};
use crate::hardware::pool::SegmentTrait;
use crate::pause_signal::PauseSignal;
use crate::scanning::window::config::WindowMetadata;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tracing::debug;

pub(super) struct CandidateProcessingContext<'a> {
    pub window_num: usize,
    pub center_freq: f64,
    pub config: &'a ScanningConfig,
    pub metadata: WindowMetadata,
    pub pause_signal: &'a Option<PauseSignal>,
    pub station_entities: &'a Option<Entities<StationEntity>>,
    #[allow(dead_code)]
    pub candidate_entities: &'a Option<Entities<CandidateEntity>>,
    pub scan_id: ScanId,
}

pub(super) fn peaks(
    station_mode: bool,
    center_freq: f64,
    config: &ScanningConfig,
    device: &dyn SegmentTrait,
) -> Result<Vec<crate::core::types::Peak>> {
    if station_mode {
        debug!(
            "Station mode: Creating direct peak for {:.1} MHz",
            center_freq / 1e6
        );
        Ok(vec![crate::core::types::Peak {
            frequency_hz: center_freq,
            magnitude: 1.0,
        }])
    } else {
        let sdr_rx_for_peaks = device.audio_subscriber();
        crate::signal::collect_peaks(config, sdr_rx_for_peaks, center_freq)
    }
}

pub(super) fn debug_peaks(
    window_num: usize,
    center_freq: f64,
    config: &ScanningConfig,
    peaks: &[crate::core::types::Peak],
) {
    if config.debug.pipeline {
        debug!(
            message = "Band scanning window analysis",
            window_number = window_num,
            window_center_mhz = center_freq / 1e6,
            peaks_found = peaks.len()
        );

        for (peak_idx, peak) in peaks.iter().enumerate() {
            debug!(
                message = "Peak detected",
                window_number = window_num,
                peak_index = peak_idx,
                frequency_mhz = peak.frequency_hz / 1e6,
                magnitude = peak.magnitude
            );
        }
    }
}

pub(super) fn candidates_from_peaks(
    station_mode: bool,
    _window_num: usize,
    center_freq: f64,
    config: &ScanningConfig,
    peaks: &[crate::core::types::Peak],
) -> Vec<crate::core::types::Candidate> {
    let mut candidates = Vec::new();

    if station_mode {
        debug!(
            "Station mode: Creating direct candidate for {:.1} MHz",
            center_freq / 1e6
        );
        candidates.push(crate::core::types::Candidate::Fm(
            crate::signal::Candidate {
                frequency_hz: center_freq,
                signal_strength: "Strong".to_string(),
                peak_count: 1,
                max_magnitude: 1.0,
                avg_magnitude: 1.0,
            },
        ));
        return candidates;
    }

    for candidate in crate::signal::find_candidates(peaks, config, center_freq) {
        let candidate_freq = candidate.frequency_hz();

        let rounded_freq = (candidate_freq / 100000.0).round() * 100000.0;
        let frequency_khz = (rounded_freq / 1000.0) as u64;

        let already_processed = {
            match crate::signal::PROCESSED_FREQUENCIES.read() {
                Ok(processed) => processed.contains(&frequency_khz),
                Err(e) => {
                    debug!(
                        error = %e,
                        "Failed to read PROCESSED_FREQUENCIES, assuming not processed"
                    );
                    false
                }
            }
        };

        if already_processed {
            debug!(
                candidate_frequency_mhz = candidate_freq / 1e6,
                "Skipping candidate creation for already processed frequency"
            );
            continue;
        }

        if config.debug.pipeline {
            let frequency_offset = candidate_freq - center_freq;
            debug!(
                message = "Candidate created",
                candidate_frequency_mhz = candidate_freq / 1e6,
                window_center_mhz = center_freq / 1e6,
                frequency_offset_khz = frequency_offset / 1e3,
                signal_strength = match &candidate {
                    crate::core::types::Candidate::Fm(fm_candidate) =>
                        &fm_candidate.signal_strength,
                }
            );
        }
        candidates.push(candidate);
    }

    candidates
}

pub(super) fn process_candidates(
    ctx: &CandidateProcessingContext,
    candidates: Vec<crate::core::types::Candidate>,
    segment: &dyn SegmentTrait,
    wait_for_threads_fn: impl FnOnce(Vec<thread::JoinHandle<Result<()>>>, Duration) -> usize,
) -> Result<Vec<crate::core::types::Signal>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let candidate_count = candidates.len();
    let mut candidate_threads = Vec::new();
    let (signal_tx, signal_rx) = std::sync::mpsc::sync_channel::<crate::core::types::Signal>(100);

    for candidate in candidates.into_iter() {
        if ctx.config.debug.print_candidates {
            tracing::info!(
                "candidate found at {:.1} MHz",
                candidate.frequency_hz() / 1e6
            );
            continue;
        }

        let freq = match &candidate {
            crate::core::types::Candidate::Fm(fm_candidate) => fm_candidate.frequency_hz,
        };

        if let Some(candidate_entities) = ctx.candidate_entities {
            use crate::ecs::CandidateId;
            let id = CandidateId::new(freq, ctx.window_num);
            if let Ok(mut entities) = candidate_entities.write()
                && let Some(entity) = entities.get_mut(&id)
            {
                entity.start_analysis();
            }
        }

        let sdr_rx = segment.audio_subscriber();
        let signal_tx_clone = signal_tx.clone();
        let config_clone = ctx.config.clone();
        let center_freq = ctx.center_freq;
        let pause_signal_clone = ctx.pause_signal.clone();
        let metadata = ctx.metadata;
        let candidate_entities_clone = ctx.candidate_entities.as_ref().map(Arc::clone);

        let handle = thread::spawn(move || -> Result<()> {
            if let Some(ref signal) = pause_signal_clone
                && signal.is_paused()
            {
                debug!("Candidate thread exiting early due to pause signal");
                return Ok(());
            }

            let context = crate::pipeline::AnalysisContext {
                config: &config_clone,
                center_freq,
                metadata,
                candidate_entities: &candidate_entities_clone,
            };
            candidate.analyze(sdr_rx, signal_tx_clone, &context)
        });
        candidate_threads.push(handle);
    }

    drop(signal_tx);

    let window_timeout = Duration::from_secs(60);
    let threads_completed = wait_for_threads_fn(candidate_threads, window_timeout);

    debug!(
        "Window {} at {:.1} MHz: {}/{} candidates completed processing",
        ctx.window_num,
        ctx.center_freq / 1e6,
        threads_completed,
        candidate_count
    );

    let mut signals = Vec::new();
    while let Ok(signal) = signal_rx.try_recv() {
        // Create StationEntity for discovered signal (pure ECS approach)
        if let Some(station_entities) = ctx.station_entities {
            let station = StationEntity::from_signal(&signal, ctx.scan_id, ctx.metadata);
            if let Ok(mut entities) = station_entities.write() {
                entities.insert(station);
                debug!(
                    frequency_mhz = signal.frequency_hz / 1e6,
                    "Created StationEntity for discovered signal"
                );
            }
        }

        // Update CandidateEntity to Signal state
        if let Some(candidate_entities) = ctx.candidate_entities {
            use crate::ecs::CandidateId;
            let id = CandidateId::new(signal.frequency_hz, ctx.window_num);
            if let Ok(mut entities) = candidate_entities.write()
                && let Some(entity) = entities.get_mut(&id)
            {
                entity.mark_as_signal(signal.audio_quality, Some(signal.signal_strength as f64));
            }
        }

        signals.push(signal);
    }

    debug!(
        "Window {} collected {} signals",
        ctx.window_num,
        signals.len()
    );

    Ok(signals)
}
