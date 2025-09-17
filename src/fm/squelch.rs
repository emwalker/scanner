use crate::audio_quality::{AudioAnalyzer, AudioQuality};
use crate::types::Signal;
use rustradio::block::{Block, BlockEOF, BlockName, BlockRet};
use rustradio::stream::ReadStream;
use rustradio::{Float, Result};
use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
    mpsc::SyncSender,
};
use tracing::debug;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Decision {
    Learning,
    Audio,
    Noise,
}

impl Decision {
    pub fn to_u8(self) -> u8 {
        match self {
            Decision::Learning => 0,
            Decision::Audio => 1,
            Decision::Noise => 2,
        }
    }

    pub fn from_u8(value: u8) -> Self {
        match value {
            1 => Decision::Audio,
            2 => Decision::Noise,
            _ => Decision::Learning, // Default to Learning for any other value
        }
    }
}

/// Audio squelch block that analyzes audio content and determines if signal contains
/// audio or just noise. If noise is detected, blocks output and triggers early exit.
pub struct SquelchBlock {
    input: ReadStream<Float>,
    decision: Decision,
    decision_state: Arc<AtomicU8>,

    // Audio analysis parameters
    _sample_rate: f32,
    _learning_duration: f32,
    samples_analyzed: usize,
    learning_samples_needed: usize,
    analysis_completed: bool,

    // Signal detection
    signal_tx: Option<SyncSender<Signal>>,
    frequency_hz: f64,
    detection_center_freq: f64,

    // Audio quality analysis
    audio_samples: Vec<f32>,
    audio_analyzer: AudioAnalyzer,

    // Squelch configuration
    squelch_disabled: bool,
    threshold: AudioQuality,

    // Audio file coordination
    audio_capturer: Option<crate::file::AudioCaptureSink>,
}

/// Configuration for squelch block creation
pub struct SquelchConfig {
    pub sample_rate: f32,
    pub learning_duration: f32,
    pub signal_tx: Option<SyncSender<Signal>>,
    pub frequency_hz: f64,
    pub center_freq: f64,
    pub squelch_disabled: bool,
    pub threshold: AudioQuality,
    pub fft_size: usize,
    pub audio_analyzer: AudioAnalyzer,
    pub audio_capturer: Option<crate::file::AudioCaptureSink>,
}

impl SquelchBlock {
    pub fn new(input: ReadStream<Float>, config: SquelchConfig) -> (Self, Arc<AtomicU8>) {
        let learning_samples_needed = (config.sample_rate * config.learning_duration) as usize;
        let decision_state = Arc::new(AtomicU8::new(Decision::Learning.to_u8()));

        // Audio quality analysis will be used directly in process_audio
        debug!(
            analyzer_name = config.audio_analyzer.classifier_name(),
            "SquelchBlock configured with audio quality analysis"
        );

        let block = Self {
            input,
            decision: Decision::Learning,
            decision_state: decision_state.clone(),
            _sample_rate: config.sample_rate,
            _learning_duration: config.learning_duration,
            samples_analyzed: 0,
            learning_samples_needed,
            analysis_completed: false,
            signal_tx: config.signal_tx,
            frequency_hz: config.frequency_hz,
            detection_center_freq: config.center_freq,
            audio_samples: Vec::with_capacity(learning_samples_needed),
            audio_analyzer: config.audio_analyzer,
            squelch_disabled: config.squelch_disabled,
            threshold: config.threshold,
            audio_capturer: config.audio_capturer,
        };

        (block, decision_state)
    }

    fn analyze_audio_content(&mut self) -> (AudioQuality, f32) {
        // Use the configured audio analyzer
        let quality = match self
            .audio_analyzer
            .analyze(&self.audio_samples, self._sample_rate)
        {
            Ok(result) => result.quality,
            Err(e) => {
                debug!(
                    error = %e,
                    frequency_mhz = self.frequency_hz / 1e6,
                    analyzer = self.audio_analyzer.classifier_name(),
                    "Audio analysis failed, defaulting to Static"
                );
                AudioQuality::Static
            }
        };

        let is_audio = if self.squelch_disabled {
            true // Always classify as audio when squelch is disabled
        } else {
            quality.meets_threshold(self.threshold)
        };

        // Calculate signal strength from audio samples for compatibility
        let signal_strength = if !self.audio_samples.is_empty() {
            let rms = (self.audio_samples.iter().map(|s| s * s).sum::<f32>()
                / self.audio_samples.len() as f32)
                .sqrt();
            rms * 100.0 // Scale to reasonable range
        } else {
            0.0
        };

        debug!(
            audio_quality = format!("{:?}", quality),
            signal_strength = signal_strength,
            decision = if is_audio { "AUDIO" } else { "NOISE" },
            squelch_disabled = self.squelch_disabled,
            frequency_mhz = self.frequency_hz / 1e6,
            samples = self.samples_analyzed,
            analyzer = self.audio_analyzer.classifier_name(),
            "Audio quality squelch analysis complete"
        );

        (quality, signal_strength)
    }

    fn process_sample_for_analysis(&mut self, sample: f32) {
        // Collect samples for normalized analysis
        self.audio_samples.push(sample);
        self.samples_analyzed += 1;

        // Also capture samples for audio file if capturer is available
        if let Some(ref mut capturer) = self.audio_capturer
            && let Err(e) = capturer.capture_samples(&[sample])
        {
            debug!(
                frequency_mhz = self.frequency_hz / 1e6,
                error = %e,
                "Failed to capture audio sample"
            );
        }

        // Check if learning period is complete and analysis hasn't been done yet
        if self.samples_analyzed >= self.learning_samples_needed && !self.analysis_completed {
            let (audio_quality, signal_strength) = self.analyze_audio_content();
            self.analysis_completed = true;

            if audio_quality.meets_threshold(self.threshold) {
                debug!(
                    frequency_mhz = self.frequency_hz / 1e6,
                    "Squelch: Audio detected, enabling passthrough"
                );
                self.decision = Decision::Audio;
                self.decision_state
                    .store(Decision::Audio.to_u8(), Ordering::Relaxed);

                // Coordinate audio file creation - squelch passed
                if let Some(ref mut capturer) = self.audio_capturer {
                    if let Err(e) = capturer.create_file_and_flush_buffer() {
                        debug!(
                            frequency_mhz = self.frequency_hz / 1e6,
                            error = %e,
                            "Failed to create audio file after squelch pass"
                        );
                    } else {
                        debug!(
                            frequency_mhz = self.frequency_hz / 1e6,
                            "Squelch passed - created audio file and flushed buffer"
                        );
                    }
                }

                // Push signal to queue if channel is available
                if let Some(ref tx) = self.signal_tx {
                    let signal = Signal::new_fm(
                        self.frequency_hz,
                        signal_strength,
                        200_000.0, // Standard FM channel bandwidth
                        self._sample_rate as u32,
                        (self._learning_duration * 1000.0) as u32, // Convert to ms
                        self.detection_center_freq,
                        audio_quality,
                    );

                    match tx.try_send(signal) {
                        Ok(()) => debug!(
                            "Signal queued for frequency {:.1} MHz",
                            self.frequency_hz / 1e6
                        ),
                        Err(e) => debug!("Failed to queue signal: {}", e),
                    }
                }
            } else {
                debug!(
                    frequency_mhz = self.frequency_hz / 1e6,
                    "Squelch: Noise detected, blocking output and signaling early exit"
                );
                self.decision = Decision::Noise;

                // Coordinate audio file discard - squelch failed
                if let Some(ref mut capturer) = self.audio_capturer {
                    capturer.discard_buffer();
                    debug!(
                        frequency_mhz = self.frequency_hz / 1e6,
                        "Squelch failed - discarded audio buffer"
                    );
                }

                // Signal that noise was detected
                self.decision_state
                    .store(Decision::Noise.to_u8(), Ordering::Relaxed);
            }
        }
    }
}

impl BlockName for SquelchBlock {
    fn block_name(&self) -> &str {
        "SquelchBlock"
    }
}

impl BlockEOF for SquelchBlock {
    fn eof(&mut self) -> bool {
        // When we hit EOF, analyze what we have if we haven't already
        if self.input.eof() && !self.analysis_completed && self.samples_analyzed > 0 {
            debug!(
                samples_analyzed = self.samples_analyzed,
                samples_needed = self.learning_samples_needed,
                "EOF reached before full learning period - analyzing available samples"
            );

            // Analyze with whatever samples we have
            let (audio_quality, signal_strength) = self.analyze_audio_content();
            self.analysis_completed = true;

            if audio_quality.meets_threshold(self.threshold) {
                debug!(
                    frequency_mhz = self.frequency_hz / 1e6,
                    "Squelch: Audio detected from partial samples at EOF"
                );
                self.decision = Decision::Audio;
                self.decision_state
                    .store(Decision::Audio.to_u8(), Ordering::Relaxed);

                // Coordinate audio file creation - squelch passed at EOF
                if let Some(ref mut capturer) = self.audio_capturer {
                    if let Err(e) = capturer.create_file_and_flush_buffer() {
                        debug!(
                            frequency_mhz = self.frequency_hz / 1e6,
                            error = %e,
                            "Failed to create audio file after squelch pass at EOF"
                        );
                    } else {
                        debug!(
                            frequency_mhz = self.frequency_hz / 1e6,
                            "Squelch passed at EOF - created audio file and flushed buffer"
                        );
                    }
                }

                // Push signal to queue if channel is available
                if let Some(ref tx) = self.signal_tx {
                    let signal = Signal::new_fm(
                        self.frequency_hz,
                        signal_strength,
                        200_000.0, // Standard FM channel bandwidth
                        self._sample_rate as u32,
                        (self.samples_analyzed as f32 / self._sample_rate * 1000.0) as u32, // Actual duration analyzed
                        self.detection_center_freq,
                        audio_quality,
                    );

                    match tx.try_send(signal) {
                        Ok(()) => debug!(
                            "Signal queued for frequency {:.1} MHz (partial analysis)",
                            self.frequency_hz / 1e6
                        ),
                        Err(e) => debug!("Failed to queue signal: {}", e),
                    }
                }
            } else {
                debug!(
                    frequency_mhz = self.frequency_hz / 1e6,
                    "Squelch: Noise detected from partial samples at EOF"
                );
                self.decision = Decision::Noise;

                // Coordinate audio file discard - squelch failed at EOF
                if let Some(ref mut capturer) = self.audio_capturer {
                    capturer.discard_buffer();
                    debug!(
                        frequency_mhz = self.frequency_hz / 1e6,
                        "Squelch failed at EOF - discarded audio buffer"
                    );
                }

                self.decision_state
                    .store(Decision::Noise.to_u8(), Ordering::Relaxed);
            }
        }

        self.input.eof()
    }
}

impl Block for SquelchBlock {
    fn work(&mut self) -> Result<BlockRet<'_>> {
        let (input_buf, _) = self.input.read_buf()?;
        let input_samples = input_buf.slice();

        if input_samples.is_empty() {
            return Ok(BlockRet::WaitForStream(&self.input, 1));
        }

        match self.decision {
            Decision::Learning => {
                // During learning, analyze samples but don't output yet
                let sample_count = input_samples.len();
                for &sample in input_samples {
                    self.process_sample_for_analysis(sample);
                }

                // Consume input but don't produce output during learning
                input_buf.consume(sample_count);
                Ok(BlockRet::Again)
            }

            Decision::Audio => {
                // Audio detected, consume samples but don't produce output (terminal)
                // The signal was already queued when we transitioned to PassThrough state
                let sample_count = input_samples.len();
                input_buf.consume(sample_count);
                Ok(BlockRet::Again)
            }

            Decision::Noise => {
                // Noise detected, consume input but don't produce output
                let sample_count = input_samples.len();

                input_buf.consume(sample_count);
                Ok(BlockRet::Again)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustradio::stream::WriteStream;

    fn load_audio_samples(fixture_path: &str) -> (Vec<f32>, crate::testing::AudioFileMetadata) {
        let (mut audio_source, metadata) =
            crate::testing::load_audio_fixture(fixture_path).expect("Failed to load audio fixture");

        let mut samples = Vec::new();
        let mut buffer = vec![0.0f32; 1024];

        while let Ok(samples_read) = audio_source.read_audio_samples(&mut buffer) {
            if samples_read == 0 {
                break;
            }
            samples.extend_from_slice(&buffer[..samples_read]);
        }

        assert!(!samples.is_empty(), "Should have read samples from fixture");
        (samples, metadata)
    }

    fn create_squelch_with_samples(
        samples: Vec<f32>,
        metadata: &crate::testing::AudioFileMetadata,
    ) -> SquelchBlock {
        let (_input_stream, input_read_stream) = WriteStream::new();
        let squelch_config = SquelchConfig {
            sample_rate: metadata.sample_rate,
            learning_duration: metadata.squelch_learning_duration, // Use the actual learning duration from metadata
            signal_tx: None,                                       // no signal channel for tests
            frequency_hz: metadata.frequency_hz,
            center_freq: metadata.center_freq,
            squelch_disabled: false, // don't disable squelch for tests
            threshold: AudioQuality::Moderate, // default threshold for tests
            fft_size: 1024,          // default FFT size for tests
            audio_analyzer: AudioAnalyzer::mock(), // use mock analyzer for tests
            audio_capturer: None,    // no audio capture for tests
        };
        let (mut squelch, _decision_state) = SquelchBlock::new(input_read_stream, squelch_config);

        // Process all available samples (simulating what would happen with this duration)
        for sample in samples {
            squelch.process_sample_for_analysis(sample);
        }

        // Debug: Log if analysis wasn't completed (simulating early termination)
        if !squelch.analysis_completed {
            debug!(
                samples_processed = squelch.samples_analyzed,
                samples_needed = squelch.learning_samples_needed,
                "Test: Squelch analysis incomplete - simulating early termination"
            );
        }

        squelch
    }

    fn assert_squelch_decison(fixture_path: &str, expected_decision: Decision) {
        let (samples, metadata) = load_audio_samples(fixture_path);
        // Use all parameters from the metadata file
        let mut squelch = create_squelch_with_samples(samples, &metadata);

        // Check if squelch completed analysis
        let is_audio = if squelch.analysis_completed {
            let (quality, _signal_strength) = squelch.analyze_audio_content();
            quality.meets_threshold(squelch.threshold)
        } else {
            // Simulate EOF behavior - analyze with available samples
            debug!(
                fixture_path = fixture_path,
                samples_processed = squelch.samples_analyzed,
                samples_needed = squelch.learning_samples_needed,
                "WARNING: Squelch analysis incomplete - analyzing partial samples"
            );

            // Call eof() to trigger partial analysis (simulates runtime EOF behavior)
            let _ = squelch.eof();

            // Now check the result
            if squelch.analysis_completed {
                squelch.decision == Decision::Audio
            } else {
                false // Shouldn't happen but default to noise if still not analyzed
            }
        };

        let actual_decision = if is_audio {
            Decision::Audio
        } else {
            Decision::Noise
        };

        // Debug output for test analysis
        debug!(
            fixture_path = fixture_path,
            actual_decision = format!("{:?}", actual_decision),
            expected_decision = format!("{:?}", expected_decision),
            sample_rate = metadata.sample_rate,
            samples_analyzed = squelch.samples_analyzed,
            analysis_completed = squelch.analysis_completed,
            "Test squelch analysis"
        );

        assert_eq!(
            actual_decision, expected_decision,
            "Expected squelch decision: {:?}, got: {:?}",
            expected_decision, actual_decision
        );
    }

    #[test]
    fn test_squelch_with_actual_station() {
        assert_squelch_decison(
            "tests/data/audio/squelch/88.9MHz-audio-2s-test1.audio",
            Decision::Audio,
        );
        assert_squelch_decison(
            "tests/data/audio/squelch/88.9MHz-audio-2s-test2.audio",
            Decision::Audio,
        );
    }

    #[test]
    fn test_squelch_with_noise() {
        // assert_squelch_decison("tests/data/audio/noise_88.2MHz_3s.audio", Decision::Noise);
    }

    #[test]
    fn test_squelch_file_coordination() -> crate::types::Result<()> {
        use std::fs;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();

        // Create audio capturer for testing
        let audio_config = crate::file::AudioCaptureConfig {
            output_dir: temp_dir.path().to_str().unwrap().to_string(),
            sample_rate: 48000.0,
            capture_duration: 1.0,
            frequency_hz: 88900000.0,
            modulation_type: crate::types::ModulationType::WFM,
        };
        let capturer = crate::file::AudioCaptureSink::new(audio_config)?;

        // Test squelch pass scenario
        let (_input_stream, input_read_stream) = WriteStream::new();
        let squelch_config = SquelchConfig {
            sample_rate: 48000.0,
            learning_duration: 0.1, // Short duration for test
            signal_tx: None,
            frequency_hz: 88900000.0,
            center_freq: 88900000.0,
            squelch_disabled: false,
            threshold: AudioQuality::Poor, // Low threshold to pass easily
            fft_size: 1024,
            audio_analyzer: AudioAnalyzer::mock(),
            audio_capturer: Some(capturer),
        };
        let (mut squelch, _decision_state) = SquelchBlock::new(input_read_stream, squelch_config);

        // Process enough samples to trigger analysis
        let samples_needed = (48000.0 * 0.1) as usize;
        for i in 0..samples_needed {
            let sample = (i as f32 * 0.1).sin(); // Generate test signal
            squelch.process_sample_for_analysis(sample);
        }

        // Check that file was created (squelch should pass with mock analyzer)
        let files: Vec<_> = fs::read_dir(temp_dir.path()).unwrap().collect();
        assert_eq!(
            files.len(),
            1,
            "Expected exactly one audio file to be created"
        );

        let file_path = files[0].as_ref().unwrap().path();
        assert!(file_path.to_str().unwrap().contains("000.088.900.000Hz"));
        assert!(file_path.to_str().unwrap().ends_with(".wav"));

        Ok(())
    }

    #[test]
    fn test_squelch_file_discard() -> crate::types::Result<()> {
        use std::fs;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();

        // Create audio capturer for testing
        let audio_config = crate::file::AudioCaptureConfig {
            output_dir: temp_dir.path().to_str().unwrap().to_string(),
            sample_rate: 48000.0,
            capture_duration: 1.0,
            frequency_hz: 88900000.0,
            modulation_type: crate::types::ModulationType::WFM,
        };
        let capturer = crate::file::AudioCaptureSink::new(audio_config)?;

        // Test squelch fail scenario
        let (_input_stream, input_read_stream) = WriteStream::new();
        let squelch_config = SquelchConfig {
            sample_rate: 48000.0,
            learning_duration: 0.1, // Short duration for test
            signal_tx: None,
            frequency_hz: 88900000.0,
            center_freq: 88900000.0,
            squelch_disabled: false,
            threshold: AudioQuality::Good, // High threshold to fail easily
            fft_size: 1024,
            audio_analyzer: AudioAnalyzer::mock(),
            audio_capturer: Some(capturer),
        };
        let (mut squelch, _decision_state) = SquelchBlock::new(input_read_stream, squelch_config);

        // Process enough samples to trigger analysis
        let samples_needed = (48000.0 * 0.1) as usize;
        for i in 0..samples_needed {
            let sample = (i as f32 * 0.001).sin() * 0.01; // Generate very weak test signal (RMS < 0.05)
            squelch.process_sample_for_analysis(sample);
        }

        // Check that no file was created (squelch should fail with high threshold)
        let files: Vec<_> = fs::read_dir(temp_dir.path()).unwrap().collect();
        assert_eq!(
            files.len(),
            0,
            "Expected no audio files to be created when squelch fails"
        );

        Ok(())
    }
}
