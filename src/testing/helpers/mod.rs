pub mod audio_testing;
pub mod file_sources_audio;
pub mod file_sources_iq;
pub mod fixtures;
pub mod framework;
pub mod metadata;
pub mod mock_sources;
pub mod stream_adapters;
pub mod trait_def;

// Re-export commonly used types and functions
pub use audio_testing::assert_classifies_audio;
pub use file_sources_audio::AudioFileSource;
pub use file_sources_iq::FileSampleSource;
#[cfg(test)]
pub use fixtures::load_audio_fixture;
pub use fixtures::load_iq_fixture;
pub use framework::{
    FrequencyTranslationResult, PipelineTestResult, ScanningMode, TestPeakResult,
    init_test_logging, test_peak_detection_isolated, with_captured_logs,
};
pub use metadata::{AudioFileMetadata, IqFileMetadataExt};
pub use mock_sources::MockSampleSource;
pub use stream_adapters::SdrStreamSource;
pub use trait_def::SampleSource;
