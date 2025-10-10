use super::file_sources_iq::FileSampleSource;
use super::metadata::IqFileMetadataExt;
use crate::core::types::Result;
use crate::file::IqFileMetadata;

/// Test helper to load both I/Q file and metadata in one call
pub fn load_iq_fixture(iq_file_path: &str) -> Result<(FileSampleSource, IqFileMetadata)> {
    // Derive metadata file path by replacing .iq extension with .json
    let metadata_path = iq_file_path.replace(".iq", ".json");
    let metadata = IqFileMetadata::from_file(&metadata_path)?;

    let file_source = FileSampleSource::new(
        iq_file_path,
        metadata.sample_rate,
        metadata.center_frequency,
    )?;

    Ok((file_source, metadata))
}

/// Test helper to load both audio file and metadata in one call
#[cfg(test)]
pub fn load_audio_fixture(
    audio_file_path: &str,
) -> Result<(
    super::file_sources_audio::AudioFileSource,
    super::metadata::AudioFileMetadata,
)> {
    use super::file_sources_audio::AudioFileSource;
    use super::metadata::AudioFileMetadata;

    // Derive metadata file path by replacing .audio extension with .json
    let metadata_path = audio_file_path.replace(".audio", ".json");
    let metadata = AudioFileMetadata::from_file(&metadata_path)?;
    let audio_source = AudioFileSource::new(audio_file_path, metadata.sample_rate)?;
    Ok((audio_source, metadata))
}
