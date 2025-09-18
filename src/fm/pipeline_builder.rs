use crate::fm::filter_config::{FilterPurpose, FmFilterConfig};
use crate::types::ScanningConfig;
use rustradio::{
    Complex, Float, blockchain,
    blocks::RationalResampler,
    fir::{self, FirFilter},
    graph::{Graph, GraphRunner},
    stream::ReadStream,
    window::WindowType,
};
use tracing::debug;

/// Shared FM pipeline building blocks to eliminate duplication between detection and audio pipelines
pub struct FmPipelineBuilder;

impl FmPipelineBuilder {
    /// Create frequency xlating filter stage with optimized FM parameters
    /// Returns (output_stream, decimation_factor)
    pub fn create_frequency_xlating_filter(
        input: ReadStream<Complex>,
        graph: &mut Graph,
        frequency_offset: f64,
        config: &ScanningConfig,
        purpose: FilterPurpose,
    ) -> rustradio::Result<(ReadStream<Complex>, u32)> {
        let filter_config = FmFilterConfig::for_purpose(purpose, config.samp_rate);

        debug!(
            message = "FM filter configuration",
            purpose = format!("{:?}", purpose),
            passband_khz = filter_config.channel_bandwidth / 1000.0,
            transition_khz = filter_config.transition_width / 1000.0,
            decimation = filter_config.decimation,
            estimated_taps = filter_config.estimated_taps,
            estimated_mflops = filter_config.estimated_mflops
        );

        // Verify we can handle the required frequency offset
        if !filter_config.can_handle_offset(frequency_offset, config.samp_rate) {
            debug!(
                message = "Frequency offset may exceed filter passband",
                frequency_offset_khz = frequency_offset / 1000.0,
                filter_cutoff_khz = filter_config.cutoff_frequency() / 1000.0
            );
        }

        let taps = fir::low_pass(
            config.samp_rate as f32,
            filter_config.cutoff_frequency(),
            filter_config.transition_width,
            &WindowType::Hamming,
        );

        // Debug: Check filter tap count vs estimation
        let tap_error_percent = ((taps.len() as f32 - filter_config.estimated_taps as f32)
            / filter_config.estimated_taps as f32
            * 100.0)
            .abs();
        debug!(
            message = "Filter tap verification",
            actual_taps = taps.len(),
            estimated_taps = filter_config.estimated_taps,
            error_percent = tap_error_percent
        );

        let decimation = filter_config.decimation;
        let (freq_xlating_block, output) = crate::freq_xlating_fir::FreqXlatingFir::with_real_taps(
            input,
            &taps,
            frequency_offset as f32,
            config.samp_rate as f32,
            decimation,
        );
        graph.add(Box::new(freq_xlating_block));

        Ok((output, decimation.try_into().unwrap()))
    }

    /// Create audio decimation chain (anti-aliasing filter + rational resampler)
    /// This converts from FM demodulated rate to final audio sample rate
    pub fn create_audio_decimation_chain(
        input: ReadStream<Float>,
        graph: &mut Graph,
        quad_rate: f32,
        config: &ScanningConfig,
        stage_name: &str,
    ) -> rustradio::Result<ReadStream<Float>> {
        let target_audio_rate = config.audio_sample_rate as f32;
        let resampling_ratio = target_audio_rate / quad_rate;

        debug!(
            "{} audio resampling: {:.1} kHz → {:.1} kHz (ratio {:.4})",
            stage_name,
            quad_rate / 1000.0,
            target_audio_rate / 1000.0,
            resampling_ratio
        );

        // Anti-aliasing filter before resampling
        let nyquist_freq = target_audio_rate / 2.0;
        let audio_cutoff = (nyquist_freq * 0.8).min(15_000.0);
        let transition_width = nyquist_freq * 0.4;

        let taps = fir::low_pass(
            quad_rate,
            audio_cutoff,
            transition_width,
            &WindowType::Hamming,
        );
        let prev = blockchain![graph, input, FirFilter::new(input, &taps)];

        // Calculate optimal rational resampler ratios dynamically
        let (interp, deci) = config.calculate_resampler_ratios(quad_rate);
        let actual_output_rate = quad_rate * (interp as f32 / deci as f32);

        debug!(
            "{} rational resampler: {}:{} ratio, output rate: {:.1} Hz",
            stage_name, interp, deci, actual_output_rate
        );

        let output = blockchain![
            graph,
            prev,
            RationalResampler::<Float>::builder()
                .interp(interp)
                .deci(deci)
                .build(prev)?
        ];

        Ok(output)
    }
}
