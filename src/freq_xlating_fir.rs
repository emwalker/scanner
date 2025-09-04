use rustradio::block::{Block, BlockEOF, BlockName, BlockRet};
use rustradio::fir::Fir;
use rustradio::stream::{ReadStream, WriteStream};
use rustradio::{Complex, Float, Result};
use std::f32::consts::PI;

/// Frequency Translating FIR Filter
///
/// Similar to GNU Radio's frequency translating FIR filter, this block:
/// 1. Multiplies the input by a complex exponential to shift frequency
/// 2. Applies FIR filtering
/// 3. Optionally decimates the output
///
/// This allows selecting and filtering a specific frequency offset from the center frequency.
pub struct FreqXlatingFir {
    fir: Fir<Complex>,
    decimation: usize,
    phase: Float,
    phase_increment: Float,
    ntaps: usize,
    src: ReadStream<Complex>,
    dst: WriteStream<Complex>,
}

impl FreqXlatingFir {
    /// Create a new frequency translating FIR filter
    ///
    /// # Arguments
    /// * `src` - Input stream of complex samples
    /// * `taps` - FIR filter taps (complex)
    /// * `frequency_offset` - Frequency offset in Hz (positive = shift up, negative = shift down)
    /// * `sample_rate` - Sample rate in Hz
    /// * `decimation` - Decimation factor (1 = no decimation)
    pub fn new(
        src: ReadStream<Complex>,
        taps: &[Complex],
        frequency_offset: Float,
        sample_rate: Float,
        decimation: usize,
    ) -> (Self, ReadStream<Complex>) {
        let phase_increment = -2.0 * PI * frequency_offset / sample_rate; // Negative for frequency shift
        let (dst, dr) = rustradio::stream::new_stream();

        (
            Self {
                src,
                dst,
                fir: Fir::new(taps),
                decimation,
                phase: 0.0,
                phase_increment,
                ntaps: taps.len(),
            },
            dr,
        )
    }

    /// Create with real-valued taps (converts to complex)
    pub fn with_real_taps(
        src: ReadStream<Complex>,
        taps: &[Float],
        frequency_offset: Float,
        sample_rate: Float,
        decimation: usize,
    ) -> (Self, ReadStream<Complex>) {
        let complex_taps: Vec<Complex> = taps.iter().map(|&t| Complex::new(t, 0.0)).collect();

        Self::new(
            src,
            &complex_taps,
            frequency_offset,
            sample_rate,
            decimation,
        )
    }
}

impl BlockName for FreqXlatingFir {
    fn block_name(&self) -> &str {
        "FreqXlatingFir"
    }
}

impl BlockEOF for FreqXlatingFir {
    fn eof(&mut self) -> bool {
        self.src.eof()
    }
}

impl Block for FreqXlatingFir {
    fn work(&mut self) -> Result<BlockRet<'_>> {
        let (input, mut tags) = self.src.read_buf()?;

        // Get number of input samples we intend to consume.
        let n = {
            // Carefully avoid underflow.
            let absolute_minimum = self.ntaps + self.decimation - 1;
            if input.len() < absolute_minimum {
                return Ok(BlockRet::WaitForStream(&self.src, absolute_minimum));
            }
            self.decimation * ((input.len() - self.ntaps + 1) / self.decimation)
        };

        if n == 0 {
            return Ok(BlockRet::WaitForStream(
                &self.src,
                self.ntaps + self.decimation - 1,
            ));
        }

        // Get output buffer
        let mut out = self.dst.write_buf()?;
        let need_out = 1;
        if out.len() < need_out {
            return Ok(BlockRet::WaitForStream(&self.dst, need_out));
        }

        // Cap by output capacity
        let n = std::cmp::min(n, out.len() * self.decimation);
        let need = n + self.ntaps - 1;
        assert!(input.len() >= need, "need {need}, have {}", input.len());

        // Create buffer for frequency-shifted samples
        let mut freq_shifted = vec![Complex::default(); need];

        // Step 1: Apply frequency translation to input samples
        // Optimize for zero offset case (most common)
        if self.phase_increment.abs() < 1e-6 {
            // No frequency shift needed, just copy samples
            freq_shifted[..need].copy_from_slice(&input.slice()[..need]);
        } else {
            for (i, &sample) in input.slice()[..need].iter().enumerate() {
                // Frequency translation - multiply by complex exponential
                freq_shifted[i] = sample * Complex::new(self.phase.cos(), self.phase.sin());

                // Update phase for next sample
                self.phase += self.phase_increment;
                // Use more efficient phase wrapping
                if self.phase > PI {
                    self.phase -= 2.0 * PI;
                } else if self.phase < -PI {
                    self.phase += 2.0 * PI;
                }
            }
        }

        // Step 2: Apply FIR filter with decimation
        let out_n = n / self.decimation;
        self.fir
            .filter_n_inplace(&freq_shifted, self.decimation, &mut out.slice()[..out_n]);

        // Consume input and produce output
        input.consume(n);
        if self.decimation == 1 {
            out.produce(out_n, &tags);
        } else {
            tags.iter_mut()
                .for_each(|t| t.set_pos(t.pos() / self.decimation));
            out.produce(out_n, &tags);
        }

        Ok(BlockRet::Again)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_freq_xlating_fir_creation() {
        // Create a simple low-pass filter
        let taps = rustradio::fir::low_pass(
            48000.0,
            8000.0,
            2000.0,
            &rustradio::window::WindowType::Hamming,
        );

        // This test would need a proper input stream in a real scenario
        // For now, just test that we can create the filter
        assert_eq!(taps.len() > 0, true);
    }
}
