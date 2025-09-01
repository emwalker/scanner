use rustradio::block::{Block, BlockEOF, BlockName, BlockRet};
use rustradio::stream::{ReadStream, WriteStream};
use rustradio::{Float, Result};

pub struct Deemphasis {
    input: ReadStream<Float>,
    output: WriteStream<Float>,
    alpha: Float,
    prev_output: Float,
}

impl Deemphasis {
    pub fn new(
        input: ReadStream<Float>,
        sample_rate: Float,
        time_constant_us: Float,
    ) -> (Self, ReadStream<Float>) {
        let time_constant_s = time_constant_us * 1e-6;
        let alpha = 1.0 / (time_constant_s * 2.0 * std::f32::consts::PI * sample_rate + 1.0);
        let (output, output_read_stream) = WriteStream::new();
        (
            Self {
                input,
                output,
                alpha,
                prev_output: 0.0,
            },
            output_read_stream,
        )
    }
}

impl BlockName for Deemphasis {
    fn block_name(&self) -> &str {
        "Deemphasis"
    }
}

impl BlockEOF for Deemphasis {
    fn eof(&mut self) -> bool {
        self.input.eof()
    }
}

impl Block for Deemphasis {
    fn work(&mut self) -> Result<BlockRet<'_>> {
        let (input_buf, _input_len) = self.input.read_buf()?;
        let mut output_buf = self.output.write_buf()?;

        let mut consumed = 0;
        let mut produced = 0;

        let n_samples = std::cmp::min(input_buf.len(), output_buf.len());

        for i in 0..n_samples {
            let input_sample = input_buf.slice()[i];
            let output_sample = self.alpha * input_sample + (1.0 - self.alpha) * self.prev_output;
            output_buf.slice()[i] = output_sample;
            self.prev_output = output_sample;
            consumed += 1;
            produced += 1;
        }

        input_buf.consume(consumed);
        output_buf.produce(produced, &[]); // Pass an empty slice for tags

        if self.input.eof() {
            Ok(BlockRet::EOF)
        } else if consumed == 0 {
            Ok(BlockRet::WaitForStream(&self.input, 1))
        } else {
            Ok(BlockRet::Again)
        }
    }
}
