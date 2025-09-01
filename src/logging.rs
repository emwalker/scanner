use crate::types::{Result, ScannerError};
use gag::Gag;
use std::io::{self, Write};
use std::sync::{Arc, Mutex, OnceLock};
use tracing::Level;
use tracing_subscriber::FmtSubscriber;
use tracing_subscriber::fmt::MakeWriter;

pub use crate::Format;

// Macros to force output to console even when stdout/stderr are gagged
macro_rules! stdout {
    ($($arg:tt)*) => {
        {
            use std::fs::OpenOptions;
            use std::io::Write;
            match OpenOptions::new().write(true).open("/dev/tty") {
                Ok(mut tty) => {
                    let _ = write!(tty, $($arg)*);
                }
                Err(_) => {
                    print!($($arg)*);
                }
            }
        }
    };
}

/// This is a shared, thread-safe buffer for captured logs.
/// We use `Arc<Mutex<...>>` to allow safe, concurrent access from different threads.
#[derive(Clone, Debug, Default)]
pub struct LogBuffer(Arc<Mutex<Vec<u8>>>);

impl LogBuffer {
    /// Consumes the buffer and returns the captured logs as a string.
    pub fn get_string(&self) -> String {
        let mut buffer = self.0.lock().unwrap();
        let s = String::from_utf8_lossy(&buffer).to_string();
        buffer.clear(); // Clear the buffer after getting the contents.
        s
    }
}

/// A custom writer that writes to our shared `LogBuffer`.
pub struct TestWriter {
    buffer: LogBuffer,
}

impl TestWriter {
    pub fn new(buffer: LogBuffer) -> Self {
        Self { buffer }
    }
}

impl Write for &TestWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut buffer = self.buffer.0.lock().unwrap();
        buffer.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Write for TestWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut buffer = self.buffer.0.lock().unwrap();
        buffer.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

// The `MakeWriter` implementation is what `tracing_subscriber` needs.
impl<'a> MakeWriter<'a> for LogBuffer {
    type Writer = TestWriter;

    fn make_writer(&self) -> Self::Writer {
        TestWriter::new(self.clone())
    }

    fn make_writer_for(&'a self, meta: &tracing::Metadata<'_>) -> Self::Writer {
        let _ = meta;
        self.make_writer()
    }
}

static BUFFER: OnceLock<LogBuffer> = OnceLock::new();

pub fn init(verbose: bool, format: Format) -> Result<()> {
    let level = if verbose { Level::DEBUG } else { Level::INFO };
    let captured_logs = LogBuffer::default();

    match format {
        Format::Json => {
            let subscriber = FmtSubscriber::builder()
                .json()
                .with_max_level(level)
                .with_writer(captured_logs.clone())
                .finish();
            tracing::subscriber::set_global_default(subscriber)
                .expect("setting default subscriber failed");
        }
        Format::Text => {
            let subscriber = FmtSubscriber::builder()
                .with_max_level(level)
                .with_writer(captured_logs.clone())
                .without_time()
                .with_target(false)
                .with_level(false)
                .finish();
            tracing::subscriber::set_global_default(subscriber)
                .expect("setting default subscriber failed");
        }
        Format::Log => {
            let subscriber = FmtSubscriber::builder()
                .with_max_level(level)
                .with_writer(captured_logs.clone())
                .with_target(false)
                .finish();
            tracing::subscriber::set_global_default(subscriber)
                .expect("setting default subscriber failed");
        }
    }

    BUFFER
        .set(captured_logs)
        .map_err(|_| ScannerError::Custom("Failed to set global log buffer".to_string()))?;

    // Suppress library output unless --verbose is specified
    let _stdout_gag = if verbose { None } else { Some(Gag::stdout()?) };
    let _stderr_gag = if verbose { None } else { Some(Gag::stderr()?) };

    Ok(())
}

pub fn flush() {
    if let Some(buffer) = BUFFER.get() {
        let logs = buffer.get_string();
        if !logs.is_empty() {
            stdout!("{}", logs);
        }
    }
}
