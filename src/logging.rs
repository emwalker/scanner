use crate::types::{Format, Logger, Result};
use gag::Gag;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use tracing::Level;
use tracing_subscriber::FmtSubscriber;
use tracing_subscriber::fmt::MakeWriter;

// Immediate flush logging - writes directly to tty/stdout

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

/// A custom writer that can either buffer logs for testing or write directly for immediate output
pub struct TestWriter {
    buffer: Option<LogBuffer>,
}

impl TestWriter {
    pub fn new(buffer: LogBuffer) -> Self {
        Self {
            buffer: Some(buffer),
        }
    }

    /// Create a writer that outputs immediately (for main application)
    pub fn new_immediate() -> Self {
        Self { buffer: None }
    }
}

impl Write for &TestWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if let Some(buffer) = &self.buffer {
            // Buffer for tests
            let mut buffer = buffer.0.lock().unwrap();
            buffer.extend_from_slice(buf);
        } else {
            // Always write directly to tty to bypass gag redirection
            use std::fs::OpenOptions;
            match OpenOptions::new().write(true).open("/dev/tty") {
                Ok(mut tty) => {
                    tty.write_all(buf)?;
                    tty.flush()?;
                }
                Err(_) => {
                    // Fallback to stdout if TTY is not available (e.g., in tests or when piped)
                    io::stdout().write_all(buf)?;
                    io::stdout().flush()?;
                }
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Write for TestWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if let Some(buffer) = &self.buffer {
            // Buffer for tests
            let mut buffer = buffer.0.lock().unwrap();
            buffer.extend_from_slice(buf);
        } else {
            // Always write directly to tty to bypass gag redirection
            use std::fs::OpenOptions;
            match OpenOptions::new().write(true).open("/dev/tty") {
                Ok(mut tty) => {
                    tty.write_all(buf)?;
                    tty.flush()?;
                }
                Err(_) => {
                    // Fallback to stdout if TTY is not available (e.g., in tests or when piped)
                    io::stdout().write_all(buf)?;
                    io::stdout().flush()?;
                }
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        use std::fs::OpenOptions;
        match OpenOptions::new().write(true).open("/dev/tty") {
            Ok(mut tty) => {
                tty.flush()?;
            }
            Err(_) => {
                io::stdout().flush()?;
            }
        }
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

/// Immediate writer for main application (no buffering)
pub(crate) struct ImmediateWriter;

impl<'a> MakeWriter<'a> for ImmediateWriter {
    type Writer = TestWriter;

    fn make_writer(&self) -> Self::Writer {
        TestWriter::new_immediate()
    }

    fn make_writer_for(&'a self, meta: &tracing::Metadata<'_>) -> Self::Writer {
        let _ = meta;
        self.make_writer()
    }
}

// flush() function removed - logging now flushes immediately

// Default Logger implementation for production use
pub struct DefaultLogger {
    verbose: bool,
    format: Format,
}

impl DefaultLogger {
    pub fn new(verbose: bool, format: Format) -> Self {
        Self { verbose, format }
    }
}

impl Logger for DefaultLogger {
    fn init(&self) -> Result<()> {
        let level = if self.verbose {
            Level::DEBUG
        } else {
            Level::INFO
        };
        let immediate_writer = ImmediateWriter;

        match self.format {
            Format::Json => {
                let subscriber = FmtSubscriber::builder()
                    .json()
                    .with_max_level(level)
                    .with_writer(immediate_writer)
                    .finish();
                tracing::subscriber::set_global_default(subscriber)
                    .expect("setting default subscriber failed");
            }
            Format::Text => {
                let subscriber = FmtSubscriber::builder()
                    .with_max_level(level)
                    .with_writer(immediate_writer)
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
                    .with_writer(immediate_writer)
                    .with_target(false)
                    .finish();
                tracing::subscriber::set_global_default(subscriber)
                    .expect("setting default subscriber failed");
            }
        }

        Ok(())
    }
}

pub fn init(logger: &dyn Logger, verbose: bool) -> Result<()> {
    let _stdout_gag = if verbose { None } else { Some(Gag::stdout()?) };
    let _stderr_gag = if verbose { None } else { Some(Gag::stderr()?) };
    let _ = logger.init();
    soapysdr::configure_logging();
    Ok(())
}
