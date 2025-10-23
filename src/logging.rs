use std::{
    io::{self, Write},
    sync::{Arc, Mutex},
};

pub use tracing::Level;
use tracing_subscriber::{FmtSubscriber, fmt::MakeWriter};

use crate::core::types::{Format, Result};

pub fn level_from_flags(_verbose: bool, quiet: bool) -> Level {
    if quiet { Level::WARN } else { Level::DEBUG }
}

struct BrokenPipeIgnoringWriter<W> {
    inner: W,
}

impl<W: Write> Write for BrokenPipeIgnoringWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self.inner.write(buf) {
            Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(buf.len()),
            other => other,
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self.inner.flush() {
            Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(()),
            other => other,
        }
    }
}

struct SafeStderr;

impl<'a> MakeWriter<'a> for SafeStderr {
    type Writer = BrokenPipeIgnoringWriter<io::Stderr>;

    fn make_writer(&'a self) -> Self::Writer {
        BrokenPipeIgnoringWriter {
            inner: io::stderr(),
        }
    }
}

// Immediate flush logging - writes directly to tty/stdout

/// This is a shared, thread-safe buffer for captured logs.
/// We use `Arc<Mutex<...>>` to allow safe, concurrent access from different threads.
#[derive(Clone, Debug, Default)]
pub struct LogBuffer(Arc<Mutex<Vec<u8>>>);

impl LogBuffer {
    /// Consumes the buffer and returns the captured logs as a string.
    pub fn into_string(&self) -> String {
        match self.0.lock() {
            Ok(mut buffer) => {
                let s = String::from_utf8_lossy(&buffer).to_string();
                buffer.clear();
                s
            }
            Err(e) => {
                // If mutex is poisoned, return what we can recover from the poisoned state
                let buffer = e.into_inner();
                String::from_utf8_lossy(&buffer).to_string()
            }
        }
    }
}

enum WriterMode {
    Buffered(LogBuffer),
    File(Arc<Mutex<std::fs::File>>),
}

/// A custom writer that can either buffer logs for testing or write to a file
pub struct TestWriter {
    mode: WriterMode,
}

impl TestWriter {
    pub fn new(buffer: LogBuffer) -> Self {
        Self {
            mode: WriterMode::Buffered(buffer),
        }
    }

    pub fn new_file(file: Arc<Mutex<std::fs::File>>) -> Self {
        Self {
            mode: WriterMode::File(file),
        }
    }
}

impl Write for &TestWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match &self.mode {
            WriterMode::Buffered(buffer) => {
                let mut buffer = buffer
                    .0
                    .lock()
                    .map_err(|_| io::Error::other("Log buffer mutex poisoned"))?;
                buffer.extend_from_slice(buf);
            }
            WriterMode::File(file) => {
                let mut file = file
                    .lock()
                    .map_err(|_| io::Error::other("Log file mutex poisoned"))?;
                file.write_all(buf)?;
                file.flush()?;
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if let WriterMode::File(file) = &self.mode {
            file.lock()
                .map_err(|_| io::Error::other("Log file mutex poisoned"))?
                .flush()?;
        }
        Ok(())
    }
}

impl Write for TestWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match &self.mode {
            WriterMode::Buffered(buffer) => {
                let mut buffer = buffer
                    .0
                    .lock()
                    .map_err(|_| io::Error::other("Log buffer mutex poisoned"))?;
                buffer.extend_from_slice(buf);
            }
            WriterMode::File(file) => {
                let mut file = file
                    .lock()
                    .map_err(|_| io::Error::other("Log file mutex poisoned"))?;
                file.write_all(buf)?;
                file.flush()?;
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        match &self.mode {
            WriterMode::File(file) => {
                file.lock()
                    .map_err(|_| io::Error::other("Log file mutex poisoned"))?
                    .flush()?;
            }
            WriterMode::Buffered(_) => {}
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

pub fn init(level: Level, format: Format, log_file: Option<String>) -> Result<()> {
    if let Some(ref log_file_path) = log_file {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(log_file_path)?;

        match format {
            Format::Json => {
                let subscriber = FmtSubscriber::builder()
                    .json()
                    .with_max_level(level)
                    .with_writer(move || file.try_clone().unwrap())
                    .finish();
                tracing::subscriber::set_global_default(subscriber)?;
            }
            Format::Text => {
                let subscriber = FmtSubscriber::builder()
                    .with_max_level(level)
                    .with_writer(move || file.try_clone().unwrap())
                    .without_time()
                    .with_target(false)
                    .with_level(false)
                    .finish();
                tracing::subscriber::set_global_default(subscriber)?;
            }
            Format::Log => {
                let subscriber = FmtSubscriber::builder()
                    .with_max_level(level)
                    .with_writer(move || file.try_clone().unwrap())
                    .with_target(false)
                    .finish();
                tracing::subscriber::set_global_default(subscriber)?;
            }
        }
    } else {
        let safe_stderr = SafeStderr;
        match format {
            Format::Json => {
                let subscriber = FmtSubscriber::builder()
                    .json()
                    .with_max_level(level)
                    .with_writer(safe_stderr)
                    .finish();
                tracing::subscriber::set_global_default(subscriber)?;
            }
            Format::Text => {
                let subscriber = FmtSubscriber::builder()
                    .with_max_level(level)
                    .with_writer(safe_stderr)
                    .without_time()
                    .with_target(false)
                    .with_level(false)
                    .finish();
                tracing::subscriber::set_global_default(subscriber)?;
            }
            Format::Log => {
                let subscriber = FmtSubscriber::builder()
                    .with_max_level(level)
                    .with_writer(safe_stderr)
                    .with_target(false)
                    .finish();
                tracing::subscriber::set_global_default(subscriber)?;
            }
        }
    }

    Ok(())
}

/// Set SoapySDR log level to suppress unwanted C++ library output
pub fn set_soapysdr_log_level(suppress_info: bool) {
    // Import the raw SoapySDR FFI bindings
    use soapysdr_sys::*;

    unsafe {
        if suppress_info {
            // Suppress all but critical errors (includes INFO, WARNING, ERROR)
            // This prevents RtAudio "deviceId argument not found" spam during enumeration
            SoapySDR_setLogLevel(SoapySDRLogLevel_SOAPY_SDR_CRITICAL);
        } else {
            // Default level - show INFO and above
            SoapySDR_setLogLevel(SoapySDRLogLevel_SOAPY_SDR_INFO);
        }
    }
}
