use crate::types::{Format, Logger, Result, ScannerError};
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
    pub fn into_string(&self) -> String {
        let mut buffer = self.0.lock().unwrap();
        let s = String::from_utf8_lossy(&buffer).to_string();
        buffer.clear(); // Clear the buffer after getting the contents.
        s
    }
}

enum WriterMode {
    Buffered(LogBuffer),
    File(Arc<Mutex<std::fs::File>>),
    Immediate,
}

/// A custom writer that can either buffer logs for testing, write to file, or write directly
pub struct TestWriter {
    mode: WriterMode,
}

impl TestWriter {
    pub fn new(buffer: LogBuffer) -> Self {
        Self {
            mode: WriterMode::Buffered(buffer),
        }
    }

    /// Create a writer that outputs immediately (for main application)
    pub fn new_immediate() -> Self {
        Self {
            mode: WriterMode::Immediate,
        }
    }

    /// Create a writer that outputs to a file
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
                let mut buffer = buffer.0.lock().unwrap();
                buffer.extend_from_slice(buf);
            }
            WriterMode::File(file) => {
                let mut file = file.lock().unwrap();
                file.write_all(buf)?;
                file.flush()?;
            }
            WriterMode::Immediate => {
                use std::fs::OpenOptions;
                match OpenOptions::new().write(true).open("/dev/tty") {
                    Ok(mut tty) => {
                        tty.write_all(buf)?;
                        tty.flush()?;
                    }
                    Err(_) => {
                        io::stdout().write_all(buf)?;
                        io::stdout().flush()?;
                    }
                }
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if let WriterMode::File(file) = &self.mode {
            file.lock().unwrap().flush()?;
        }
        Ok(())
    }
}

impl Write for TestWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match &self.mode {
            WriterMode::Buffered(buffer) => {
                let mut buffer = buffer.0.lock().unwrap();
                buffer.extend_from_slice(buf);
            }
            WriterMode::File(file) => {
                let mut file = file.lock().unwrap();
                file.write_all(buf)?;
                file.flush()?;
            }
            WriterMode::Immediate => {
                use std::fs::OpenOptions;
                match OpenOptions::new().write(true).open("/dev/tty") {
                    Ok(mut tty) => {
                        tty.write_all(buf)?;
                        tty.flush()?;
                    }
                    Err(_) => {
                        io::stdout().write_all(buf)?;
                        io::stdout().flush()?;
                    }
                }
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        match &self.mode {
            WriterMode::File(file) => {
                file.lock().unwrap().flush()?;
            }
            WriterMode::Immediate => {
                use std::fs::OpenOptions;
                match OpenOptions::new().write(true).open("/dev/tty") {
                    Ok(mut tty) => {
                        tty.flush()?;
                    }
                    Err(_) => {
                        io::stdout().flush()?;
                    }
                }
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

/// File writer for logging to a file
pub struct FileWriter {
    file: Arc<Mutex<std::fs::File>>,
}

impl FileWriter {
    pub fn new(file: std::fs::File) -> Self {
        Self {
            file: Arc::new(Mutex::new(file)),
        }
    }
}

impl<'a> MakeWriter<'a> for FileWriter {
    type Writer = TestWriter;

    fn make_writer(&self) -> Self::Writer {
        TestWriter::new_file(self.file.clone())
    }

    fn make_writer_for(&'a self, meta: &tracing::Metadata<'_>) -> Self::Writer {
        let _ = meta;
        self.make_writer()
    }
}

// Default Logger implementation for production use
pub struct DefaultLogger {
    verbose: bool,
    format: Format,
    log_file: Option<String>,
}

impl DefaultLogger {
    pub fn new(verbose: bool, format: Format) -> Self {
        Self {
            verbose,
            format,
            log_file: None,
        }
    }

    pub fn with_log_file(mut self, log_file: Option<String>) -> Self {
        self.log_file = log_file;
        self
    }
}

impl Logger for DefaultLogger {
    fn init(&self) -> Result<()> {
        let level = if self.verbose {
            Level::DEBUG
        } else {
            Level::INFO
        };

        if let Some(ref log_file_path) = self.log_file {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(log_file_path)
                .map_err(|e| ScannerError::Custom(format!("Failed to open log file: {}", e)))?;
            let file_writer = FileWriter::new(file);

            match self.format {
                Format::Json => {
                    let subscriber = FmtSubscriber::builder()
                        .json()
                        .with_max_level(level)
                        .with_writer(file_writer)
                        .finish();
                    tracing::subscriber::set_global_default(subscriber)
                        .expect("setting default subscriber failed");
                }
                Format::Text => {
                    let subscriber = FmtSubscriber::builder()
                        .with_max_level(level)
                        .with_writer(file_writer)
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
                        .with_writer(file_writer)
                        .with_target(false)
                        .finish();
                    tracing::subscriber::set_global_default(subscriber)
                        .expect("setting default subscriber failed");
                }
            }
        } else {
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
        }

        Ok(())
    }
}

pub fn init(logger: &dyn Logger, verbose: bool) -> Result<()> {
    let _ = verbose; // No longer used, keeping for compatibility
    let _ = logger.init();
    soapysdr::configure_logging();
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
