//! Device trait for individual SDR devices

use super::{Capabilities, DeviceId};
use crate::types::Result;
use rustradio::Complex;
use std::any::Any;

/// Individual device abstraction (backend-agnostic)
///
/// This trait represents a single SDR device and provides methods to configure
/// it and integrate it with rustradio's graph-based processing system.
///
/// # Design Notes
///
/// The key method is `add_source_to_graph()` which integrates the device with
/// rustradio's graph system. This approach:
/// - Maintains type safety (returns concrete `ReadStream<Complex>`)
/// - Avoids `Box<dyn Any>` downcasting
/// - Integrates naturally with rustradio's block-based architecture
/// - Places dynamic dispatch at setup time, not in sample processing
///
/// # Examples
///
/// ```no_run
/// use rustradio::graph::Graph;
/// use scanner::sdr::{Backend, Soapy};
///
/// let backend = Soapy;
/// let devices = backend.enumerate_devices()?;
/// let device = backend.open_device(&devices[0].id)?;
///
/// let mut graph = Graph::new();
/// let stream = device.add_source_to_graph(
///     &mut graph,
///     88.9e6,  // 88.9 MHz
///     2.4e6,   // 2.4 MHz sample rate
///     20.0,    // 20 dB gain
/// )?;
/// # Ok::<(), scanner::types::ScannerError>(())
/// ```
pub trait DeviceTrait: Send {
    /// Stable device identifier
    fn id(&self) -> &DeviceId;

    /// Device capabilities (frequency range, sample rates, etc.)
    fn capabilities(&self) -> &Capabilities;

    /// Add source block to rustradio graph and return stream handle
    ///
    /// This is called each time a new graph is created. The device should:
    /// 1. Create a source block configured for the given parameters
    /// 2. Add the block to the graph
    /// 3. Return the output stream handle
    ///
    /// Unlike returning `Box<dyn Any>`, this maintains type safety by always
    /// returning a concrete `ReadStream<Complex>` that works with rustradio.
    ///
    /// # Arguments
    ///
    /// * `graph` - The rustradio graph to add the source block to
    /// * `freq` - Center frequency in Hz
    /// * `samp_rate` - Sample rate in Hz
    /// * `gain_db` - Gain in dB
    fn add_source_to_graph(
        &self,
        graph: &mut rustradio::graph::Graph,
        freq: f64,
        samp_rate: f64,
        gain_db: f64,
    ) -> Result<rustradio::stream::ReadStream<Complex>>;

    /// Tune to frequency (for devices that support runtime retuning)
    ///
    /// Not all devices support this - some require rebuilding the graph.
    /// Returns an error if the device doesn't support runtime retuning.
    fn tune(&mut self, freq: f64) -> Result<()>;

    /// Set gain (for devices that support runtime gain adjustment)
    ///
    /// Returns an error if the device doesn't support runtime gain adjustment.
    fn set_gain(&mut self, gain: f64) -> Result<()>;

    /// Consume device and return backend-specific representation
    ///
    /// Provides escape hatch for advanced users who need direct backend access.
    /// Following embedded-hal best practices.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use scanner::sdr::{Backend, Soapy};
    ///
    /// let backend = Soapy;
    /// let devices = backend.enumerate_devices()?;
    /// let device = backend.open_device(&devices[0].id)?;
    ///
    /// // Get raw device args for advanced configuration
    /// let raw = device.into_inner();
    /// let device_args = raw.downcast::<String>().unwrap();
    /// # Ok::<(), scanner::types::ScannerError>(())
    /// ```
    fn into_inner(self: Box<Self>) -> Box<dyn Any>;
}
