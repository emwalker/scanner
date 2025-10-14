//! Seify backend stub (native Rust SDR abstraction)
//!
//! Seify is actively developed and maturing faster than expected:
//! - Version 0.17.0 published ~1 month ago (actively maintained)
//! - Native RTL-SDR driver already implemented (seify-rtlsdr)
//! - Uses DeviceTrait + RxStreamer/TxStreamer pattern
//! - Designed for zero-installation (no system libraries via rusb)
//!
//! This stub will be implemented in Phase 2 (2025-2026) when Seify v1.0 is released.

use super::{Backend, DeviceError, DeviceErrorKind, DeviceInfo, DeviceTrait};
use crate::core::types::Result;

/// Seify backend (native Rust)
///
/// NOTE: Seify is actively developed - consider implementing this in Phase 2
pub struct Seify;

impl Backend for Seify {
    fn enumerate_devices(&self) -> Result<Vec<DeviceInfo>> {
        // When Seify is ready (v1.0):
        // let devices = seify::enumerate()?;
        Ok(vec![]) // Stub for now
    }

    fn open_tuner(
        &self,
        _tuner_id: &crate::hardware::pool::TunerId,
    ) -> Result<Box<dyn DeviceTrait>> {
        // TODO: Implement when Seify reaches v1.0
        // Will require SeifyBridgeBlock to connect to rustradio
        //
        // Implementation will look like:
        // let device = seify::Device::new()?;
        // Ok(Box::new(SeifyDevice::new(device)?))
        Err(DeviceError::new(
            DeviceErrorKind::Unsupported,
            "Seify",
            "Seify backend planned for Phase 2 (when v1.0 is released)",
        )
        .into())
    }

    fn open_streaming_tuner(
        &self,
        _tuner_id: &crate::hardware::pool::TunerId,
    ) -> Result<Box<dyn super::streaming::StreamingDevice>> {
        Err(DeviceError::new(
            DeviceErrorKind::Unsupported,
            "Seify",
            "Seify backend planned for Phase 2 (when v1.0 is released)",
        )
        .into())
    }

    fn name(&self) -> &str {
        "Seify (native Rust)"
    }
}

// Future implementation notes:
//
// pub struct SeifyDevice {
//     inner: seify::Device,
//     capabilities: Capabilities,
// }
//
// impl Device for SeifyDevice {
//     fn add_source_to_graph(...) -> Result<ReadStream<Complex>> {
//         // Bridge Seify's RxStreamer to rustradio block
//         let rx_streamer = self.inner.rx_streamer(&[0])?;
//         let bridge_block = SeifyBridgeBlock::new(rx_streamer, freq, samp_rate, gain_db);
//         let stream = bridge_block.output_stream();
//         graph.add(Box::new(bridge_block));
//         Ok(stream)
//     }
// }
//
// struct SeifyBridgeBlock {
//     streamer: seify::RxStreamer,
//     // ... configuration
// }
//
// impl rustradio::Block for SeifyBridgeBlock {
//     fn work(&mut self) -> rustradio::Result<()> {
//         // Read from Seify streamer, push to rustradio stream
//         let mut buffer = vec![Complex::default(); 1024];
//         let n = self.streamer.read(&[&mut buffer], 100_000)?;
//         // ... push to output stream
//         Ok(())
//     }
// }
