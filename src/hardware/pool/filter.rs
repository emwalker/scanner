//! Pool filtering and tuning mode controls

use std::collections::HashSet;

use tracing::debug;

use crate::{hardware, hardware::pool::types::TunerId};

/// Controls which tuners are available for allocation
///
/// Used for gradual rollout of multi-tuner support:
/// - Phase 1: Constrain by backend, driver, or tuning mode
/// - Phase 2+: Gradually relax constraints
/// - Final: allow_all() - full multi-tuner support
#[derive(Debug)]
pub struct PoolFilter {
    backend: Option<hardware::types::Backend>,
    driver: Option<String>,
    mode: Option<TuningMode>,
    device_mode: Option<String>,
    channel: Option<usize>,
    specific_tuners: Option<HashSet<TunerId>>,
}

/// Tuning mode constraint for filtering
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TuningMode {
    /// Single-tuner mode (ST) - only one tuner can be allocated at a time
    SingleTuner,
    /// Multi-tuner mode (MT) - multiple tuners can be allocated simultaneously
    MultiTuner,
}

impl PoolFilter {
    /// Create a new filter with optional constraints
    ///
    /// # Examples
    /// ```
    /// use scanner::hardware::{
    ///     pool::{PoolFilter, TuningMode},
    ///     types::Backend,
    /// };
    ///
    /// // Allow only sdrplay devices in single-tuner mode
    /// let filter = PoolFilter::new()
    ///     .with_driver("sdrplay")
    ///     .with_mode(TuningMode::SingleTuner);
    ///
    /// // Allow only soapy backend
    /// let filter = PoolFilter::new().with_backend(Backend::Soapy);
    /// ```
    pub fn new() -> Self {
        Self {
            backend: None,
            driver: None,
            mode: None,
            device_mode: None,
            channel: None,
            specific_tuners: None,
        }
    }

    /// Constrain to specific backend
    pub fn with_backend(mut self, backend: hardware::types::Backend) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Constrain to specific driver (e.g., "sdrplay", "rtlsdr")
    pub fn with_driver(mut self, driver: impl Into<String>) -> Self {
        self.driver = Some(driver.into());
        self
    }

    /// Constrain to specific tuning mode
    pub fn with_mode(mut self, mode: TuningMode) -> Self {
        self.mode = Some(mode);
        self
    }

    /// Constrain to specific device mode (e.g., "ST", "DT", "MA" for RSPduo)
    pub fn with_device_mode(mut self, device_mode: impl Into<String>) -> Self {
        self.device_mode = Some(device_mode.into());
        self
    }

    /// Constrain to specific channel index (e.g., 0 for first tuner, 1 for second tuner)
    pub fn with_channel(mut self, channel: usize) -> Self {
        self.channel = Some(channel);
        self
    }

    /// Constrain to specific tuner IDs (most restrictive)
    pub fn with_tuners(mut self, tuners: Vec<TunerId>) -> Self {
        self.specific_tuners = Some(tuners.into_iter().collect());
        self
    }

    /// Allow all tuners (full multi-tuner mode)
    pub fn allow_all() -> Self {
        Self::new()
    }

    /// Check if a tuner is allowed for allocation
    pub(crate) fn is_allowed(
        &self,
        tuner_id: &TunerId,
        backend: &hardware::types::Backend,
        allocated_count: usize,
        mode: &str,
    ) -> bool {
        // Check specific tuners first (most restrictive)
        if let Some(allowed) = &self.specific_tuners {
            if !allowed.contains(tuner_id) {
                debug!(tuner_id = ?tuner_id, allowed = ?allowed, "Filter rejected: tuner not in allowed set");
                return false;
            }
        } else {
            // Check backend (only if specific tuners not set)
            if let Some(allowed_backend) = &self.backend
                && backend != allowed_backend
            {
                debug!(
                    tuner_id = ?tuner_id,
                    backend = ?backend,
                    allowed_backend = ?allowed_backend,
                    "Filter rejected: backend mismatch"
                );
                return false;
            }

            // Check driver (case-insensitive, only if specific tuners not set)
            if let Some(allowed_driver) = &self.driver {
                match &tuner_id.device_id {
                    hardware::DeviceId::Driver { driver, .. } => {
                        if !driver.eq_ignore_ascii_case(allowed_driver) {
                            debug!(
                                tuner_id = ?tuner_id,
                                driver = driver,
                                allowed_driver = allowed_driver,
                                "Filter rejected: driver mismatch"
                            );
                            return false;
                        }
                    }
                    hardware::DeviceId::Usb { .. } => {
                        debug!(tuner_id = ?tuner_id, "Filter rejected: USB devices not allowed when driver filter is set");
                        return false;
                    }
                }
            }

            // Check channel index (only if specific tuners not set)
            if let Some(allowed_channel) = self.channel
                && tuner_id.channel_index != allowed_channel
            {
                debug!(
                    tuner_id = ?tuner_id,
                    channel = tuner_id.channel_index,
                    allowed_channel = allowed_channel,
                    "Filter rejected: channel mismatch"
                );
                return false;
            }

            // Check device mode (only if specific tuners not set)
            if let Some(allowed_mode) = &self.device_mode
                && mode != allowed_mode
            {
                debug!(
                    tuner_id = ?tuner_id,
                    mode = mode,
                    allowed_mode = allowed_mode,
                    "Filter rejected: device mode mismatch"
                );
                return false;
            }
        }

        // Check tuning mode
        if let Some(tuning_mode) = &self.mode {
            match tuning_mode {
                TuningMode::SingleTuner => {
                    if allocated_count > 0 {
                        debug!(
                            tuner_id = ?tuner_id,
                            allocated_count = allocated_count,
                            "Filter rejected: SingleTuner mode and {} tuner(s) already allocated",
                            allocated_count
                        );
                        return false;
                    }
                }
                TuningMode::MultiTuner => {}
            }
        }

        debug!(tuner_id = ?tuner_id, "Filter allowed tuner");
        true
    }
}

impl Default for PoolFilter {
    fn default() -> Self {
        Self::new()
    }
}
