//! Subprocess handle for managing device worker lifecycle

use crate::core::types::{Result, ScannerError};
use crate::hardware::{DeviceId, streaming::ActualConfig};
use crate::ipc::{
    ControlChannel, ControlMessage, DataReceiver, IQPacket, UnixControlChannel, UnixDataReceiver,
};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::debug;

/// Handle for managing a device worker subprocess
///
/// One subprocess per device (not per tuner). Multi-channel devices like RSPduo
/// have a single subprocess managing all channels.
pub struct SubprocessHandle {
    device_id: DeviceId,
    process: Mutex<Child>,
    control_channel: Mutex<UnixControlChannel>,
    pub data_receiver: Arc<Mutex<UnixDataReceiver>>,
    shutdown_flag: Arc<AtomicBool>,
    socket_paths: (PathBuf, PathBuf),
}

impl SubprocessHandle {
    /// Spawn a new device worker subprocess
    pub fn spawn(
        device_id: DeviceId,
        shutdown_flag: Arc<AtomicBool>,
        parent_log_file: Option<&str>,
    ) -> Result<Self> {
        if shutdown_flag.load(Ordering::SeqCst) {
            return Err(ScannerError::PoolShutdown);
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);

        let device_id_str = serde_json::to_string(&device_id)?;

        let control_socket_path = PathBuf::from(format!(
            "/tmp/scanner-{}-{}-ctl.sock",
            device_id.to_string().replace([':', '/', ' '], "_"),
            timestamp
        ));
        let data_socket_path = PathBuf::from(format!(
            "/tmp/scanner-{}-{}-dat.sock",
            device_id.to_string().replace([':', '/', ' '], "_"),
            timestamp
        ));

        debug!(
            device_id = ?device_id,
            control_socket = %control_socket_path.display(),
            data_socket = %data_socket_path.display(),
            "Spawning device worker subprocess"
        );

        use crate::cli::worker_logging::{WorkerContext, WorkerType, generate_worker_log_path};

        let worker_log_path = generate_worker_log_path(
            parent_log_file,
            WorkerType::Device,
            &WorkerContext {
                device_id: Some(device_id.to_string().replace([':', '/', ' '], "_")),
                timestamp: Some(timestamp),
                backend: None,
            },
        );

        let binary_path = std::env::current_exe()?;
        let binary_path_str = binary_path.to_string_lossy();

        let binary_to_use = if binary_path_str.contains("/deps/") {
            let target_dir = binary_path
                .parent()
                .and_then(|p| p.parent())
                .ok_or_else(|| {
                    ScannerError::Custom("Failed to find target directory".to_string())
                })?;
            target_dir.join("scanner")
        } else {
            binary_path
        };

        let mut cmd = Command::new(binary_to_use);
        cmd.arg("worker")
            .arg("device")
            .arg("--device-id-str")
            .arg(&device_id_str)
            .arg("--control-socket-path")
            .arg(&control_socket_path)
            .arg("--data-socket-path")
            .arg(&data_socket_path);

        if let Some(log_path) = worker_log_path {
            cmd.arg("--log-file").arg(&log_path);
        }

        cmd.stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        let mut child = cmd.spawn()?;

        let start = Instant::now();
        let timeout = Duration::from_secs(5);

        while !control_socket_path.exists() || !data_socket_path.exists() {
            // Check if process has already exited (failed to start)
            if let Ok(Some(status)) = child.try_wait() {
                return Err(ScannerError::Custom(format!(
                    "Device worker process exited immediately with status: {}",
                    status
                )));
            }

            if start.elapsed() > timeout {
                return Err(ScannerError::Custom(
                    "Device worker socket creation timeout".to_string(),
                ));
            }
            thread::sleep(Duration::from_millis(10));
        }

        let control_stream = UnixStream::connect(&control_socket_path)?;
        let data_stream = UnixStream::connect(&data_socket_path)?;

        control_stream.set_read_timeout(Some(Duration::from_secs(10)))?;
        data_stream.set_read_timeout(Some(Duration::from_millis(100)))?;

        let mut control_channel = UnixControlChannel::new(control_stream);
        let data_receiver = UnixDataReceiver::new(data_stream);

        match control_channel.recv()? {
            ControlMessage::Ready {
                device_id: _,
                channels: _,
            } => {
                debug!(device_id = ?device_id, "Device worker ready");
            }
            msg => {
                return Err(ScannerError::Custom(format!(
                    "Expected Ready message, got {:?}",
                    msg
                )));
            }
        }

        let handle = Self {
            device_id,
            process: Mutex::new(child),
            control_channel: Mutex::new(control_channel),
            data_receiver: Arc::new(Mutex::new(data_receiver)),
            shutdown_flag,
            socket_paths: (control_socket_path, data_socket_path),
        };

        #[cfg(debug_assertions)]
        Self::validate_subprocess_state(&handle)?;

        Ok(handle)
    }

    #[cfg(debug_assertions)]
    fn validate_subprocess_state(handle: &SubprocessHandle) -> Result<()> {
        debug_assert!(
            handle.socket_paths.0.exists(),
            "Control socket missing for device {}",
            handle.device_id
        );
        debug_assert!(
            handle.socket_paths.1.exists(),
            "Data socket missing for device {}",
            handle.device_id
        );

        if let Ok(mut proc) = handle.process.try_lock() {
            match proc.try_wait() {
                Ok(None) => {}
                Ok(Some(status)) => {
                    return Err(ScannerError::Custom(format!(
                        "Subprocess for device {} died unexpectedly: {:?}",
                        handle.device_id, status
                    )));
                }
                Err(e) => {
                    return Err(ScannerError::Custom(format!(
                        "Cannot check subprocess for device {}: {:?}",
                        handle.device_id, e
                    )));
                }
            }
        }

        Ok(())
    }

    /// Configure and start streaming on a channel
    pub fn configure_and_start(
        &self,
        channel: usize,
        freq_hz: f64,
        gain_db: f64,
        sample_rate: f64,
    ) -> Result<ActualConfig> {
        #[cfg(debug_assertions)]
        Self::validate_subprocess_state(self)?;
        if self.shutdown_flag.load(Ordering::SeqCst) {
            return Err(ScannerError::PoolShutdown);
        }

        let mut control = self
            .control_channel
            .lock()
            .map_err(|e| ScannerError::Custom(format!("Control channel lock failed: {}", e)))?;

        control.send(&ControlMessage::ConfigureAndStart {
            channel,
            freq_hz,
            gain_db,
            sample_rate,
        })?;

        match control.recv()? {
            ControlMessage::StreamStarted {
                channel: _,
                actual_freq,
                actual_gain,
                actual_sample_rate,
            } => Ok(ActualConfig {
                freq_hz: actual_freq,
                sample_rate: actual_sample_rate,
                gain_db: actual_gain,
            }),
            ControlMessage::Error { message, .. } => Err(ScannerError::Custom(format!(
                "Stream start failed: {}",
                message
            ))),
            msg => Err(ScannerError::Custom(format!(
                "Unexpected response to ConfigureAndStart: {:?}",
                msg
            ))),
        }
    }

    /// Stop streaming on a channel
    ///
    /// During normal operation, waits for StreamStopped acknowledgment to ensure
    /// the worker is ready for new commands. During shutdown, uses fire-and-forget.
    pub fn stop_stream(&self, channel: usize) -> Result<()> {
        #[cfg(debug_assertions)]
        Self::validate_subprocess_state(self)?;

        let shutdown_mode = self.shutdown_flag.load(Ordering::SeqCst);

        if shutdown_mode {
            debug!(channel, "Shutdown mode: fire-and-forget StopStream");
            if let Ok(mut control) = self.control_channel.try_lock() {
                let _ = control.send(&ControlMessage::StopStream { channel });
            }
            return Ok(());
        }

        let mut control = self
            .control_channel
            .lock()
            .map_err(|e| ScannerError::Custom(format!("Control channel lock failed: {}", e)))?;

        debug!(channel, "Sending StopStream message (waiting for ack)");
        control.send(&ControlMessage::StopStream { channel })?;

        match control.recv()? {
            ControlMessage::StreamStopped {
                channel: resp_channel,
            } if resp_channel == channel => {
                debug!(channel, "Stream stopped successfully");
                Ok(())
            }
            ControlMessage::Error { message, .. } => Err(ScannerError::Custom(format!(
                "Failed to stop stream: {}",
                message
            ))),
            msg => Err(ScannerError::Custom(format!(
                "Unexpected response to StopStream: {:?}",
                msg
            ))),
        }
    }

    /// Read I/Q samples from the data channel (non-blocking)
    pub fn read_samples(&self, _timeout_ms: u64) -> Result<Option<IQPacket>> {
        let mut data = self
            .data_receiver
            .lock()
            .map_err(|e| ScannerError::Custom(format!("Data receiver lock failed: {}", e)))?;

        match data.recv() {
            Ok(packet) => Ok(Some(packet)),
            Err(e) => {
                if e.to_string().contains("would block") {
                    Ok(None)
                } else {
                    Err(e)
                }
            }
        }
    }

    /// Graceful shutdown with timeout escalation
    pub fn shutdown(&mut self) -> Result<()> {
        debug!(device_id = ?self.device_id, "Shutting down device subprocess");

        self.shutdown_flag.store(true, Ordering::SeqCst);

        match self.control_channel.lock() {
            Ok(mut control) => {
                if let Err(e) = control.send(&ControlMessage::Shutdown) {
                    debug!(device_id = ?self.device_id, error = ?e, "Failed to send Shutdown message");
                } else {
                    debug!(device_id = ?self.device_id, "Shutdown message sent, waiting for process exit");
                }
            }
            Err(e) => {
                debug!(device_id = ?self.device_id, error = ?e, "Failed to lock control channel for shutdown");
            }
        }

        let timeout = Duration::from_secs(2);
        let start = Instant::now();

        let mut process = self
            .process
            .lock()
            .map_err(|e| ScannerError::Custom(format!("Process lock failed: {}", e)))?;

        loop {
            match process.try_wait()? {
                Some(status) => {
                    debug!(device_id = ?self.device_id, ?status, "Subprocess exited gracefully");
                    self.cleanup_sockets();
                    return Ok(());
                }
                None if start.elapsed() < timeout => {
                    thread::sleep(Duration::from_millis(50));
                }
                None => break,
            }
        }

        debug!(device_id = ?self.device_id, "Graceful shutdown timeout, sending SIGTERM");

        #[cfg(unix)]
        {
            let pid = process.id();
            unsafe {
                libc::kill(pid as i32, libc::SIGTERM);
            }

            let sigterm_timeout = Duration::from_secs(1);
            let sigterm_start = Instant::now();

            loop {
                match process.try_wait()? {
                    Some(status) => {
                        debug!(device_id = ?self.device_id, ?status, "Subprocess exited after SIGTERM");
                        self.cleanup_sockets();
                        return Ok(());
                    }
                    None if sigterm_start.elapsed() < sigterm_timeout => {
                        thread::sleep(Duration::from_millis(50));
                    }
                    None => break,
                }
            }
        }

        debug!(device_id = ?self.device_id, "SIGTERM timeout, force killing with SIGKILL");
        process.kill()?;

        let status = process.wait()?;
        debug!(device_id = ?self.device_id, ?status, "Subprocess killed with SIGKILL");

        self.cleanup_sockets();
        Ok(())
    }

    fn cleanup_sockets(&self) {
        let _ = std::fs::remove_file(&self.socket_paths.0);
        let _ = std::fs::remove_file(&self.socket_paths.1);
    }
}

impl Drop for SubprocessHandle {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}
