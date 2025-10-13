use crate::core::errors::ScannerError;
use crate::core::types::Result;
use crate::hardware::{Backend, Mock, Soapy, StreamingDevice};
use crate::ipc::{
    ControlChannel, ControlMessage, DataSender, IQPacket, UnixControlChannel, UnixDataSender,
};
use rustradio::Complex;
use std::collections::HashMap;
use std::error::Error;
use std::os::unix::net::UnixListener;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::SystemTime;
use tracing::debug;

pub fn handle_enumerate_command(
    backend_name: &str,
    socket_path: &str,
    log_file: Option<&str>,
) -> Result<()> {
    if let Some(log_path) = log_file {
        use std::fs::OpenOptions;
        use std::io::Write;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)?;

        writeln!(file, "Enumeration worker initializing")?;
        file.flush()?;

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)?;

        let file_writer = crate::logging::FileWriter::new(file);

        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_writer(file_writer)
            .with_ansi(false)
            .try_init();
    }

    debug!(
        backend = backend_name,
        pid = %std::process::id(),
        socket_path = socket_path,
        "Enumeration worker starting"
    );

    let listener = UnixListener::bind(socket_path)?;
    debug!("Waiting for parent connection");

    let (stream, _) = listener.accept()?;
    let mut channel = UnixControlChannel::with_cleanup(stream, socket_path.into());

    debug!("Parent connected, enumerating devices");

    use crate::hardware::types::Backend as BackendEnum;

    let backend_enum: BackendEnum = backend_name.parse().unwrap();
    let backend: Box<dyn Backend> = match backend_enum {
        BackendEnum::Soapy => Box::new(Soapy),
        BackendEnum::Mock => Box::new(Mock),
        BackendEnum::Usb => {
            return Err(ScannerError::Custom(
                "USB backend not supported for enumeration".to_string(),
            ));
        }
        BackendEnum::Unknown(name) => {
            return Err(ScannerError::Custom(format!("Unknown backend: {}", name)));
        }
    };
    let result = backend.enumerate_devices();

    match result {
        Ok(devices) => {
            debug!(device_count = devices.len(), "Enumeration successful");
            channel.send(&ControlMessage::DeviceList { devices })?;
        }
        Err(e) => {
            debug!(error = %e, "Enumeration failed");
            channel.send(&ControlMessage::Error {
                channel: None,
                message: e.to_string(),
            })?;
        }
    }

    debug!("Enumeration worker complete");
    Ok(())
}

pub fn handle_device_command(
    device_id_str: &str,
    control_socket_path: &str,
    data_socket_path: &str,
    log_file: Option<&str>,
) -> Result<()> {
    use std::fs::OpenOptions;

    if let Some(log_path) = log_file {
        use std::io::Write;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)?;

        writeln!(file, "Device worker initializing")?;
        file.flush()?;

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)?;

        let file_writer = crate::logging::FileWriter::new(file);

        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_writer(file_writer)
            .with_ansi(false)
            .try_init();
    }

    soapysdr::configure_logging();

    use crate::hardware::types::Backend as BackendEnum;

    let device_id: crate::hardware::DeviceId = serde_json::from_str(device_id_str)?;
    debug!(device_id = ?device_id, "Device worker starting");

    if device_id.backend() == BackendEnum::Soapy {
        debug!("Resetting SoapySDR state");
        crate::hardware::soapy::reset_soapysdr_state();
    }

    // Open first tuner (channel_index 0) for streaming
    let tuner_id = crate::hardware::pool::TunerId::new(device_id.clone(), 0);

    let backend_enum = device_id.backend();
    let backend = match backend_enum {
        BackendEnum::Soapy => Box::new(Soapy) as Box<dyn Backend>,
        BackendEnum::Mock => Box::new(Mock),
        BackendEnum::Usb => {
            return Err(ScannerError::Custom(
                "USB devices not supported in worker".to_string(),
            ));
        }
        BackendEnum::Unknown(name) => {
            return Err(ScannerError::Custom(format!("Unknown backend: {}", name)));
        }
    };
    let ctl_listener = UnixListener::bind(control_socket_path)?;
    let dat_listener = UnixListener::bind(data_socket_path)?;

    debug!("Waiting for parent connections");
    let (ctl_stream, _) = ctl_listener.accept()?;
    let (dat_stream, _) = dat_listener.accept()?;

    // Set write timeout so data thread can respond to commands even when parent stops reading
    // 100ms timeout allows checking for commands ~10 times per second
    dat_stream.set_write_timeout(Some(std::time::Duration::from_millis(100)))?;

    let mut ctl_channel = UnixControlChannel::with_cleanup(ctl_stream, control_socket_path.into());
    let dat_sender = UnixDataSender::with_cleanup(dat_stream, data_socket_path.into());

    ctl_channel.send(&ControlMessage::Ready {
        device_id: device_id_str.to_string(),
        channels: 1,
    })?;

    debug!("Entering main loop");
    main_loop(ctl_channel, dat_sender, backend, tuner_id)?;

    debug!("Device worker shutting down");
    Ok(())
}

struct StreamState {
    sequence_number: u64,
}

/// Internal command from control thread to data thread
enum InternalCommand {
    ConfigureAndStart {
        channel: usize,
        freq_hz: f64,
        gain_db: f64,
        sample_rate: f64,
    },
    StopStream {
        channel: usize,
    },
    Shutdown,
}

fn main_loop(
    mut ctl_channel: impl ControlChannel + Send + 'static,
    mut dat_sender: impl DataSender + Send + 'static,
    backend: Box<dyn Backend>,
    tuner_id: crate::hardware::pool::TunerId,
) -> Result<()> {
    // Create channels for inter-thread communication
    let (cmd_tx, cmd_rx): (Sender<InternalCommand>, Receiver<InternalCommand>) = mpsc::channel();
    let (resp_tx, resp_rx): (Sender<ControlMessage>, Receiver<ControlMessage>) = mpsc::channel();

    // Spawn control thread (handles control socket)
    let control_thread = thread::spawn(move || {
        debug!("Control thread starting");
        loop {
            match ctl_channel.recv() {
                Ok(msg) => {
                    debug!(message = ?msg, "Control thread received external message");
                    match msg {
                        ControlMessage::ConfigureAndStart {
                            channel,
                            freq_hz,
                            gain_db,
                            sample_rate,
                        } => {
                            if cmd_tx
                                .send(InternalCommand::ConfigureAndStart {
                                    channel,
                                    freq_hz,
                                    gain_db,
                                    sample_rate,
                                })
                                .is_err()
                            {
                                debug!("Data thread disconnected");
                                break;
                            }

                            // Wait for response from data thread
                            match resp_rx.recv() {
                                Ok(response) => {
                                    if let Err(e) = ctl_channel.send(&response) {
                                        debug!(error = ?e, "Failed to send response to parent");
                                        break;
                                    }
                                }
                                Err(_) => {
                                    debug!("Data thread disconnected while waiting for response");
                                    break;
                                }
                            }
                        }

                        ControlMessage::StopStream { channel } => {
                            debug!(
                                channel,
                                "Control thread received StopStream, forwarding to data thread"
                            );
                            if cmd_tx
                                .send(InternalCommand::StopStream { channel })
                                .is_err()
                            {
                                debug!("Data thread disconnected");
                                break;
                            }

                            // Wait for response from data thread
                            match resp_rx.recv() {
                                Ok(response) => {
                                    if let Err(e) = ctl_channel.send(&response) {
                                        debug!(error = ?e, "Failed to send response to parent");
                                        break;
                                    }
                                }
                                Err(_) => {
                                    debug!(
                                        "Data thread disconnected while waiting for StopStream response"
                                    );
                                    break;
                                }
                            }
                        }

                        ControlMessage::Shutdown => {
                            debug!("Control thread received Shutdown, signaling data thread");
                            let _ = cmd_tx.send(InternalCommand::Shutdown);
                            break;
                        }

                        _ => {
                            debug!("Control thread received unexpected message");
                        }
                    }
                }
                Err(e) => {
                    debug!(error = ?e, "Control thread recv error");
                    break;
                }
            }
        }
        debug!("Control thread exiting");
    });

    // Data thread (handles device streaming)
    #[allow(clippy::cognitive_complexity)]
    let data_thread = thread::spawn(move || {
        debug!("Data thread starting");
        let mut device: Option<Box<dyn StreamingDevice>> = None;
        let mut active_streams: HashMap<usize, StreamState> = HashMap::new();
        let mut sample_buffer = vec![Complex::new(0.0, 0.0); 2048];
        let mut running = true;
        let mut pending_command: Option<InternalCommand> = None;

        while running {
            // Check for commands (non-blocking)
            // First check if we have a pending command from last iteration
            let cmd_result = if let Some(cmd) = pending_command.take() {
                Ok(cmd)
            } else {
                cmd_rx.try_recv()
            };

            match cmd_result {
                Ok(InternalCommand::ConfigureAndStart {
                    channel,
                    freq_hz,
                    gain_db,
                    sample_rate,
                }) => {
                    debug!(
                        channel,
                        freq_hz, gain_db, sample_rate, "Data thread: ConfigureAndStart"
                    );

                    let result = (|| {
                        debug!("Recreating device for new stream");
                        let mut new_device = backend.open_streaming_tuner(&tuner_id)?;
                        debug!("Device opened successfully");

                        debug!(
                            channel,
                            freq_hz, sample_rate, gain_db, "Calling configure_rx"
                        );
                        let actual =
                            new_device.configure_rx(channel, freq_hz, sample_rate, gain_db)?;
                        debug!(channel, "configure_rx completed, calling start_stream");
                        new_device.start_stream(channel)?;
                        debug!(channel, "start_stream completed");

                        device = Some(new_device);
                        Ok::<_, crate::core::types::ScannerError>((actual, channel))
                    })();

                    let response = match result {
                        Ok((actual, channel)) => {
                            active_streams.insert(channel, StreamState { sequence_number: 0 });
                            ControlMessage::StreamStarted {
                                channel,
                                actual_freq: actual.freq_hz,
                                actual_gain: actual.gain_db,
                                actual_sample_rate: actual.sample_rate,
                            }
                        }
                        Err(e) => {
                            debug!(channel, error = ?e, "Failed to configure or start stream");
                            ControlMessage::Error {
                                channel: Some(channel),
                                message: format!("Failed to configure/start stream: {}", e),
                            }
                        }
                    };

                    if resp_tx.send(response).is_err() {
                        debug!("Control thread disconnected");
                        break;
                    }
                }

                Ok(InternalCommand::StopStream { channel }) => {
                    debug!(channel, "Data thread: StopStream");

                    let response = if let Some(mut dev) = device.take() {
                        match dev.stop_stream(channel) {
                            Ok(_) => {
                                active_streams.remove(&channel);
                                debug!(channel, "Stream stopped, dropping device");
                                drop(dev);
                                ControlMessage::StreamStopped { channel }
                            }
                            Err(e) => {
                                debug!(channel, error = ?e, "Failed to stop stream");
                                drop(dev);
                                ControlMessage::Error {
                                    channel: Some(channel),
                                    message: format!("Failed to stop stream: {}", e),
                                }
                            }
                        }
                    } else {
                        debug!(channel, "No device to stop");
                        ControlMessage::StreamStopped { channel }
                    };

                    if resp_tx.send(response).is_err() {
                        debug!("Control thread disconnected");
                        break;
                    }
                }

                Ok(InternalCommand::Shutdown) => {
                    debug!("Data thread received Shutdown command");
                    running = false;
                }

                Err(mpsc::TryRecvError::Empty) => {
                    // No command available, continue streaming
                }

                Err(mpsc::TryRecvError::Disconnected) => {
                    debug!("Command channel disconnected");
                    break;
                }
            }

            // Stream data for active channels
            let mut did_work = false;
            if let Some(ref mut dev) = device {
                for (channel, state) in active_streams.iter_mut() {
                    match dev.read_samples(*channel, &mut sample_buffer, 100_000) {
                        Ok(n) if n > 0 => {
                            // Check for pending commands before potentially blocking send
                            // This prevents deadlock when parent stops reading and sends StopStream
                            match cmd_rx.try_recv() {
                                Ok(cmd) => {
                                    // Command available - save it and break to process at top of loop
                                    pending_command = Some(cmd);
                                    break;
                                }
                                Err(mpsc::TryRecvError::Disconnected) => {
                                    debug!("Command channel disconnected while streaming");
                                    break;
                                }
                                Err(mpsc::TryRecvError::Empty) => {
                                    // No command - proceed with send
                                }
                            }

                            let packet = IQPacket {
                                channel: *channel,
                                samples: sample_buffer[..n].to_vec(),
                                timestamp: SystemTime::now()
                                    .duration_since(SystemTime::UNIX_EPOCH)
                                    .unwrap()
                                    .as_micros() as u64,
                                sequence: state.sequence_number,
                            };

                            state.sequence_number += 1;

                            match dat_sender.send(&packet) {
                                Ok(_) => {
                                    did_work = true;
                                }
                                Err(e) => {
                                    let is_timeout_or_backpressure = if let Some(io_err) =
                                        e.source().and_then(|s| s.downcast_ref::<std::io::Error>())
                                    {
                                        matches!(
                                            io_err.kind(),
                                            std::io::ErrorKind::WouldBlock
                                                | std::io::ErrorKind::TimedOut
                                        )
                                    } else {
                                        false
                                    };

                                    if is_timeout_or_backpressure {
                                        if state.sequence_number % 100 == 0 {
                                            debug!(
                                                channel = *channel,
                                                "Backpressure/timeout: dropping samples"
                                            );
                                        }
                                        break;
                                    } else {
                                        debug!(channel = *channel, error = %e, "Data send error");
                                    }
                                }
                            }
                        }
                        Ok(_) => {}
                        Err(e) => {
                            if e.to_string().contains("Timeout") {
                                // Timeout is normal - break out of streaming loop to check for commands
                                break;
                            } else {
                                debug!(channel = *channel, error = %e, "Stream read error");
                            }
                        }
                    }
                }
            }

            // If we didn't send any data (timeouts or no active streams), sleep briefly
            // to avoid busy-looping while waiting for commands or parent to resume reading
            if !did_work && !active_streams.is_empty() {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }

        // Clean up active streams
        if let Some(ref mut dev) = device {
            for (channel, _) in active_streams.drain() {
                let _ = dev.stop_stream(channel);
            }
        }

        debug!("Data thread exiting");
    });

    control_thread
        .join()
        .map_err(|_| ScannerError::Custom("Control thread panicked".to_string()))?;
    data_thread
        .join()
        .map_err(|_| ScannerError::Custom("Data thread panicked".to_string()))?;

    Ok(())
}
