use crate::core::errors::ScannerError;
use crate::core::types::Result;
use crate::hardware::{Backend, Mock, Soapy, StreamingDevice};
use crate::ipc::{
    ControlChannel, ControlMessage, DataSender, IQPacket, UnixControlChannel, UnixDataSender,
};
use rustradio::Complex;
use std::collections::HashMap;
use std::os::unix::net::UnixListener;
use std::time::SystemTime;
use tracing::debug;

fn backend_from_name(name: &str) -> Result<Box<dyn Backend>> {
    match name {
        "soapy" => Ok(Box::new(Soapy)),
        "mock" => Ok(Box::new(Mock)),
        other => Err(ScannerError::Custom(format!("Unknown backend: {}", other))),
    }
}

pub fn handle_enumerate_command(
    backend_name: &str,
    socket_path: &str,
    log_file: Option<&str>,
) -> Result<()> {
    if let Some(log_path) = log_file {
        use std::fs::OpenOptions;
        use tracing_subscriber::fmt::writer::MakeWriterExt;

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)?;

        tracing_subscriber::fmt()
            .with_writer(file.with_max_level(tracing::Level::DEBUG))
            .with_ansi(false)
            .init();
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

    let result = backend_from_name(backend_name).and_then(|backend| backend.enumerate_devices());

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
        use tracing_subscriber::fmt::writer::MakeWriterExt;

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)?;

        tracing_subscriber::fmt()
            .with_writer(file.with_max_level(tracing::Level::DEBUG))
            .with_ansi(false)
            .init();
    }

    let device_id: crate::hardware::DeviceId = serde_json::from_str(device_id_str)?;
    debug!(device_id = ?device_id, "Device worker starting");

    let backend = backend_from_name(device_id.backend())?;
    let mut streaming_device = backend.open_streaming_device(&device_id)?;

    let ctl_listener = UnixListener::bind(control_socket_path)?;
    let dat_listener = UnixListener::bind(data_socket_path)?;

    debug!("Waiting for parent connections");
    let (ctl_stream, _) = ctl_listener.accept()?;
    let (dat_stream, _) = dat_listener.accept()?;

    let mut ctl_channel = UnixControlChannel::with_cleanup(ctl_stream, control_socket_path.into());
    let mut dat_sender = UnixDataSender::with_cleanup(dat_stream, data_socket_path.into());

    ctl_channel.send(&ControlMessage::Ready {
        device_id: device_id_str.to_string(),
        channels: streaming_device.channels(),
    })?;

    debug!("Entering main loop");
    main_loop(&mut ctl_channel, &mut dat_sender, &mut *streaming_device)?;

    debug!("Device worker shutting down");
    Ok(())
}

struct StreamState {
    sequence_number: u64,
}

fn main_loop(
    ctl_channel: &mut impl ControlChannel,
    dat_sender: &mut impl DataSender,
    device: &mut dyn StreamingDevice,
) -> Result<()> {
    let mut active_streams: HashMap<usize, StreamState> = HashMap::new();
    let mut running = true;
    let mut sample_buffer = vec![Complex::new(0.0, 0.0); 2048];

    while running {
        if let Some(msg) = ctl_channel.try_recv()? {
            match msg {
                ControlMessage::ConfigureAndStart {
                    channel,
                    freq_hz,
                    gain_db,
                    sample_rate,
                } => {
                    debug!(channel, freq_hz, gain_db, sample_rate, "ConfigureAndStart");

                    let actual = device.configure_rx(channel, freq_hz, sample_rate, gain_db)?;
                    device.start_stream(channel)?;

                    active_streams.insert(channel, StreamState { sequence_number: 0 });

                    ctl_channel.send(&ControlMessage::StreamStarted {
                        channel,
                        actual_freq: actual.freq_hz,
                        actual_gain: actual.gain_db,
                        actual_sample_rate: actual.sample_rate,
                    })?;
                }

                ControlMessage::StopStream { channel } => {
                    debug!(channel, "StopStream");
                    device.stop_stream(channel)?;
                    active_streams.remove(&channel);

                    ctl_channel.send(&ControlMessage::StreamStopped { channel })?;
                }

                ControlMessage::Shutdown => {
                    debug!("Shutdown requested");
                    running = false;
                }

                _ => {
                    debug!("Unexpected control message");
                }
            }
        }

        for (channel, state) in active_streams.iter_mut() {
            match device.read_samples(*channel, &mut sample_buffer, 100_000) {
                Ok(n) if n > 0 => {
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

                    if let Err(e) = dat_sender.send(&packet) {
                        if e.to_string().contains("would block") {
                            if state.sequence_number % 100 == 0 {
                                debug!(channel = *channel, "Backpressure: dropping samples");
                            }
                        } else {
                            debug!(channel = *channel, error = %e, "Data send error");
                        }
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    debug!(channel = *channel, error = %e, "Stream read error");
                }
            }
        }
    }

    for (channel, _) in active_streams.drain() {
        let _ = device.stop_stream(channel);
    }

    Ok(())
}
