use crate::core::types::Result;
use crate::scanning::window::{create_audio_stream, setup_audio_device};
use cpal::traits::StreamTrait;
use cpal::{BufferSize, SampleFormat, StreamConfig};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use tracing::{debug, info};

pub enum AudioThreadCommand {
    Shutdown,
}

pub struct AudioThread {
    command_tx: mpsc::Sender<AudioThreadCommand>,
    thread_handle: Option<thread::JoinHandle<()>>,
}

impl AudioThread {
    pub fn new(
        audio_sample_rate: u32,
        audio_buffer_size: u32,
        audio_rx: mpsc::Receiver<crate::mpsc::AudioPacket>,
        volume: f32,
    ) -> Result<Self> {
        let (command_tx, command_rx) = mpsc::channel();

        let thread_handle = thread::spawn(move || {
            info!("AudioThread: Starting");

            let (audio_device, supported_config) = match setup_audio_device(audio_sample_rate) {
                Ok(result) => result,
                Err(e) => {
                    debug!(error = ?e, "AudioThread: Failed to setup audio device");
                    return;
                }
            };

            let sample_format = supported_config.sample_format();
            let mut stream_config: StreamConfig = supported_config.into();
            stream_config.buffer_size = BufferSize::Fixed(audio_buffer_size);

            let stream = match sample_format {
                SampleFormat::F32 => {
                    match create_audio_stream(&audio_device, &stream_config, audio_rx, volume) {
                        Ok(stream) => stream,
                        Err(e) => {
                            debug!(error = ?e, "AudioThread: Failed to create audio stream");
                            return;
                        }
                    }
                }
                _ => {
                    debug!("AudioThread: Unsupported audio format");
                    return;
                }
            };

            if let Err(e) = stream.play() {
                debug!(error = ?e, "AudioThread: Failed to start stream");
                return;
            }
            info!("AudioThread: Stream started");

            loop {
                match command_rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(AudioThreadCommand::Shutdown) => {
                        info!("AudioThread: Received Shutdown command");
                        break;
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        continue;
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        debug!("AudioThread: Command channel disconnected");
                        break;
                    }
                }
            }

            info!("AudioThread: Dropping stream");
            drop(stream);
            info!("AudioThread: Shutdown complete");
        });

        Ok(Self {
            command_tx,
            thread_handle: Some(thread_handle),
        })
    }

    pub fn shutdown(&mut self) {
        info!("AudioThread: Sending shutdown command");
        if let Err(e) = self.command_tx.send(AudioThreadCommand::Shutdown) {
            debug!(error = ?e, "AudioThread: Failed to send shutdown command (thread may have already exited)");
        }

        if let Some(handle) = self.thread_handle.take() {
            debug!("AudioThread: Waiting for thread to join");
            if let Err(e) = handle.join() {
                debug!(error = ?e, "AudioThread: Thread panicked during join");
            } else {
                debug!("AudioThread: Thread joined successfully");
            }
        }
    }
}

impl Drop for AudioThread {
    fn drop(&mut self) {
        debug!("AudioThread: Dropping");
        self.shutdown();
    }
}
