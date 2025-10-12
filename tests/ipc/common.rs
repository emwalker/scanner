use scanner::hardware::DeviceId;
use scanner::ipc::{
    ControlChannel, ControlMessage, DataReceiver, IQPacket, UnixControlChannel, UnixDataReceiver,
};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;
use tempfile::TempDir;

/// Test fixture for manual subprocess testing with explicit control
pub struct SubprocessTestFixture {
    pub socket_dir: TempDir,
    pub control_socket: PathBuf,
    pub data_socket: PathBuf,
    process: Option<Child>,
    control_channel: Option<UnixControlChannel>,
    data_receiver: Option<UnixDataReceiver>,
}

impl SubprocessTestFixture {
    pub fn new() -> Self {
        let socket_dir = TempDir::new().expect("Failed to create temp dir");
        let control_socket = socket_dir.path().join("control.sock");
        let data_socket = socket_dir.path().join("data.sock");

        Self {
            socket_dir,
            control_socket,
            data_socket,
            process: None,
            control_channel: None,
            data_receiver: None,
        }
    }

    pub fn spawn_worker(&mut self, device_id: DeviceId) -> Result<(), String> {
        let device_id_str = serde_json::to_string(&device_id)
            .map_err(|e| format!("Failed to serialize device_id: {}", e))?;

        let binary_path =
            std::env::current_exe().map_err(|e| format!("Failed to get current exe: {}", e))?;

        let binary_to_use = if binary_path.to_string_lossy().contains("/deps/") {
            binary_path
                .parent()
                .and_then(|p| p.parent())
                .ok_or("Failed to find target directory")?
                .join("scanner")
        } else {
            binary_path
        };

        let log_path = self.socket_dir.path().join("worker.log");

        let mut cmd = Command::new(binary_to_use);
        cmd.arg("worker")
            .arg("device")
            .arg("--device-id-str")
            .arg(&device_id_str)
            .arg("--control-socket-path")
            .arg(&self.control_socket)
            .arg("--data-socket-path")
            .arg(&self.data_socket)
            .arg("--log-file")
            .arg(&log_path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        let child = cmd
            .spawn()
            .map_err(|e| format!("Failed to spawn worker: {}", e))?;

        self.process = Some(child);

        let timeout = Duration::from_secs(5);
        let start = std::time::Instant::now();
        while !self.control_socket.exists() || !self.data_socket.exists() {
            if start.elapsed() > timeout {
                return Err("Socket creation timeout".to_string());
            }
            std::thread::sleep(Duration::from_millis(10));
        }

        std::thread::sleep(Duration::from_millis(50));

        Ok(())
    }

    pub fn connect(&mut self) -> Result<(), String> {
        let control_stream = UnixStream::connect(&self.control_socket)
            .map_err(|e| format!("Failed to connect to control socket: {}", e))?;

        let data_stream = UnixStream::connect(&self.data_socket)
            .map_err(|e| format!("Failed to connect to data socket: {}", e))?;

        data_stream
            .set_read_timeout(Some(Duration::from_millis(100)))
            .map_err(|e| format!("Failed to set read timeout: {}", e))?;

        self.control_channel = Some(UnixControlChannel::new(control_stream));
        self.data_receiver = Some(UnixDataReceiver::new(data_stream));

        Ok(())
    }

    pub fn send_control(&mut self, msg: &ControlMessage) -> Result<(), String> {
        let channel = self
            .control_channel
            .as_mut()
            .ok_or("Not connected - call connect() first")?;

        channel
            .send(msg)
            .map_err(|e| format!("Failed to send control message: {}", e))
    }

    pub fn recv_control(&mut self) -> Result<ControlMessage, String> {
        let channel = self
            .control_channel
            .as_mut()
            .ok_or("Not connected - call connect() first")?;

        channel
            .recv()
            .map_err(|e| format!("Failed to receive control message: {}", e))
    }

    pub fn recv_data(&mut self) -> Result<IQPacket, String> {
        let receiver = self
            .data_receiver
            .as_mut()
            .ok_or("Not connected - call connect() first")?;

        receiver
            .recv()
            .map_err(|e| format!("Failed to receive data packet: {}", e))
    }

    pub fn shutdown(&mut self) -> Result<(), String> {
        if let Some(ref mut process) = self.process {
            process
                .kill()
                .map_err(|e| format!("Failed to kill process: {}", e))?;
            process
                .wait()
                .map_err(|e| format!("Failed to wait for process: {}", e))?;
        }
        Ok(())
    }
}

impl Drop for SubprocessTestFixture {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

/// Helper to check for zombie scanner processes spawned by this test
pub fn assert_no_zombies() {
    let current_pid = std::process::id();

    let output = Command::new("ps")
        .args(["-o", "pid,ppid,state,comm", "--no-headers"])
        .output()
        .expect("Failed to run ps command");

    let stdout = String::from_utf8_lossy(&output.stdout);

    let scanner_zombies: Vec<_> = stdout
        .lines()
        .filter(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                let ppid = parts[1].parse::<u32>().ok();
                let state = parts[2];
                let comm = parts[3];

                ppid == Some(current_pid) && state.contains('Z') && comm.contains("scanner")
            } else {
                false
            }
        })
        .collect();

    assert!(
        scanner_zombies.is_empty(),
        "Scanner zombie process detected (child of test PID {}):\n{}",
        current_pid,
        scanner_zombies.join("\n")
    );
}

/// Helper to verify socket cleanup
pub fn assert_sockets_cleaned(pattern: &str) {
    let paths: Vec<_> = glob::glob(pattern)
        .expect("Failed to read glob pattern")
        .collect();

    assert_eq!(
        paths.len(),
        0,
        "Stale sockets found: {:?}",
        paths.into_iter().filter_map(Result::ok).collect::<Vec<_>>()
    );
}
