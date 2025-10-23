use clap::Parser;
use scanner::{
    cli::{Cli, Commands, WorkerCommand},
    core::types::Result,
};

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Scan(args) => scanner::cli::handle_scan_command(*args),
        Commands::Train(args) => scanner::cli::handle_train_command(args),
        Commands::Worker(cmd) => match cmd {
            WorkerCommand::Enumerate {
                backend,
                socket_path,
                log_file,
            } => {
                scanner::cli::handle_enumerate_command(&backend, &socket_path, log_file.as_deref())
            }
            WorkerCommand::Device {
                device_id_str,
                control_socket_path,
                data_socket_path,
                log_file,
            } => scanner::cli::handle_device_command(
                &device_id_str,
                &control_socket_path,
                &data_socket_path,
                log_file.as_deref(),
            ),
        },
    }
}
