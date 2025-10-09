use clap::Parser;
use scanner::cli::{Cli, Commands};
use scanner::core::types::Result;

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Scan(args) => scanner::cli::handle_scan_command(*args),
        Commands::Train(args) => scanner::cli::handle_train_command(args),
    }
}
