use crate::hardware::soapy;
use crate::shutdown::ShutdownCoordinator;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub fn setup_signal_handler(shutdown_coordinator: Arc<ShutdownCoordinator>) {
    static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

    #[allow(clippy::print_stderr)]
    ctrlc::set_handler(move || {
        if SHUTDOWN_REQUESTED.swap(true, Ordering::SeqCst) {
            eprintln!("\nForce quit - device may be left in inconsistent state");
            eprintln!("Run 'sudo systemctl restart sdrplay' if next startup fails");
            soapy::cleanup_soapysdr_state();
            std::process::exit(1);
        } else {
            eprintln!("\nShutting down gracefully...");
            eprintln!("Press Ctrl+C again to force quit");
            shutdown_coordinator.shutdown();
        }
    })
    .expect("Failed to set signal handler");
}
