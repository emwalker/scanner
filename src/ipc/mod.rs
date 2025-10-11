mod mock;
mod protocol;
mod traits;

// Export traits
pub use traits::{ControlChannel, DataReceiver, DataSender};

// Export message types
pub use protocol::{ControlMessage, IQPacket};

// Export concrete Unix socket implementations
pub use protocol::{UnixControlChannel, UnixDataReceiver, UnixDataSender};

// Export mock implementations for testing
pub use mock::{MockControlChannel, MockDataReceiver, MockDataSender};
