mod mock;
mod protocol;
mod traits;

// Export traits
// Export mock implementations for testing
pub use mock::{MockControlChannel, MockDataReceiver, MockDataSender};
// Export message types
pub use protocol::{ControlMessage, IQPacket};
// Export concrete Unix socket implementations
pub use protocol::{UnixControlChannel, UnixDataReceiver, UnixDataSender};
pub use traits::{ControlChannel, DataReceiver, DataSender};
