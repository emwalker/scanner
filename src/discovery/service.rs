use crate::sdr;
use std::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub trait Service: Send {
    fn run(&mut self, event_tx: mpsc::Sender<Event>, cancel: CancellationToken);
}

#[derive(Debug, Clone)]
pub enum Event {
    Added(sdr::TunerInfo),
    Removed(sdr::TunerId),
}
