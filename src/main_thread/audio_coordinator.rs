use crate::audio::session::AudioSession;
use tracing::debug;

#[allow(dead_code)]
pub struct AudioCoordinator;

impl AudioCoordinator {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self
    }

    #[allow(dead_code)]
    pub fn stop_listening(
        &self,
        audio_session: &mut Option<AudioSession>,
        station_id: Option<crate::ecs::components::station::StationId>,
        audio_id: Option<crate::ecs::components::audio::AudioId>,
        station_entities: &Option<
            std::sync::Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::StationEntity>>>,
        >,
        audio_entities: &Option<
            std::sync::Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::AudioEntity>>>,
        >,
        candidate_entities: &Option<
            std::sync::Arc<std::sync::RwLock<crate::ecs::EntityWorld<crate::ecs::CandidateEntity>>>,
        >,
    ) {
        debug!("Stopped listening, returning to browsing mode");

        if let Some(session) = audio_session {
            session.stop_current_station();
        }

        // Extract minimal data from entities while holding locks
        let event_data = if let (Some(sid), Some(aid)) = (station_id, audio_id) {
            // Query station data
            let station_data = if let Some(entities_arc) = station_entities {
                if let Ok(entities) = entities_arc.read() {
                    entities.get(&sid).map(|s| {
                        (
                            s.frequency(),
                            s.discovery.window_id,
                            s.info.audio_quality,
                            s.signal_strength() as f64,
                        )
                    })
                } else {
                    None
                }
            } else {
                None
            };

            // Query audio data
            let audio_data = if let Some(entities_arc) = audio_entities {
                if let Ok(entities) = entities_arc.read() {
                    entities
                        .get(&aid)
                        .map(|a| (a.tuning.center_frequency_hz, a.tuner_id().cloned()))
                } else {
                    None
                }
            } else {
                None
            };

            // Combine the data
            if let (Some((freq, window_id, quality, strength)), Some((center_freq, tuner_id))) =
                (station_data, audio_data)
            {
                Some((freq, window_id, quality, strength, center_freq, tuner_id))
            } else {
                None
            }
        } else {
            None
        };

        // Use extracted data to send event (no locks held)
        if let Some((
            frequency_hz,
            window_id,
            _audio_quality,
            _signal_strength,
            _center_freq,
            _tuner_id,
        )) = event_data
        {
            // Update candidate entity to Completed state
            if let Some(entities_arc) = candidate_entities {
                use crate::ecs::CandidateId;
                let id = CandidateId::new(frequency_hz, window_id);
                if let Ok(mut entities) = entities_arc.write()
                    && let Some(entity) = entities.get_mut(&id)
                {
                    entity.complete_playback();
                }
            }
        }
    }
}
