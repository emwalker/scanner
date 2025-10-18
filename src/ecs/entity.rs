//! Entity trait for ECS architecture

use std::hash::Hash;

/// Trait for entities in the ECS system
///
/// Entities are lightweight identifiers that tie together components.
/// Each entity has a unique ID that identifies it within its entity world.
pub trait Entity: Sized {
    /// The type used to uniquely identify this entity
    type Id: Clone + Eq + Hash;

    /// Get the unique identifier for this entity
    fn id(&self) -> &Self::Id;
}
