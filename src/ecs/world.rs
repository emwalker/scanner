//! Entity storage for ECS architecture

use super::entity::Entity;
use std::collections::HashMap;

/// Storage for entities of a specific type
///
/// EntityWorld provides a simple HashMap-based storage for entities.
/// Unlike full ECS implementations (like Bevy), we don't optimize for
/// cache-friendly memory layout since we have dozens of entities, not thousands.
pub struct EntityWorld<E: Entity> {
    entities: HashMap<E::Id, E>,
}

impl<E: Entity> EntityWorld<E> {
    /// Create a new empty entity world
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
        }
    }

    /// Insert an entity into the world
    ///
    /// If an entity with the same ID already exists, it will be replaced
    /// and the old entity will be returned.
    pub fn insert(&mut self, entity: E) -> Option<E> {
        let id = entity.id().clone();
        self.entities.insert(id, entity)
    }

    /// Get an entity by ID
    pub fn get(&self, id: &E::Id) -> Option<&E> {
        self.entities.get(id)
    }

    /// Get a mutable reference to an entity by ID
    pub fn get_mut(&mut self, id: &E::Id) -> Option<&mut E> {
        self.entities.get_mut(id)
    }

    /// Remove an entity by ID
    ///
    /// Returns the removed entity if it existed.
    pub fn remove(&mut self, id: &E::Id) -> Option<E> {
        self.entities.remove(id)
    }

    /// Iterate over all entities
    pub fn iter(&self) -> impl Iterator<Item = &E> {
        self.entities.values()
    }

    /// Iterate over all entities mutably
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut E> {
        self.entities.values_mut()
    }

    /// Get the number of entities in the world
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Check if the world is empty
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Clear all entities from the world
    pub fn clear(&mut self) {
        self.entities.clear();
    }
}

impl<E: Entity> Default for EntityWorld<E> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct MockEntity {
        id: u64,
        data: String,
    }

    impl Entity for MockEntity {
        type Id = u64;

        fn id(&self) -> &Self::Id {
            &self.id
        }
    }

    impl MockEntity {
        fn new(id: u64, data: &str) -> Self {
            Self {
                id,
                data: data.to_string(),
            }
        }
    }

    #[test]
    fn test_new_world_is_empty() {
        let world: EntityWorld<MockEntity> = EntityWorld::new();
        assert!(world.is_empty());
        assert_eq!(world.len(), 0);
    }

    #[test]
    fn test_insert_and_get() {
        let mut world = EntityWorld::new();
        let entity = MockEntity::new(1, "test");

        assert_eq!(world.insert(entity.clone()), None);
        assert_eq!(world.len(), 1);

        let retrieved = world.get(&1);
        assert_eq!(retrieved, Some(&entity));
    }

    #[test]
    fn test_insert_replaces_existing() {
        let mut world = EntityWorld::new();
        let entity1 = MockEntity::new(1, "first");
        let entity2 = MockEntity::new(1, "second");

        assert_eq!(world.insert(entity1.clone()), None);
        let old = world.insert(entity2.clone());
        assert_eq!(old, Some(entity1));
        assert_eq!(world.len(), 1);
        assert_eq!(world.get(&1), Some(&entity2));
    }

    #[test]
    fn test_get_mut() {
        let mut world = EntityWorld::new();
        let entity = MockEntity::new(1, "test");
        world.insert(entity);

        let entity_mut = world.get_mut(&1).unwrap();
        entity_mut.data = "modified".to_string();

        assert_eq!(world.get(&1).unwrap().data, "modified");
    }

    #[test]
    fn test_remove() {
        let mut world = EntityWorld::new();
        let entity = MockEntity::new(1, "test");
        world.insert(entity.clone());

        let removed = world.remove(&1);
        assert_eq!(removed, Some(entity));
        assert!(world.is_empty());
        assert_eq!(world.get(&1), None);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut world: EntityWorld<MockEntity> = EntityWorld::new();
        assert_eq!(world.remove(&999), None);
    }

    #[test]
    fn test_iter() {
        let mut world = EntityWorld::new();
        world.insert(MockEntity::new(1, "first"));
        world.insert(MockEntity::new(2, "second"));
        world.insert(MockEntity::new(3, "third"));

        let mut ids: Vec<u64> = world.iter().map(|e| *e.id()).collect();
        ids.sort();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_iter_mut() {
        let mut world = EntityWorld::new();
        world.insert(MockEntity::new(1, "first"));
        world.insert(MockEntity::new(2, "second"));

        for entity in world.iter_mut() {
            entity.data = format!("modified_{}", entity.id);
        }

        assert_eq!(world.get(&1).unwrap().data, "modified_1");
        assert_eq!(world.get(&2).unwrap().data, "modified_2");
    }

    #[test]
    fn test_clear() {
        let mut world = EntityWorld::new();
        world.insert(MockEntity::new(1, "first"));
        world.insert(MockEntity::new(2, "second"));
        assert_eq!(world.len(), 2);

        world.clear();
        assert!(world.is_empty());
        assert_eq!(world.len(), 0);
    }

    #[test]
    fn test_multiple_entities() {
        let mut world = EntityWorld::new();

        for i in 0..100 {
            world.insert(MockEntity::new(i, &format!("entity_{}", i)));
        }

        assert_eq!(world.len(), 100);

        for i in 0..100 {
            assert!(world.get(&i).is_some());
        }

        for i in (0..100).step_by(2) {
            world.remove(&i);
        }

        assert_eq!(world.len(), 50);
    }
}
