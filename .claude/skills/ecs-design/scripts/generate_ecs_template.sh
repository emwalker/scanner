#!/bin/bash
# generate_ecs_template.sh - Generate ECS component, entity, or system templates
# Usage: ./generate_ecs_template.sh <type> <name>
#   type: component, entity, or system
#   name: CamelCase name (e.g., MyComponent, MyEntity, MySystem)

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <type> <name>" >&2
    echo "  type: component, entity, or system" >&2
    echo "  name: CamelCase name (e.g., MyComponent, MyEntity, MySystem)" >&2
    exit 1
fi

TYPE=$1
NAME=$2

# Convert CamelCase to snake_case for filename
FILENAME=$(echo "$NAME" | sed 's/\([A-Z]\)/_\L\1/g' | sed 's/^_//')

case "$TYPE" in
    component)
        cat > "${FILENAME}.rs" << 'EOF'
use std::fmt;

/// TODO: Add component description
#[derive(Debug, Clone)]
pub struct COMPONENT_NAME {
    // TODO: Add component fields
}

impl COMPONENT_NAME {
    pub fn new() -> Self {
        Self {
            // TODO: Initialize fields
        }
    }

    // TODO: Add state transition methods
    // Example:
    // pub fn transition_to_state(&mut self) {
    //     self.state = NewState;
    // }
    //
    // pub fn is_in_state(&self) -> bool {
    //     matches!(self.state, ExpectedState)
    // }
}

impl Default for COMPONENT_NAME {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_creation() {
        let component = COMPONENT_NAME::new();
        // TODO: Add assertions
    }

    #[test]
    fn test_state_transitions() {
        let mut component = COMPONENT_NAME::new();
        // TODO: Test state transitions
        // component.transition_to_state();
        // assert!(component.is_in_state());
    }
}
EOF
        sed -i "s/COMPONENT_NAME/$NAME/g" "${FILENAME}.rs"
        echo "Created component template: ${FILENAME}.rs"
        echo "Next steps:"
        echo "1. Fill in component fields and methods"
        echo "2. Add component to src/ecs/components/mod.rs exports"
        echo "3. Write state transition tests"
        ;;

    entity)
        cat > "${FILENAME}.rs" << 'EOF'
use crate::ecs::{Entity, EntityWorld};
use std::fmt;

/// Unique identifier for ENTITY_NAME
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ENTITY_NAMEId(String);

impl ENTITY_NAMEId {
    pub fn new(id: String) -> Self {
        Self(id)
    }
}

impl fmt::Display for ENTITY_NAMEId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// TODO: Add entity description
pub struct ENTITY_NAME {
    id: ENTITY_NAMEId,
    // TODO: Add component fields
    // pub component_a: ComponentA,
    // pub component_b: ComponentB,
}

impl ENTITY_NAME {
    pub fn new(id: String /* TODO: Add component params */) -> Self {
        Self {
            id: ENTITY_NAMEId::new(id),
            // TODO: Initialize components
        }
    }

    // TODO: Add convenience query methods
    // pub fn is_available(&self) -> bool {
    //     self.component_a.check() && self.component_b.check()
    // }
}

impl Entity for ENTITY_NAME {
    type Id = ENTITY_NAMEId;

    fn id(&self) -> &Self::Id {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_creation() {
        let entity = ENTITY_NAME::new("test-id".to_string());
        // TODO: Add assertions
        assert_eq!(entity.id().to_string(), "test-id");
    }

    #[test]
    fn test_entity_queries() {
        let entity = ENTITY_NAME::new("test-id".to_string());
        // TODO: Test convenience methods
        // assert!(entity.is_available());
    }
}
EOF
        sed -i "s/ENTITY_NAME/$NAME/g" "${FILENAME}.rs"
        echo "Created entity template: ${FILENAME}.rs"
        echo "Next steps:"
        echo "1. Add component fields to the entity"
        echo "2. Implement convenience query methods"
        echo "3. Add entity to src/ecs/entities/mod.rs exports"
        echo "4. Write entity creation and query tests"
        ;;

    system)
        cat > "${FILENAME}.rs" << 'EOF'
use crate::ecs::{System, SystemContext};
use anyhow::Result;
use log::{debug, info};

/// TODO: Add system description
pub struct SYSTEM_NAME {
    // TODO: Add system state (e.g., pending requests queue)
}

impl SYSTEM_NAME {
    pub fn new() -> Self {
        Self {
            // TODO: Initialize system state
        }
    }
}

impl Default for SYSTEM_NAME {
    fn default() -> Self {
        Self::new()
    }
}

impl System for SYSTEM_NAME {
    fn name(&self) -> &'static str {
        "SYSTEM_NAME"
    }

    fn run(&mut self, context: &mut SystemContext) -> Result<()> {
        // TODO: Choose which entity world(s) to access
        // let entities = match &context.tuner_entities {
        //     Some(entities) => entities.clone(),
        //     None => return Ok(()),
        // };
        //
        // let entities = entities.lock().unwrap();
        // for entity in entities.iter() {
        //     // TODO: Process entity
        // }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::SystemContext;

    #[test]
    fn test_system_with_empty_context() {
        let mut system = SYSTEM_NAME::new();
        let mut context = SystemContext::new();

        let result = system.run(&mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_system_with_entities() {
        let mut system = SYSTEM_NAME::new();
        // TODO: Create test context with entities
        // let mut context = SystemContext::new()
        //     .with_tuner_entities(test_entities);
        //
        // let result = system.run(&mut context);
        // assert!(result.is_ok());
        // TODO: Add assertions about entity state changes
    }
}
EOF
        sed -i "s/SYSTEM_NAME/$NAME/g" "${FILENAME}.rs"
        echo "Created system template: ${FILENAME}.rs"
        echo "Next steps:"
        echo "1. Implement system logic in run() method"
        echo "2. Add system to src/ecs/systems/mod.rs exports"
        echo "3. Add system to scheduler in main loop"
        echo "4. Write tests for empty context and various entity states"
        ;;

    *)
        echo "Error: Unknown type '$TYPE'. Must be: component, entity, or system" >&2
        exit 1
        ;;
esac
