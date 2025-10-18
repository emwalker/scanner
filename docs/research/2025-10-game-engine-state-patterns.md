# Video Game State Management Patterns Research

## Research Context

Investigation into how video games handle complex stateful interactions including hierarchical menu systems, character attributes with modifiers, temporary buffs/debuffs, and component interactions.

## Entity Component System (ECS)

The dominant architectural pattern in modern game engines for managing complex state.

### Core Architecture

Entity Component System (ECS) is a software architectural pattern mostly used in video game development for representation of game world objects. An ECS comprises:

**Entities**: Unique identifiers for game objects, generally represented as integer values. An entity is a single "thing" in the game.

**Components**: Data types that can be added to or removed from entities. Components are generally plain data types and not encapsulated. They represent the state of the game.

**Systems**: Executable objects that operate on entities with specific component combinations. Systems operate on the state of the game.

### Key Principles

ECS follows the principle of **composition over inheritance**. Every entity is defined not by a type hierarchy, but by the components associated with it.

**Strict separation between data and logic**: Components hold state variables, systems contain logic. This allows optimal memory alignment for fast sequential access.

### State Management

The merits of using ECS for storing game state have been proclaimed by many game developers. From that comes an architecture of components representing game state, and systems operating on that state.

### Performance Benefits

In ECS, component objects are simple structures holding state variables. Thanks to strict separation between data and logic, data can be optimally aligned in memory, allowing fast sequential access.

### Unity vs Bevy Implementation

**Unity ECS**: Tables with columns of components. An entity type is based on components it holds. Each entity type has a table (called an archetype) holding columns of components.

**Bevy ECS (Rust)**: Uses explicit dependency declarations between systems and system sets to control execution order. Automatically determines parallelization opportunities based on what data each system accesses. Systems are normal Rust functions, and the type system determines what data needs to be sent to the system.

## Game Loop and Update Pattern

The fundamental pattern for managing time-based state updates.

### Game Loop Structure

A game loop runs continuously during gameplay. Each turn of the loop:
1. Processes user input without blocking
2. Updates game state
3. Renders the game
4. Tracks passage of time to control gameplay rate

### Update Method

Even the most basic game loops include a function, often called `Update` or `Tick`, which handles anything that needs to be updated each frame. The common terms for one crank of the game loop are "tick" or "frame".

### Three Main Phases

The loop comprises:
- **Initialize**: Setup game state
- **Update**: Process input, update entities, handle physics
- **Draw**: Render current state

### Time Management

A game state update-rate (ticksPerSecond) ensures game state is updated consistently. Games process user input but don't stop and wait for it (non-blocking).

### Loop Variations

- Fixed time step loops
- Variable time step loops
- Frame-skipping loops
- Independent update/render loops with interpolation

## Behavior Trees and State Machines

Complementary patterns for managing AI and gameplay logic.

### Behavior Trees

Behavior trees have been extended into a tree-like organization of behaviors with extensive application in the game industry as a powerful tool to model NPC behavior. Used in high-profile games like Halo, Bioshock, and Spore.

Behavior trees are now treated in Game AI textbooks and generic game environments like Unity and Unreal Engine.

### Finite State Machines (FSMs)

At the highest level, behavior trees are used for AI while finite state machines are used for more general visual programming.

State machines consist of:
- **States**: Discrete conditions or modes
- **Transitions**: Conditions that trigger state changes
- **Events**: Triggers for transitions

### Comparison

Behavior trees allow AI to react to current game state easier than finite state machines. It's easier to create a behavior tree that reacts to all sorts of situations, whereas it would take many states and transitions with an FSM for similar AI.

### Hybrid Approaches

**Unreal StateTree**: A general-purpose hierarchical state machine that combines Selectors from behavior trees with States and Transitions from state machines.

**Behavior trees and FSMs don't have to be mutually exclusive**: Behavior trees can describe the flow of AI while the FSM describes the function.

### Unity State Machines

State machines offer a structured and modular approach to managing complex behaviors by organizing code into discrete states, improving readability, maintainability, and scalability.

State machines help developers create more dynamic and responsive gameplay experiences by organizing different game states into separate states and utilizing a state machine to manage state transitions.

## Status Effects and Modifier Systems

How games handle temporary buffs, debuffs, and attribute modifications.

### Status Effect Definition

Status effects are temporary modifications to a character's attributes, abilities, or conditions:
- **Buffs**: Positive enhancements to attributes like damage, defense, or speed
- **Debuffs**: Negative conditions such as reduced damage, immobilization, or damage-over-time

### Architecture Approaches

**Component-Based Approach (ECS)**:
When a status effect is applied to an entity, that entity gets a component which is removed when the status effect expires or is otherwise removed. One component per status effect/buff.

**Buff/Debuff Class System**:
- Buff/debuff class represents the effect itself
- Buff/debuff type class contains information about what the buff affects and how
- Characters, items, and possibly locations have a list or collection property to contain buffs and debuffs

### Two Main Storage Methods

1. **Direct Modification**: Buffs directly add to stats and subtract their value when they end
2. **Array Processing**: Buffs tracked in array/list, and when stats are relevant, the array is processed to find current value

### Key Properties

Buff/debuff types contain:
- Who/what it can be applied to (player, monster, location, item)
- What type of effect it is (positive, negative)
- Whether it is multiplicative or additive
- What type of stat it impacts
- When it should be checked
- Whether it can be removed

### Time Management

Simple approach uses array of flags identifying buffs/debuffs with associated countdown timers:
- Each timer decreases with each frame elapsed
- When timer reaches zero, effects turn off
- Timers increment when buffs/debuffs are reapplied

### Unity Framework Pattern

Status effects defined as effects that alter character attributes until some condition is met.

Key properties:
- **Varied behaviors**: A buff can do pretty much any modification to a character
- **Lifetime**: The buff is applied for some time and then wears off
- **State tracking**: System tracks active effects

## Attribute Calculation and Modifier Stacks

How games calculate final attribute values from base stats plus modifiers.

### Modifier Types

**Shifting Modifiers**: Add or remove a set value (e.g., +10 strength)

**Scaling Modifiers**: Multiply by a factor (e.g., ×1.5 damage)

### Two Stacking Approaches

**Iterative/Stacked Approach**:
- Each cumulative modifier has dramatic effect
- Five ×2 modifiers multiply stat by 32
- Adding one more ×2 modifier multiplies stat by 64
- Order of application matters

**From Base Approach**:
- Each modifier acts on base value
- Five ×2 modifiers result in stat multiplied by 5 (not 32)
- Adding another multiplies total by 6 (not 64)
- Guarantees reasonable increases
- Order of application doesn't matter

### Terminology

ARPGs like Path of Exile and Diablo use "additive" and "multiplicative" to describe stacking behavior. Easy way is to clearly specify whether it is total strength or base strength being modified.

### Implementation Pattern

Characters contain:
- Set of **Attributes** indexed by type (Strength, Intelligence, Health)
- Attributes contain current and max values
- **CharacterEffects** that contain **AttributeEffects**
- AttributeEffects have modifier types (Add, Multiply, SetAbsolute) and values

### Data-Driven Design

Data-driven design means putting information in external storage, loading it at runtime, and acting on it. Application code does what external data tells it to, rather than writing code that directly does what you think the end result should be.

## Observer Pattern and Event Systems

How game components communicate without tight coupling.

### Observer Pattern

A software design technique where objects (observers) can sign up to be notified of events happening in some other object (the subject).

The observer pattern allows outside code to control notifications while the subject communicates with observers without being coupled to them.

### Game Applications

Particularly useful for:
- Health bars updating when character takes damage
- Achievement systems triggering on player actions
- UI updates responding to game state changes
- Audio cues for gameplay events

### Observer vs Events

**Observer Systems**: You observe the thing that did something interesting (the subject)

**Event Systems**: You observe an object that represents the interesting thing that happened (the event)

### Component Communication

In component-based systems, communication can be blind:
- Components emit events to the void (or event manager)
- Subscribers get events from event manager
- One-way communication
- Components decoupled - game objects don't have to be directly tied to objects they interact with

### Unity Implementation

With observer pattern, an object (subject) keeps a list of dependents (observers). When something happens in the game, the subject can invoke a function that observers subscribe to.

## Game UI State Management

How hierarchical menu systems and navigation flows are managed.

### Game State Management Pattern

Typically involves using "current_state" variable to manage different screens like titlescreen, game, and gameover. The approach essentially turns on and off various systems.

Common pattern is separating logic into different screens:
- Define an interface called Screen
- Multiple screens implement it: LoadingScreen, MainMenuScreen, GameScreen, GameOverScreen, HighScoreScreen

### Directed Acyclic Graph (DAG) Structure

Some developers use DAG data structure with nodes representing game systems (UI, world, input, rendering) where each node points at other nodes that come before or after it.

### Navigation Design (UI Map)

Also referred to as Information Architecture, Interaction Architecture, or UI Map.

Process:
- Identify menus and core features
- Arrange them as a flowchart
- Document all options, features, input requirements, and notifications

**Benefits**:
- Helps designers and engineers understand navigation path
- Identifies potential roadblocks
- Estimates number of screens players must move through

**Best Practices**:
- Minimize effort needed to navigate within UI flow
- Incorporate shortcuts for circumnavigating menu sections
- Use UI Map to estimate screen transitions

### Hierarchical Menu Structure

Menu systems organized with:
- **Categories** at the base
- Each category can have multiple **menu items** that players select
- Categories can have **parent categories** (hierarchical nesting)

### Visual Flow Editors

Tools like UI Graph for Unity allow creating complete user interfaces by:
- Visually creating flow in node-based editor
- Constructing complex multi-screen layouts by nesting screens
- Managing state transitions graphically

## Unreal Engine Specific Patterns

### Game Framework Component Manager

A Game Instance Subsystem in Modular Gameplay plugin that provides functionality for use with Game Feature Plugins. Functions can be used by Game Feature Actions to support extensibility.

### Initialization State System

Provides functions for tracking initialization and general lifecycle of different features attached to Actors. Not meant to be a generic gameplay state machine because states are globally defined for an entire game.

### Modular Architecture

Unreal emphasizes modular game features that can be added/removed dynamically, requiring careful state management across feature boundaries.

## Key Takeaways for Complex State Management

### Composition Over Inheritance

Modern games universally favor composition (ECS) over inheritance hierarchies for managing complex state. This provides:
- Better performance through data-oriented design
- Easier addition/removal of capabilities at runtime
- Clearer separation of concerns

### Separation of Data and Logic

Components hold state, systems process state. This strict separation enables:
- Optimal memory layout
- Better cache performance
- Easier parallelization
- Clearer code organization

### Event-Driven Communication

Observer pattern and event systems enable loose coupling between components:
- UI can react to game state without direct references
- Achievements can listen for player actions
- Audio can respond to gameplay events
- All without tight coupling

### Layered State Management

Games use different patterns at different levels:
- **Low-level**: ECS for entity state
- **Mid-level**: State machines for AI and gameplay logic
- **High-level**: Screen/scene managers for UI and game modes

### Time-Based Updates

Everything flows through the game loop's update tick:
- Consistent update rate
- Deterministic state progression
- Easy rollback/replay for networking
- Clear temporal ordering

### Data-Driven Design

Complex state is externalized to data files:
- Modifiers defined in data, not code
- Abilities specified declaratively
- Easy balance adjustments without recompilation
- Designers can iterate without programmer involvement

## Implications for SDR Scanner Application

### Applicable Patterns

**Component-Based Design**: Scanner could use components for:
- Device state (connected, scanning, listening)
- Current operation (which window, which frequency)
- Audio session state
- UI state

**Event System**: Decouple device events from UI:
- Device added/removed events
- Scan progress events
- Station detection events
- Audio quality events

**State Machine**: Clear states for scanning operations:
- Idle, Scanning, Listening, Paused
- Explicit transitions with guards
- State-specific data in substates

**Modifier Pattern**: For audio processing parameters:
- Base sample rate
- Gain modifiers
- Filter modifiers
- Easy stacking and removal

**Game Loop Pattern**: Already have this!
- Task scheduler is essentially game loop
- Tasks update on each iteration
- Time-based progression

### Not Directly Applicable

**ECS Full Architecture**: Overkill for this application
- Don't have thousands of entities
- Don't need high-performance parallel iteration
- Current architecture is already working well

**Behavior Trees**: Not needed
- Don't have complex AI
- State machines sufficient for device/scan logic

**Complex UI State**: Limited UI complexity
- Not a 3D game with camera, inventory, etc.
- TUI has simpler state needs

## References

- Entity Component System: https://en.wikipedia.org/wiki/Entity_component_system
- ECS FAQ: https://github.com/SanderMertens/ecs-faq
- Game Programming Patterns: https://gameprogrammingpatterns.com/
- Game Loop Pattern: https://gameprogrammingpatterns.com/game-loop.html
- Update Method Pattern: https://gameprogrammingpatterns.com/update-method.html
- Observer Pattern: https://gameprogrammingpatterns.com/observer.html
- Bevy ECS Documentation: https://docs.rs/bevy_ecs/latest/bevy_ecs/
- Unity State Machines: https://docs.unity3d.com/Manual/StateMachineBasics.html
- Behavior Trees for AI: https://www.gamedeveloper.com/programming/behavior-trees-for-ai-how-they-work
- Status Effects in ECS: https://www.gamedev.net/forums/topic/692150-status-effects-buffs-debuffs-in-an-ecs-architecture/
- Character Stats/Modifiers: https://medium.com/@kryzarel/character-stats-attributes-in-unity-pt-2-additive-modifiers-optimizations-1dfb2d42f3c8
