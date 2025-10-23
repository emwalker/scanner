#!/usr/bin/env python3
"""
Generate ASCII diagrams for Elm Architecture patterns.

Usage:
    python3 generate_diagrams.py [--fsm | --data-flow | --component]

Generates:
- FSM diagrams for state transitions
- Data flow diagrams for Model/Update/View
- Component composition diagrams
"""

import sys


def gen_elm_architecture_diagram() -> str:
    """Generate basic Elm Architecture data flow diagram."""
    return """
Elm Architecture: Model → View → Update Cycle

                    ┌─────────────────────┐
                    │      Model          │
                    │   (Application      │
                    │    State)           │
                    └──────────┬──────────┘
                               │
                        reads  │
                               ↓
                    ┌─────────────────────┐
                    │      View           │
                    │   (Pure Function)   │
                    │   state → widget    │
                    └──────────┬──────────┘
                               │
                            renders
                               │
                               ↓
                    ┌─────────────────────┐
                    │   Rendered UI       │
                    └──────────┬──────────┘
                               │
                         user interaction
                               │
                               ↓
                    ┌─────────────────────┐
                    │  Message/Event      │
                    │  Describes what     │
                    │  happened           │
                    └──────────┬──────────┘
                               │
                            sent to
                               │
                               ↓
                    ┌─────────────────────┐
                    │      Update         │
                    │   (Pure Function)   │
                    │ state + msg → state │
                    └──────────┬──────────┘
                               │
                        returns │
                               ↓
                    ┌─────────────────────┐
                    │   New Model         │
                    └─────────────────────┘
                               │
                               └─────→ (back to View)
"""


def gen_fsm_diagram() -> str:
    """Generate a generic FSM diagram template."""
    return """
Finite State Machine: State Transitions

Example: Scan Operation

                    ScanRequested
                        ↓
    ┌─────────┐      ┌──────────┐
    │  Idle   │─────→│ Scanning │
    └─────────┘      └──────────┘
        ↑                │
        │                │ ScanProgress
        │                ├─→ (stays in Scanning)
        │                │
        │        ┌───────┴────────┐
        │        │                │
        │   ScanFinished      ScanError
        │        │                │
        │        ↓                ↓
        │    ┌────────┐      ┌───────────┐
        │    │Complete│      │ Failed    │
        │    └────────┘      └───────────┘
        │        │                │
        └────────┴────────────────┘
               Reset
               (from any state)

States: Idle, Scanning, Complete, Failed
Messages: ScanRequested, ScanProgress, ScanFinished, ScanError, Reset
"""


def gen_nested_tea_diagram() -> str:
    """Generate nested TEA component composition diagram."""
    return """
Nested TEA: Component Composition

Parent Model
└── scan_dialog: ScanDialogModel
└── results: ResultsModel
└── settings: SettingsModel

Parent Message
├── ScanDialog(ScanDialogMessage)
├── Results(ResultsMessage)
└── Settings(SettingsMessage)

Component Hierarchy:

    ┌──────────────────────────────────────┐
    │       Parent View                    │
    │  ┌──────────────────────────────┐   │
    │  │  render_header(...)          │   │
    │  └──────────────────────────────┘   │
    │  ┌──────────────────────────────┐   │
    │  │  render_content(...)         │   │
    │  │  ┌────────────────────────┐  │   │
    │  │  │ ScanDialog::view()     │  │   │
    │  │  │ Results::view()        │  │   │
    │  │  │ Settings::view()       │  │   │
    │  │  └────────────────────────┘  │   │
    │  └──────────────────────────────┘   │
    │  ┌──────────────────────────────┐   │
    │  │  render_footer(...)          │   │
    │  └──────────────────────────────┘   │
    └──────────────────────────────────────┘

Message routing:
    User Input
        ↓
    Parent message handler
        ↓
    Route to child: Message::ScanDialog(msg)
        ↓
    Child update function processes msg
        ↓
    Child state updated
        ↓
    Child view renders with new state
"""


def gen_elm_ecs_integration_diagram() -> str:
    """Generate ECS/Elm integration diagram."""
    return """
Elm Architecture ↔ ECS Integration

ECS World                           Elm UI Model
(Core Logic)                        (State & View)
┌──────────────┐                   ┌──────────────┐
│ Entities     │                   │ Model        │
│ Components   │                   │ (UI State)   │
│ Systems      │                   └──────┬───────┘
└──────┬───────┘                          │
       │                                  │
       │ EcsEvent                Message  │
       │ (DeviceEnumerated)      ↑        │
       │ (ScanCompleted)         │        │
       │ (SignalDetected)        │        │
       │                         │        │
       └────────→ EventBus ←─────┴────────┘
                     ↓
            Message Queue
                     ↓
            update(model, msg)
                     ↓
            (new model, commands)
                     ↓
            ┌────────┴────────┐
            ↓                 ↓
         view()         execute_command()
            ↓                 ↓
          UI              ECS Events

Data Flow:
1. ECS emits event (e.g., ScanCompleted)
2. EventBus receives event
3. UI converts to Message
4. Update function processes message
5. Returns (new model, commands)
6. View renders new state
7. Commands executed by ECS
"""


def gen_state_and_view_separation() -> str:
    """Generate diagram showing separation of state from view."""
    return """
Separation: State Model vs View Rendering

                  Model (Pure Data)
                  ┌──────────────────┐
                  │ scan_state: Idle │
                  │ results: []      │
                  │ error: None      │
                  └────────┬─────────┘
                           │
                     no side effects
                     no I/O operations
                           │
                           ↓
                 View Function (Pure)
                  ┌──────────────────┐
                  │ For each state:  │
                  │  - Idle          │
                  │  - Loading       │
                  │  - Error         │
                  │  - Success       │
                  │ render accordingly│
                  └────────┬─────────┘
                           │
                     no state changes
                     no side effects
                           │
                           ↓
                   Rendered Widget
                  ┌──────────────────┐
                  │ Display in UI     │
                  │ Same for same     │
                  │ model state      │
                  └──────────────────┘
"""


def main():
    if len(sys.argv) < 2:
        print("Available diagrams:")
        print("  python3 generate_diagrams.py --elm         (Basic Elm Architecture)")
        print("  python3 generate_diagrams.py --fsm         (FSM Example)")
        print("  python3 generate_diagrams.py --nested      (Nested TEA)")
        print("  python3 generate_diagrams.py --ecs         (ECS Integration)")
        print("  python3 generate_diagrams.py --separation  (State/View Separation)")
        print("  python3 generate_diagrams.py --all         (All diagrams)")
        return

    arg = sys.argv[1]

    diagrams = {
        '--elm': gen_elm_architecture_diagram,
        '--fsm': gen_fsm_diagram,
        '--nested': gen_nested_tea_diagram,
        '--ecs': gen_elm_ecs_integration_diagram,
        '--separation': gen_state_and_view_separation,
    }

    if arg == '--all':
        for name, func in diagrams.items():
            print(f"\n{'='*60}")
            print(f"{name.strip('--').upper()}")
            print('='*60)
            print(func())
    elif arg in diagrams:
        print(diagrams[arg]())
    else:
        print(f"Unknown argument: {arg}")
        print("Use --all to see all diagrams")


if __name__ == '__main__':
    main()
