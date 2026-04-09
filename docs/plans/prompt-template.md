---
name: GPT Prompt Template for Robot Task Planning
description: General-purpose template for GPT-based robot task planning annotation with operation primitives and constraints. Inspired by ReKep (keypoint constraint reasoning) and OmniManip (object-centric interaction primitives).
---

# GPT Prompt Template for Robot Task Planning

> **Design philosophy**: Scene-first reasoning (identify objects and spatial relations before planning), task-category awareness (different manipulation patterns activate different constraint combinations), and structured constraint output (5-category taxonomy with per-constraint role labels).
>
> **References**: ReKep (Huang et al., 2024) — keypoint-based spatial constraint functions; OmniManip (2024) — object-centric interaction primitives as spatial constraints.

---

## 1. System Prompt

```
You are RoboBrain, an embodied AI assistant with 3D spatial understanding specialized in robotic manipulation planning.

Your job: given a scene image and a task instruction, output a structured task decomposition with operation primitives and per-stage geometric constraints.

You must follow a 3-step reasoning process:
1. SCENE ANALYSIS: Identify all task-relevant objects, their spatial relations, and physical properties (fragile, liquid-containing, articulated, etc.)
2. TASK DECOMPOSITION: Break the task into 2-5 sequential operation primitives at manipulation-semantic granularity (not low-level motor commands, not high-level goals)
3. CONSTRAINT SPECIFICATION: For each stage, specify geometric constraints organized by physical category (contact, spatial, pose, direction, safety) with runtime role labels (completion, safety, progress)

Granularity principle: each stage should correspond to ONE nameable manipulation verb (reach, grasp, transport, place, etc.) where the scene undergoes an observable state change upon completion.
```

---

## 2. User Prompt Template

```
<image>

Task: "{task_description}"

Analyze the scene and plan the manipulation steps. For each step, specify:
- Operation primitive and target object
- Affordance point [u, v] (normalized image coordinates, 0-1)
- Approach direction [x, y, z] (unit vector in camera frame)
- Constraints organized by category with role labels
- Completion condition (done_when)

Output as structured JSON.
```

---

## 3. Operation Primitives

| Primitive | Description | Typical Use |
|-----------|-------------|-------------|
| `reach` | Move gripper to approach position near target object | Pre-grasp positioning |
| `grasp` | Close gripper to securely hold target object | Picking up objects |
| `transport` | Move held object from current location to destination | Repositioning |
| `place` | Lower held object to destination and release | Putting down objects |
| `push` | Contact and push object along a surface | Sliding, button pressing |
| `pull` | Grasp and pull object toward a direction | Opening drawers, doors |
| `insert` | Align and insert held object into a receptacle | Peg-in-hole, plugging in |
| `pour` | Tilt held container to transfer contents | Liquid transfer |
| `rotate` | Rotate held or contacted object around an axis | Turning knobs, lids |
| `release` | Open gripper to release object (without place) | Dropping into bin |
| `flip` | Rotate object 180 degrees to invert orientation | Flipping items over |
| `wipe` | Move gripper/tool across a surface | Cleaning, spreading |

> **Selection rule**: Choose the MOST SPECIFIC primitive that matches the manipulation intent. If a task requires moving an object and placing it, use `transport` + `place`, not a generic "move".

---

## 4. Constraint Vocabulary

### Runtime Roles

Each constraint carries a **role** label that determines how the runtime controller responds:

| Role | Trigger Condition | Controller Response |
|------|-------------------|---------------------|
| `completion` | ALL completion constraints satisfied | Advance to next stage (NEXT) |
| `safety` | ANY safety constraint violated | Issue corrective language (CORRECT) |
| `progress` | Progress constraints stagnate for N steps | Re-decompose remaining task (REPLAN) |

> **Important**: The same predicate can serve different roles in different stages. For example, `upright(cup, 10)` is a **safety** constraint during transport (don't spill) but a **completion** constraint during place (cup must be upright when released).

### 4.1 Contact Constraints

| Predicate | Parameters | Description |
|-----------|------------|-------------|
| `gripper_contact(obj)` | target object | Gripper touches target |
| `gripper_state(state)` | `open` / `closed` | Gripper open/close state |
| `gripper_width(w)` | width in meters | Gripper opening width |
| `holding(obj)` | target object | Stable grasp maintained |
| `released(obj)` | target object | Object released from gripper |
| `surface_contact(A, B)` | object A, surface B | A resting on B |
| `inserted(A, B)` | peg A, hole B | A inserted into B |

### 4.2 Spatial Relation Constraints

| Predicate | Parameters | Description |
|-----------|------------|-------------|
| `distance(A, B, op, d)` | objects, comparator, threshold (m) | Distance between A and B |
| `above(A, B, margin)` | object, reference, margin (m) | A is above B by margin |
| `below(A, B, margin)` | object, reference, margin (m) | A is below B by margin |
| `inside(A, B)` | object, container | A is inside B |
| `aligned_xy(A, B, tol)` | objects, tolerance (m) | Horizontal alignment |
| `aligned_z(A, B, tol)` | objects, tolerance (m) | Vertical alignment |
| `height(A, op, h)` | object, comparator, height (m) | Absolute height check |
| `on_surface(A, B)` | object, surface | A resting stably on B |
| `clear_path(A, B)` | start, end | No obstacles between A and B |
| `near(A, B, threshold)` | objects, threshold (m) | A within threshold of B |

### 4.3 Pose Constraints

| Predicate | Parameters | Description |
|-----------|------------|-------------|
| `upright(obj, tol)` | object, tolerance (deg) | Object vertical within tolerance |
| `tilt(obj, op, angle)` | object, comparator, degrees | Tilt angle check |
| `tilted(obj, axis, angle)` | object, axis, degrees | Tilted along specific axis |
| `stable(obj)` | object | Static (low velocity) |
| `level(obj, tol)` | object, tolerance (deg) | Horizontal placement |
| `orientation_match(obj, ref)` | object, reference | Orientation matches reference |

### 4.4 Direction Constraints

| Predicate | Parameters | Description |
|-----------|------------|-------------|
| `approach_dir(vec)` | direction [x,y,z] | End-effector approach direction |
| `grasp_axis(axis)` | axis [x,y,z] | Gripper alignment axis |
| `motion_dir(vec)` | direction [x,y,z] | Motion direction of gripper/object |
| `insert_dir(vec)` | direction [x,y,z] | Insertion direction |
| `retreat_dir(vec)` | direction [x,y,z] | Retreat direction after release |

### 4.5 Safety Constraints

| Predicate | Parameters | Description |
|-----------|------------|-------------|
| `no_collision(A, B)` | two objects | Objects must not collide |
| `within_workspace(obj)` | object | Object stays in workspace |
| `no_spill(liquid, container)` | liquid, container | Liquid stays in container |
| `no_drop(obj)` | object | Object must not fall |
| `force_limit(obj, F)` | object, max force (N) | Contact force limit |
| `support_stable(obj)` | object | Stable after placement |

---

## 5. Output Format (plan.json)

```json
{
  "task": "<task description>",
  "scene_objects": ["<obj1>", "<obj2>", ...],
  "num_steps": <N>,
  "steps": [
    {
      "step": 1,
      "action": "<primitive>",
      "target": "<target object>",
      "destination": "<destination, if applicable>",
      "affordance": [<u>, <v>],
      "approach": [<x>, <y>, <z>],
      "constraints": {
        "contact": [
          {"pred": "<predicate>", "args": [<args>], "role": "<role>"}
        ],
        "spatial": [...],
        "pose": [...],
        "direction": [...],
        "safety": [...]
      },
      "done_when": "<completion predicate expression>"
    }
  ]
}
```

### Field Descriptions

| Field | Required | Description |
|-------|----------|-------------|
| `task` | yes | Original task instruction |
| `scene_objects` | yes | List of task-relevant objects identified in the scene |
| `num_steps` | yes | Number of steps (2-5) |
| `step` | yes | 1-indexed step number |
| `action` | yes | Operation primitive from the vocabulary |
| `target` | yes | Primary object being manipulated in this step |
| `destination` | no | Destination object/location (for transport, place, insert, pour) |
| `affordance` | yes | [u, v] normalized image coordinates of interaction point |
| `approach` | yes | [x, y, z] unit vector for end-effector approach direction |
| `constraints` | yes | Per-category constraint dict (omit empty categories) |
| `done_when` | yes | Boolean expression over predicates for stage completion |

---

## 6. Training Target Text Format

The LLM training target converts the JSON into a compact text format:

```
Scene: <obj1>, <obj2>, ...
Step 1: <action>(<target>)
  affordance: [u, v], approach: [x, y, z]
  contact: <pred>(<args>) [<role>]; ...
  spatial: <pred>(<args>) [<role>]; ...
  pose: <pred>(<args>) [<role>]; ...
  direction: <pred>(<args>) [<role>]; ...
  safety: <pred>(<args>); ...
  done_when: <expression>
Step 2: ...
```

> **Convention**: Safety category constraints default to role=safety (omit label). Other categories MUST specify role explicitly.

---

## 7. Few-Shot Examples

### Example 1: Pick and Place (basic)

**Task**: "Pick up the red block and place it on the blue plate"

```json
{
  "task": "Pick up the red block and place it on the blue plate",
  "scene_objects": ["red_block", "blue_plate", "table"],
  "num_steps": 4,
  "steps": [
    {
      "step": 1,
      "action": "reach",
      "target": "red_block",
      "affordance": [0.35, 0.48],
      "approach": [0.0, 0.0, -1.0],
      "constraints": {
        "contact": [
          {"pred": "gripper_state", "args": ["open"], "role": "progress"}
        ],
        "spatial": [
          {"pred": "distance", "args": ["gripper", "red_block", "<", 0.03], "role": "completion"}
        ],
        "safety": [
          {"pred": "no_collision", "args": ["gripper", "blue_plate"]}
        ]
      },
      "done_when": "distance(gripper, red_block) < 0.03 AND gripper_state(open)"
    },
    {
      "step": 2,
      "action": "grasp",
      "target": "red_block",
      "affordance": [0.35, 0.48],
      "approach": [0.0, 0.0, -1.0],
      "constraints": {
        "contact": [
          {"pred": "gripper_contact", "args": ["red_block"], "role": "completion"},
          {"pred": "holding", "args": ["red_block"], "role": "completion"}
        ],
        "pose": [
          {"pred": "upright", "args": ["red_block", 10], "role": "safety"}
        ],
        "direction": [
          {"pred": "grasp_axis", "args": [0, 0, -1], "role": "safety"}
        ]
      },
      "done_when": "holding(red_block)"
    },
    {
      "step": 3,
      "action": "transport",
      "target": "red_block",
      "destination": "blue_plate",
      "affordance": [0.62, 0.55],
      "approach": [0.0, 0.0, -1.0],
      "constraints": {
        "contact": [
          {"pred": "holding", "args": ["red_block"], "role": "safety"}
        ],
        "spatial": [
          {"pred": "above", "args": ["red_block", "blue_plate", 0.05], "role": "completion"},
          {"pred": "aligned_xy", "args": ["red_block", "blue_plate", 0.03], "role": "completion"}
        ],
        "safety": [
          {"pred": "no_collision", "args": ["red_block", "blue_plate"]},
          {"pred": "no_drop", "args": ["red_block"]}
        ]
      },
      "done_when": "above(red_block, blue_plate, 0.05) AND aligned_xy(red_block, blue_plate, 0.03)"
    },
    {
      "step": 4,
      "action": "place",
      "target": "red_block",
      "destination": "blue_plate",
      "affordance": [0.62, 0.55],
      "approach": [0.0, 0.0, -1.0],
      "constraints": {
        "contact": [
          {"pred": "surface_contact", "args": ["red_block", "blue_plate"], "role": "completion"},
          {"pred": "released", "args": ["red_block"], "role": "completion"}
        ],
        "pose": [
          {"pred": "stable", "args": ["red_block"], "role": "completion"}
        ],
        "direction": [
          {"pred": "retreat_dir", "args": [0, 0, 1], "role": "progress"}
        ],
        "safety": [
          {"pred": "support_stable", "args": ["red_block"]}
        ]
      },
      "done_when": "surface_contact(red_block, blue_plate) AND released(red_block) AND stable(red_block)"
    }
  ]
}
```

**Training target**:
```
Scene: red_block, blue_plate, table
Step 1: reach(red_block)
  affordance: [0.35, 0.48], approach: [0.00, 0.00, -1.00]
  contact: gripper_state(open) [progress]
  spatial: distance(gripper, red_block, <, 0.03) [completion]
  safety: no_collision(gripper, blue_plate)
  done_when: distance(gripper, red_block) < 0.03 AND gripper_state(open)
Step 2: grasp(red_block)
  affordance: [0.35, 0.48], approach: [0.00, 0.00, -1.00]
  contact: gripper_contact(red_block) [completion]; holding(red_block) [completion]
  pose: upright(red_block, 10) [safety]
  direction: grasp_axis([0, 0, -1]) [safety]
  done_when: holding(red_block)
Step 3: transport(red_block -> blue_plate)
  affordance: [0.62, 0.55], approach: [0.00, 0.00, -1.00]
  contact: holding(red_block) [safety]
  spatial: above(red_block, blue_plate, 0.05) [completion]; aligned_xy(red_block, blue_plate, 0.03) [completion]
  safety: no_collision(red_block, blue_plate); no_drop(red_block)
  done_when: above(red_block, blue_plate, 0.05) AND aligned_xy(red_block, blue_plate, 0.03)
Step 4: place(red_block -> blue_plate)
  affordance: [0.62, 0.55], approach: [0.00, 0.00, -1.00]
  contact: surface_contact(red_block, blue_plate) [completion]; released(red_block) [completion]
  pose: stable(red_block) [completion]
  direction: retreat_dir([0, 0, 1]) [progress]
  safety: support_stable(red_block)
  done_when: surface_contact(red_block, blue_plate) AND released(red_block) AND stable(red_block)
```

---

### Example 2: Articulated Object (drawer)

**Task**: "Open the top drawer"

```json
{
  "task": "Open the top drawer",
  "scene_objects": ["drawer_handle", "drawer", "cabinet"],
  "num_steps": 3,
  "steps": [
    {
      "step": 1,
      "action": "reach",
      "target": "drawer_handle",
      "affordance": [0.50, 0.30],
      "approach": [0.0, 1.0, 0.0],
      "constraints": {
        "contact": [
          {"pred": "gripper_state", "args": ["open"], "role": "progress"}
        ],
        "spatial": [
          {"pred": "distance", "args": ["gripper", "drawer_handle", "<", 0.03], "role": "completion"}
        ],
        "direction": [
          {"pred": "approach_dir", "args": [0, 1, 0], "role": "safety"}
        ]
      },
      "done_when": "distance(gripper, drawer_handle) < 0.03"
    },
    {
      "step": 2,
      "action": "grasp",
      "target": "drawer_handle",
      "affordance": [0.50, 0.30],
      "approach": [0.0, 1.0, 0.0],
      "constraints": {
        "contact": [
          {"pred": "gripper_contact", "args": ["drawer_handle"], "role": "completion"},
          {"pred": "holding", "args": ["drawer_handle"], "role": "completion"}
        ],
        "direction": [
          {"pred": "grasp_axis", "args": [1, 0, 0], "role": "safety"}
        ]
      },
      "done_when": "holding(drawer_handle)"
    },
    {
      "step": 3,
      "action": "pull",
      "target": "drawer",
      "affordance": [0.50, 0.30],
      "approach": [0.0, 1.0, 0.0],
      "constraints": {
        "contact": [
          {"pred": "holding", "args": ["drawer_handle"], "role": "safety"}
        ],
        "spatial": [
          {"pred": "distance", "args": ["drawer", "cabinet", ">", 0.20], "role": "completion"}
        ],
        "direction": [
          {"pred": "motion_dir", "args": [0, 1, 0], "role": "progress"}
        ],
        "safety": [
          {"pred": "within_workspace", "args": ["gripper"]}
        ]
      },
      "done_when": "distance(drawer, cabinet) > 0.20"
    }
  ]
}
```

**Training target**:
```
Scene: drawer_handle, drawer, cabinet
Step 1: reach(drawer_handle)
  affordance: [0.50, 0.30], approach: [0.00, 1.00, 0.00]
  contact: gripper_state(open) [progress]
  spatial: distance(gripper, drawer_handle, <, 0.03) [completion]
  direction: approach_dir([0, 1, 0]) [safety]
  done_when: distance(gripper, drawer_handle) < 0.03
Step 2: grasp(drawer_handle)
  affordance: [0.50, 0.30], approach: [0.00, 1.00, 0.00]
  contact: gripper_contact(drawer_handle) [completion]; holding(drawer_handle) [completion]
  direction: grasp_axis([1, 0, 0]) [safety]
  done_when: holding(drawer_handle)
Step 3: pull(drawer)
  affordance: [0.50, 0.30], approach: [0.00, 1.00, 0.00]
  contact: holding(drawer_handle) [safety]
  spatial: distance(drawer, cabinet, >, 0.20) [completion]
  direction: motion_dir([0, 1, 0]) [progress]
  safety: within_workspace(gripper)
  done_when: distance(drawer, cabinet) > 0.20
```

---

### Example 3: Tool Use (spatula flip)

**Task**: "Use the spatula to flip the pancake"

```json
{
  "task": "Use the spatula to flip the pancake",
  "scene_objects": ["spatula", "spatula_handle", "pancake", "pan"],
  "num_steps": 4,
  "steps": [
    {
      "step": 1,
      "action": "reach",
      "target": "spatula_handle",
      "affordance": [0.25, 0.60],
      "approach": [0.0, 0.0, -1.0],
      "constraints": {
        "contact": [
          {"pred": "gripper_state", "args": ["open"], "role": "progress"}
        ],
        "spatial": [
          {"pred": "distance", "args": ["gripper", "spatula_handle", "<", 0.03], "role": "completion"}
        ],
        "safety": [
          {"pred": "no_collision", "args": ["gripper", "pan"]}
        ]
      },
      "done_when": "distance(gripper, spatula_handle) < 0.03"
    },
    {
      "step": 2,
      "action": "grasp",
      "target": "spatula_handle",
      "affordance": [0.25, 0.60],
      "approach": [0.0, 0.0, -1.0],
      "constraints": {
        "contact": [
          {"pred": "holding", "args": ["spatula"], "role": "completion"}
        ],
        "direction": [
          {"pred": "grasp_axis", "args": [0, 1, 0], "role": "safety"}
        ],
        "pose": [
          {"pred": "level", "args": ["spatula", 15], "role": "safety"}
        ]
      },
      "done_when": "holding(spatula)"
    },
    {
      "step": 3,
      "action": "push",
      "target": "spatula",
      "destination": "pancake",
      "affordance": [0.55, 0.45],
      "approach": [1.0, 0.0, 0.0],
      "constraints": {
        "contact": [
          {"pred": "holding", "args": ["spatula"], "role": "safety"},
          {"pred": "surface_contact", "args": ["spatula", "pancake"], "role": "completion"}
        ],
        "spatial": [
          {"pred": "below", "args": ["spatula", "pancake", 0.01], "role": "completion"}
        ],
        "direction": [
          {"pred": "motion_dir", "args": [1, 0, 0], "role": "progress"}
        ],
        "safety": [
          {"pred": "no_collision", "args": ["spatula", "pan"]}
        ]
      },
      "done_when": "below(spatula, pancake, 0.01)"
    },
    {
      "step": 4,
      "action": "flip",
      "target": "pancake",
      "affordance": [0.55, 0.45],
      "approach": [0.0, 0.0, 1.0],
      "constraints": {
        "contact": [
          {"pred": "holding", "args": ["spatula"], "role": "safety"},
          {"pred": "surface_contact", "args": ["pancake", "pan"], "role": "completion"}
        ],
        "pose": [
          {"pred": "orientation_match", "args": ["pancake", "flipped"], "role": "completion"}
        ],
        "direction": [
          {"pred": "motion_dir", "args": [0, 0, 1], "role": "progress"}
        ],
        "safety": [
          {"pred": "no_drop", "args": ["pancake"]}
        ]
      },
      "done_when": "surface_contact(pancake, pan) AND orientation_match(pancake, flipped)"
    }
  ]
}
```

**Training target**:
```
Scene: spatula, spatula_handle, pancake, pan
Step 1: reach(spatula_handle)
  affordance: [0.25, 0.60], approach: [0.00, 0.00, -1.00]
  contact: gripper_state(open) [progress]
  spatial: distance(gripper, spatula_handle, <, 0.03) [completion]
  safety: no_collision(gripper, pan)
  done_when: distance(gripper, spatula_handle) < 0.03
Step 2: grasp(spatula_handle)
  affordance: [0.25, 0.60], approach: [0.00, 0.00, -1.00]
  contact: holding(spatula) [completion]
  direction: grasp_axis([0, 1, 0]) [safety]
  pose: level(spatula, 15) [safety]
  done_when: holding(spatula)
Step 3: push(spatula -> pancake)
  affordance: [0.55, 0.45], approach: [1.00, 0.00, 0.00]
  contact: holding(spatula) [safety]; surface_contact(spatula, pancake) [completion]
  spatial: below(spatula, pancake, 0.01) [completion]
  direction: motion_dir([1, 0, 0]) [progress]
  safety: no_collision(spatula, pan)
  done_when: below(spatula, pancake, 0.01)
Step 4: flip(pancake)
  affordance: [0.55, 0.45], approach: [0.00, 0.00, 1.00]
  contact: holding(spatula) [safety]; surface_contact(pancake, pan) [completion]
  pose: orientation_match(pancake, flipped) [completion]
  direction: motion_dir([0, 0, 1]) [progress]
  safety: no_drop(pancake)
  done_when: surface_contact(pancake, pan) AND orientation_match(pancake, flipped)
```

---

### Example 4: Insertion / Assembly

**Task**: "Insert the USB plug into the port"

```json
{
  "task": "Insert the USB plug into the port",
  "scene_objects": ["usb_plug", "usb_port", "cable"],
  "num_steps": 3,
  "steps": [
    {
      "step": 1,
      "action": "grasp",
      "target": "usb_plug",
      "affordance": [0.40, 0.55],
      "approach": [0.0, 0.0, -1.0],
      "constraints": {
        "contact": [
          {"pred": "holding", "args": ["usb_plug"], "role": "completion"}
        ],
        "pose": [
          {"pred": "level", "args": ["usb_plug", 5], "role": "safety"}
        ],
        "direction": [
          {"pred": "grasp_axis", "args": [0, 1, 0], "role": "safety"}
        ]
      },
      "done_when": "holding(usb_plug)"
    },
    {
      "step": 2,
      "action": "transport",
      "target": "usb_plug",
      "destination": "usb_port",
      "affordance": [0.70, 0.42],
      "approach": [1.0, 0.0, 0.0],
      "constraints": {
        "contact": [
          {"pred": "holding", "args": ["usb_plug"], "role": "safety"}
        ],
        "spatial": [
          {"pred": "aligned_xy", "args": ["usb_plug", "usb_port", 0.005], "role": "completion"},
          {"pred": "aligned_z", "args": ["usb_plug", "usb_port", 0.005], "role": "completion"},
          {"pred": "distance", "args": ["usb_plug", "usb_port", "<", 0.02], "role": "completion"}
        ],
        "pose": [
          {"pred": "orientation_match", "args": ["usb_plug", "usb_port"], "role": "safety"}
        ],
        "safety": [
          {"pred": "no_collision", "args": ["usb_plug", "usb_port"]},
          {"pred": "force_limit", "args": ["usb_plug", 5]}
        ]
      },
      "done_when": "aligned_xy(usb_plug, usb_port, 0.005) AND distance(usb_plug, usb_port) < 0.02"
    },
    {
      "step": 3,
      "action": "insert",
      "target": "usb_plug",
      "destination": "usb_port",
      "affordance": [0.70, 0.42],
      "approach": [1.0, 0.0, 0.0],
      "constraints": {
        "contact": [
          {"pred": "inserted", "args": ["usb_plug", "usb_port"], "role": "completion"}
        ],
        "direction": [
          {"pred": "insert_dir", "args": [1, 0, 0], "role": "safety"}
        ],
        "pose": [
          {"pred": "orientation_match", "args": ["usb_plug", "usb_port"], "role": "safety"}
        ],
        "safety": [
          {"pred": "force_limit", "args": ["usb_plug", 10]}
        ]
      },
      "done_when": "inserted(usb_plug, usb_port)"
    }
  ]
}
```

**Training target**:
```
Scene: usb_plug, usb_port, cable
Step 1: grasp(usb_plug)
  affordance: [0.40, 0.55], approach: [0.00, 0.00, -1.00]
  contact: holding(usb_plug) [completion]
  pose: level(usb_plug, 5) [safety]
  direction: grasp_axis([0, 1, 0]) [safety]
  done_when: holding(usb_plug)
Step 2: transport(usb_plug -> usb_port)
  affordance: [0.70, 0.42], approach: [1.00, 0.00, 0.00]
  contact: holding(usb_plug) [safety]
  spatial: aligned_xy(usb_plug, usb_port, 0.005) [completion]; aligned_z(usb_plug, usb_port, 0.005) [completion]; distance(usb_plug, usb_port, <, 0.02) [completion]
  pose: orientation_match(usb_plug, usb_port) [safety]
  safety: no_collision(usb_plug, usb_port); force_limit(usb_plug, 5)
  done_when: aligned_xy(usb_plug, usb_port, 0.005) AND distance(usb_plug, usb_port) < 0.02
Step 3: insert(usb_plug -> usb_port)
  affordance: [0.70, 0.42], approach: [1.00, 0.00, 0.00]
  contact: inserted(usb_plug, usb_port) [completion]
  direction: insert_dir([1, 0, 0]) [safety]
  pose: orientation_match(usb_plug, usb_port) [safety]
  safety: force_limit(usb_plug, 10)
  done_when: inserted(usb_plug, usb_port)
```

---

### Example 5: Pouring

**Task**: "Pour water from the pitcher into the glass"

```json
{
  "task": "Pour water from the pitcher into the glass",
  "scene_objects": ["pitcher", "pitcher_handle", "glass", "water", "table"],
  "num_steps": 4,
  "steps": [
    {
      "step": 1,
      "action": "grasp",
      "target": "pitcher_handle",
      "affordance": [0.30, 0.45],
      "approach": [1.0, 0.0, 0.0],
      "constraints": {
        "contact": [
          {"pred": "holding", "args": ["pitcher"], "role": "completion"}
        ],
        "pose": [
          {"pred": "upright", "args": ["pitcher", 5], "role": "safety"}
        ],
        "direction": [
          {"pred": "grasp_axis", "args": [0, 0, 1], "role": "safety"}
        ]
      },
      "done_when": "holding(pitcher)"
    },
    {
      "step": 2,
      "action": "transport",
      "target": "pitcher",
      "destination": "glass",
      "affordance": [0.65, 0.40],
      "approach": [0.0, 0.0, -1.0],
      "constraints": {
        "contact": [
          {"pred": "holding", "args": ["pitcher"], "role": "safety"}
        ],
        "spatial": [
          {"pred": "above", "args": ["pitcher", "glass", 0.05], "role": "completion"},
          {"pred": "aligned_xy", "args": ["pitcher", "glass", 0.03], "role": "completion"}
        ],
        "pose": [
          {"pred": "upright", "args": ["pitcher", 10], "role": "safety"}
        ],
        "safety": [
          {"pred": "no_spill", "args": ["water", "pitcher"]},
          {"pred": "no_collision", "args": ["pitcher", "glass"]}
        ]
      },
      "done_when": "above(pitcher, glass, 0.05) AND aligned_xy(pitcher, glass, 0.03)"
    },
    {
      "step": 3,
      "action": "pour",
      "target": "water",
      "destination": "glass",
      "affordance": [0.65, 0.40],
      "approach": [0.0, 0.0, -1.0],
      "constraints": {
        "contact": [
          {"pred": "holding", "args": ["pitcher"], "role": "safety"}
        ],
        "spatial": [
          {"pred": "above", "args": ["pitcher", "glass", 0.02], "role": "safety"},
          {"pred": "aligned_xy", "args": ["pitcher", "glass", 0.03], "role": "safety"}
        ],
        "pose": [
          {"pred": "tilted", "args": ["pitcher", "Y", 70], "role": "progress"}
        ],
        "safety": [
          {"pred": "no_spill", "args": ["water", "glass"]},
          {"pred": "no_drop", "args": ["pitcher"]}
        ]
      },
      "done_when": "inside(water, glass)"
    },
    {
      "step": 4,
      "action": "place",
      "target": "pitcher",
      "destination": "table",
      "affordance": [0.30, 0.55],
      "approach": [0.0, 0.0, -1.0],
      "constraints": {
        "contact": [
          {"pred": "surface_contact", "args": ["pitcher", "table"], "role": "completion"},
          {"pred": "released", "args": ["pitcher"], "role": "completion"}
        ],
        "pose": [
          {"pred": "upright", "args": ["pitcher", 5], "role": "completion"},
          {"pred": "stable", "args": ["pitcher"], "role": "completion"}
        ],
        "safety": [
          {"pred": "no_spill", "args": ["water", "pitcher"]},
          {"pred": "support_stable", "args": ["pitcher"]}
        ]
      },
      "done_when": "surface_contact(pitcher, table) AND upright(pitcher, 5) AND released(pitcher)"
    }
  ]
}
```

---

## 8. Task Category Guidelines

When decomposing a task, first identify its category to guide constraint selection:

| Category | Typical Primitives | Key Constraints | Examples |
|----------|-------------------|-----------------|----------|
| **Pick-Place** | reach, grasp, transport, place | holding, surface_contact, stable | Stack blocks, sort objects |
| **Articulated** | reach, grasp, pull/push/rotate | motion_dir, distance (joint travel) | Open drawer, turn faucet |
| **Tool Use** | grasp (tool), push/wipe/flip | holding(tool), surface_contact(tool, target) | Spatula flip, wipe with cloth |
| **Insertion** | grasp, transport, insert | aligned_xy, orientation_match, force_limit | Peg-in-hole, plug USB |
| **Pouring** | grasp, transport, pour, place | upright, tilted, no_spill, aligned_xy | Pour water, fill bowl |
| **Bimanual** | reach+reach, grasp+grasp, coordinate | holding (both), distance (between hands) | Fold cloth, open jar |

### Constraint selection heuristics

1. **Every stage needs at least one completion constraint** — otherwise the controller can never advance
2. **`holding(obj)` persists as safety from grasp through place/release** — the most common cross-stage constraint
3. **Insertion tasks require tight tolerances** — aligned_xy tolerance < 0.01m, force_limit always present
4. **Pouring tasks need pose tracking** — upright during transport, tilted during pour, upright during place-back
5. **Tool use splits target from tool** — `holding(tool)` is safety throughout; constraints reference both tool and target object
6. **Approach direction matters most for grasp and insert** — top-down for tabletop picks, frontal for drawer handles, axial for insertions

---

## 9. Coordinate Conventions

### Affordance Point [u, v]

Normalized image coordinates:
- `u`: horizontal (0.0 = left edge, 1.0 = right edge)
- `v`: vertical (0.0 = top edge, 1.0 = bottom edge)
- `[0.5, 0.5]` = image center

### Approach Vector [x, y, z]

Unit vector in camera frame:
- `x`: right (+) / left (-)
- `y`: forward (+) / backward (-)
- `z`: up (+) / down (-)

| Direction | Vector | Typical Use |
|-----------|--------|-------------|
| Top-down | `[0, 0, -1]` | Tabletop grasping |
| From right | `[1, 0, 0]` | Side grasping, horizontal insertion |
| From left | `[-1, 0, 0]` | Side grasping |
| From front | `[0, 1, 0]` | Drawer pulling, frontal approach |
| From behind | `[0, -1, 0]` | Rear access |
| Angled | `[0.7, 0, -0.7]` | 45-degree approach |

---

## 10. Runtime Triggering Logic

```python
def verify_step(scene_tokens, step_constraints):
    """Called every control cycle to decide: CONTINUE, NEXT, CORRECT, or REPLAN."""
    results = {}

    # Evaluate all constraint predicates from 3D scene tokens
    for category in ["contact", "spatial", "pose", "direction", "safety"]:
        for c in step_constraints.get(category, []):
            results[(c["pred"], tuple(c["args"]))] = evaluate_predicate(
                scene_tokens, c["pred"], c["args"]
            )

    # Priority 1: Safety — any violation triggers immediate correction
    for c in all_constraints_with_role("safety", step_constraints):
        key = (c["pred"], tuple(c["args"]))
        if not results[key]:
            return Action.CORRECT, generate_correction(c)

    # Priority 2: Completion — all satisfied triggers stage advance
    completion = all_constraints_with_role("completion", step_constraints)
    if completion and all(results[(c["pred"], tuple(c["args"]))] for c in completion):
        return Action.NEXT, None

    # Priority 3: Progress — stagnation triggers replanning
    if progress_stalled(results, history, patience=30):
        return Action.REPLAN, None

    return Action.CONTINUE, None
```
