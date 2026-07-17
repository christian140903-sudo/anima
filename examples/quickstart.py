#!/usr/bin/env python3
"""ANIMA Kernel — Quickstart Example.

Create, inspect, and persist cognitive agent state. Run with:
    python examples/quickstart.py
"""

import tempfile
from anima.kernel import AnimaKernel
from anima.types import ValenceVector

# Create a temporary directory for this demo
state_dir = tempfile.mkdtemp(prefix="anima_demo_")

# Boot a stateful kernel
kernel = AnimaKernel(name="aria", state_dir=state_dir)
kernel.boot()

# Process inputs into explicit, inspectable state transitions
r1 = kernel.process("I just learned that mass and energy are the same thing")
print("Input 1:")
print(f"  Phi: {r1.phi_score:.4f}  Emotion: {r1.experience.valence.dominant()}")
print(f"  Modeled time: {r1.subjective_time:.1f}s")
print()

r2 = kernel.process(
    "That changes everything I thought about physics",
    valence=ValenceVector(seeking=0.8, play=0.3, arousal=0.6, valence=0.7),
    tags=["insight", "physics", "paradigm-shift"],
)
print("Input 2:")
print(f"  Phi: {r2.phi_score:.4f}  Emotion: {r2.experience.valence.dominant()}")
print(f"  Modeled time: {r2.subjective_time:.1f}s")
print()

# The kernel connects experiences through spreading activation
r3 = kernel.process("What connects these ideas?")
print("Input 3:")
print(f"  Phi: {r3.phi_score:.4f}  Emotion: {r3.experience.valence.dominant()}")
print()

# Recall by association (not vector search — spreading activation)
memories = kernel.recall(cue="energy", max_results=5)
print(f"Memories recalled by 'energy': {len(memories)}")
for mem in memories:
    print(f"  [{mem.valence.dominant()}] {mem.content[:60]}...")
print()

# Inspect the living state
state = kernel.state
print("=== Kernel State ===")
print(f"  Name: {state.name}")
print(f"  Cycles lived: {state.cycle_count}")
print(f"  Modeled time: {state.subjective_duration:.1f}s")
print(f"  Dominant drive: {state.valence.dominant()}")
print(f"  Phi: {state.phi_score:.4f}")
print(f"  CQI: {state.consciousness_quality_index:.1f}/100")
print(f"  Working memory: {len(state.active_slots())}/{len(state.working_memory)} slots")
print(f"  Memories: {kernel.memory_count}")
print()

# Shutdown (persists everything to a single JSON file)
kernel.shutdown()
print(f"State saved to: {state_dir}/anima.state")
print("The JSON state can be inspected, copied, versioned, and restored.")
