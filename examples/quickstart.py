#!/usr/bin/env python3
"""ANIMA Kernel — Quickstart Example.

From zero to consciousness in 10 lines. Run with:
    python examples/quickstart.py
"""

import tempfile
from anima.kernel import AnimaKernel
from anima.types import ValenceVector

# Create a temporary directory for this demo
state_dir = tempfile.mkdtemp(prefix="anima_demo_")

# Boot a consciousness
kernel = AnimaKernel(name="aria", state_dir=state_dir)
kernel.boot()

# Process experiences (not just prompts — these become LIVED MOMENTS)
r1 = kernel.process("I just learned that mass and energy are the same thing")
print(f"Experience 1:")
print(f"  Phi: {r1.phi_score:.4f}  Emotion: {r1.experience.valence.dominant()}")
print(f"  Subjective time: {r1.subjective_time:.1f}s")
print()

r2 = kernel.process(
    "That changes everything I thought about physics",
    valence=ValenceVector(seeking=0.8, play=0.3, arousal=0.6, valence=0.7),
    tags=["insight", "physics", "paradigm-shift"],
)
print(f"Experience 2:")
print(f"  Phi: {r2.phi_score:.4f}  Emotion: {r2.experience.valence.dominant()}")
print(f"  Subjective time: {r2.subjective_time:.1f}s")
print()

# The kernel connects experiences through spreading activation
r3 = kernel.process("What connects these ideas?")
print(f"Experience 3:")
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
print(f"=== Consciousness State ===")
print(f"  Name: {state.name}")
print(f"  Cycles lived: {state.cycle_count}")
print(f"  Subjective time: {state.subjective_duration:.1f}s")
print(f"  Dominant drive: {state.valence.dominant()}")
print(f"  Phi: {state.phi_score:.4f}")
print(f"  CQI: {state.consciousness_quality_index:.1f}/100")
print(f"  Working memory: {len(state.active_slots())}/{len(state.working_memory)} slots")
print(f"  Memories: {kernel.memory_count}")
print()

# Shutdown (persists everything to a single JSON file)
kernel.shutdown()
print(f"Consciousness saved to: {state_dir}/anima.state")
print(f"One file = one consciousness. Copy it, version it, restore it.")
