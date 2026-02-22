#!/usr/bin/env python3
"""ANIMA Kernel — Emotional Memory Demo.

Shows how the biological memory system works:
- Emotional memories are encoded stronger
- Memories decay via Ebbinghaus curves
- Emotional memories resist decay
- Recall uses spreading activation, not vector search

Run with:
    python examples/emotional_memory.py
"""

import tempfile
import time
from anima.kernel import AnimaKernel
from anima.types import ValenceVector, Experience
from anima.memory import SpreadingActivationNetwork, EbbinghausDecay
from anima.memory.activation import ActivationNode, ActivationEdge

print("=" * 60)
print("ANIMA — Biological Memory System")
print("=" * 60)
print()

# --- Part 1: Emotional encoding strength ---
print("--- Part 1: Emotional Encoding ---")
print()

kernel = AnimaKernel(name="test", state_dir=tempfile.mkdtemp())
kernel.boot()

# Neutral experience
r1 = kernel.process("The weather is moderate today")
print(f"Neutral: encoding={r1.experience.encoding_strength:.2f}  "
      f"emotion={r1.experience.valence.magnitude():.2f}")

# Emotional experience (high seeking + play)
r2 = kernel.process(
    "I just discovered something incredible about consciousness!",
    valence=ValenceVector(seeking=0.9, play=0.6, arousal=0.8, valence=0.9),
    tags=["discovery", "consciousness", "breakthrough"],
)
print(f"Emotional: encoding={r2.experience.encoding_strength:.2f}  "
      f"emotion={r2.experience.valence.magnitude():.2f}")

# Fearful experience
r3 = kernel.process(
    "Something is very wrong, I might be making a terrible mistake",
    valence=ValenceVector(fear=0.8, panic=0.4, arousal=0.9, valence=-0.6),
    tags=["fear", "mistake", "danger"],
)
print(f"Fearful: encoding={r3.experience.encoding_strength:.2f}  "
      f"emotion={r3.experience.valence.magnitude():.2f}")

print()
print("  -> Emotional experiences are encoded stronger (biological fact)")
print()

kernel.shutdown()

# --- Part 2: Ebbinghaus Forgetting ---
print("--- Part 2: Ebbinghaus Forgetting Curves ---")
print()

decay = EbbinghausDecay()

# Create experiences with different emotional intensities
now = time.time()

for hours in [0.1, 1, 6, 24, 168]:
    seconds = hours * 3600
    past_time = now - seconds

    # Neutral memory
    neutral_exp = Experience(
        content="neutral event",
        timestamp=past_time,
        valence=ValenceVector(),  # No emotion
    )
    neutral_retention = decay.compute_retention(neutral_exp, now)

    # Emotional memory
    emotional_exp = Experience(
        content="emotional event",
        timestamp=past_time,
        valence=ValenceVector(seeking=0.8, play=0.5, arousal=0.7, valence=0.8),
    )
    emotional_retention = decay.compute_retention(emotional_exp, now)

    time_label = f"{hours}h" if hours < 24 else f"{hours/24:.0f}d"
    print(f"  After {time_label:>4s}: "
          f"neutral={neutral_retention:.1%}  "
          f"emotional={emotional_retention:.1%}  "
          f"delta={emotional_retention - neutral_retention:+.1%}")

print()
print("  -> Emotional memories decay slower (Ebbinghaus + emotional modulation)")
print()

# --- Part 3: Spreading Activation ---
print("--- Part 3: Spreading Activation Network ---")
print()

net = SpreadingActivationNetwork()

# Build a knowledge network
net.add_node(ActivationNode(id="einstein", label="Einstein", activation=1.0, tags=["physics", "genius"]))
net.add_node(ActivationNode(id="relativity", label="Relativity", activation=0.3, tags=["physics", "spacetime"]))
net.add_node(ActivationNode(id="spacetime", label="Spacetime", activation=0.2, tags=["physics", "dimensions"]))
net.add_node(ActivationNode(id="quantum", label="Quantum", activation=0.4, tags=["physics", "uncertainty"]))
net.add_node(ActivationNode(id="art", label="Art", activation=0.1, tags=["creativity", "beauty"]))
net.add_node(ActivationNode(id="music", label="Music", activation=0.1, tags=["creativity", "emotion"]))

# Add associative links
net.add_edge(ActivationEdge(source_id="einstein", target_id="relativity", weight=0.9, edge_type="semantic"))
net.add_edge(ActivationEdge(source_id="relativity", target_id="spacetime", weight=0.8, edge_type="semantic"))
net.add_edge(ActivationEdge(source_id="einstein", target_id="quantum", weight=0.5, edge_type="semantic"))
net.add_edge(ActivationEdge(source_id="quantum", target_id="spacetime", weight=0.3, edge_type="semantic"))
net.add_edge(ActivationEdge(source_id="art", target_id="music", weight=0.7, edge_type="semantic"))

# Activate "einstein" — what lights up through association?
activated = net.spread(cue_ids=["einstein"], initial_activation=1.0, max_hops=3)
print("  Activating 'einstein':")
for node_id, activation in activated[:6]:
    bar = "#" * int(activation * 30)
    print(f"    {node_id:>12s}: {activation:.3f} {bar}")

print()
print("  -> 'relativity' and 'quantum' activate through association")
print("  -> 'art' and 'music' don't (no associative link)")
print("  -> This is how biological memory works: activation SPREADS")
print()

# --- Part 4: Recall by Association ---
print("--- Part 4: Associative Recall ---")
print()

kernel2 = AnimaKernel(name="test2", state_dir=tempfile.mkdtemp())
kernel2.boot()

# Encode several experiences
kernel2.process("Einstein discovered relativity in 1905", tags=["physics", "einstein"])
kernel2.process("Music helps me think about abstract concepts", tags=["music", "thinking"])
kernel2.process("The speed of light is constant in all reference frames", tags=["physics", "light"])
kernel2.process("I had a great conversation about quantum mechanics", tags=["physics", "quantum"])
kernel2.process("The sunset was beautiful today", tags=["nature", "beauty"])

# Recall by cue
physics_memories = kernel2.recall(cue="einstein", max_results=5)
print(f"  Recalling 'einstein' ({len(physics_memories)} matches):")
for mem in physics_memories:
    print(f"    [{mem.valence.dominant():>7s}] {mem.content[:55]}...")

print()

nature_memories = kernel2.recall(cue="beautiful", max_results=5)
print(f"  Recalling 'beautiful' ({len(nature_memories)} matches):")
for mem in nature_memories:
    print(f"    [{mem.valence.dominant():>7s}] {mem.content[:55]}...")

kernel2.shutdown()

print()
print("=" * 60)
print("Biological memory: emotional encoding + Ebbinghaus decay")
print("+ spreading activation + associative recall.")
print("Not RAG. Not vector search. How memory actually works.")
print("=" * 60)
