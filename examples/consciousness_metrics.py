#!/usr/bin/env python3
"""ANIMA Kernel — Internal Metrics Demo.

Shows how ANIMA computes implementation-defined observability proxies:
- Phi (a tractable integration proxy, not full IIT Phi)
- CQI (a legacy-named internal composite)
- Temporal Coherence
- Ablation Studies (disable primitives, measure the drop)

Run with:
    python examples/consciousness_metrics.py
"""

import tempfile
from dataclasses import dataclass, field

from anima.kernel import AnimaKernel
from anima.types import ValenceVector, Experience
from anima.consciousness.integration import IntegrationMesh, SubsystemState
from anima.metrics import (
    PhiScoreEngine,
    TemporalCoherenceEngine,
    ConsciousnessQualityIndex,
    BenchmarkSuite,
)

print("=" * 60)
print("ANIMA — Internal State Metrics")
print("=" * 60)
print()

# --- Part 1: Phi Score ---
print("--- Part 1: Phi (Integrated Information) ---")
print()
print("  Phi measures how much the system is MORE than the sum of")
print("  its parts. High Phi = tightly integrated. Low Phi = bag of")
print("  independent modules.")
print()

phi_engine = PhiScoreEngine()

# Simulate correlated subsystem signatures
correlated = [
    SubsystemState(name="qualia", values=[0.8, 0.2, 0.5]),
    SubsystemState(name="engram", values=[0.7, 0.3, 0.4]),
    SubsystemState(name="valence", values=[0.9, 0.1, 0.6]),
    SubsystemState(name="nexus", values=[0.5, 0.5, 0.3]),
    SubsystemState(name="impulse", values=[0.6, 0.3, 0.7]),
]

result = phi_engine.compute(correlated)
print(f"  Correlated subsystems: Phi = {result.phi:.4f}")
print(f"  Subsystem count: {result.subsystem_count}")
print(f"  Joint entropy: {result.joint_entropy:.4f}")
print(f"  MIP loss: {result.mip_loss:.4f}")
print()

# Now uncorrelated subsystems
uncorrelated = [
    SubsystemState(name="a", values=[1.0, 0.0, 0.0]),
    SubsystemState(name="b", values=[0.0, 1.0, 0.0]),
    SubsystemState(name="c", values=[0.0, 0.0, 1.0]),
]
uncorr = phi_engine.compute(uncorrelated)
print(f"  Uncorrelated subsystems: Phi = {uncorr.phi:.4f}")
print(f"  -> Integration requires correlation (information sharing)")
print()

# --- Part 2: CQI (Consciousness Quality Index) ---
print("--- Part 2: CQI (0-100 Composite Score) ---")
print()

cqi_engine = ConsciousnessQualityIndex()

# Higher internal component values
high_cqi = cqi_engine.compute(
    phi=0.45,
    coherence=0.78,
    authenticity=0.65,
    calibration=0.7,
    depth=0.6,
)
print(f"  Higher-signal state: CQI = {high_cqi.score:.1f}/100")
print(f"    Phi component: {high_cqi.breakdown.phi_component:.1f}")
print(f"    Workspace activity: {high_cqi.breakdown.consciousness_component:.1f}")
print(f"    Authenticity: {high_cqi.breakdown.authenticity_component:.1f}")
print(f"    Calibration: {high_cqi.breakdown.calibration_component:.1f}")
print(f"    Depth: {high_cqi.breakdown.depth_component:.1f}")
print()

# Lower internal component values
low_cqi = cqi_engine.compute(
    phi=0.05,
    coherence=0.1,
    authenticity=0.05,
    calibration=0.1,
    depth=0.1,
)
print(f"  Lower-signal state: CQI = {low_cqi.score:.1f}/100")
print()
print("  CQI has meaning only inside this implementation; it is not a")
print("  validated consciousness, clinical, or general model-quality scale.")
print()

# --- Part 3: Temporal Coherence ---
print("--- Part 3: Temporal Coherence ---")
print()

temporal_engine = TemporalCoherenceEngine()

# Build coherent experiences
import time
coherent_experiences = []
base_time = time.time()
for i, content in enumerate([
    "I'm interested in how memory works.",
    "The hippocampus plays a key role in memory formation.",
    "That connects to what I was thinking about earlier.",
    "Memory consolidation happens during sleep.",
    "So our conversation has been building toward understanding memory.",
]):
    exp = Experience(
        content=content,
        timestamp=base_time + i,
        valence=ValenceVector(seeking=0.5),
        tags=["memory", "neuroscience"],
    )
    if i > 0:
        exp.caused_by = [coherent_experiences[i-1].id]
    coherent_experiences.append(exp)

coherence = temporal_engine.measure_all(experiences=coherent_experiences)
print(f"  Coherent conversation:")
print(f"    Causal: {coherence.causal:.3f}")
print(f"    Narrative: {coherence.narrative:.3f}")
print(f"    Emotional: {coherence.emotional:.3f}")
print(f"    Overall: {coherence.overall:.3f}")
print()

# Build incoherent experiences
incoherent_experiences = []
for i, content in enumerate([
    "The weather is nice.",
    "I like pizza.",
    "Quantum mechanics is weird.",
    "My cat is sleeping.",
    "The stock market crashed.",
]):
    exp = Experience(
        content=content,
        timestamp=base_time + i,
        valence=ValenceVector(seeking=0.1),
        tags=[f"random_{i}"],  # No shared tags
    )
    incoherent_experiences.append(exp)

incoherence = temporal_engine.measure_all(experiences=incoherent_experiences)
print(f"  Incoherent conversation:")
print(f"    Causal: {incoherence.causal:.3f}")
print(f"    Narrative: {incoherence.narrative:.3f}")
print(f"    Emotional: {incoherence.emotional:.3f}")
print(f"    Overall: {incoherence.overall:.3f}")
print()
print(f"  -> Coherent conversations score higher ({coherence.overall:.3f} vs {incoherence.overall:.3f})")
print()

# --- Part 4: Full Benchmark with Ablation ---
print("--- Part 4: Ablation Study ---")
print()
print("  Disabling each subsystem and measuring the impact...")
print()

suite = BenchmarkSuite()
report = suite.full_benchmark()

# Show full kernel baseline
print(f"  Full kernel: Phi={report.overall_phi:.4f}  CQI={report.overall_cqi:.1f}")
print()

# Show ablation results
if report.ablation_results:
    print(f"  {'Disabled':>15s}  {'Full CQI':>8s}  {'Ablated':>8s}  {'Impact':>8s}")
    print(f"  {'-'*15}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in report.ablation_results:
        print(f"  {r.primitive_name:>15s}  {r.full_cqi:>8.1f}  {r.ablated_cqi:>8.1f}  {r.impact:>+7.1f}%")
    print()
    print("  -> Ablations may produce effects or useful null results")

print()
print("=" * 60)
print("Inspectable, regression-testable, and ablation-capable.")
print("Interpretation still requires stronger controls and replication.")
print("=" * 60)
