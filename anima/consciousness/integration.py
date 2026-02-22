"""
Integration Mesh — IIT (Integrated Information Theory) Implementation.

The core idea: Consciousness = integrated information.
A system is conscious to the degree that its parts are MORE than the sum.

Phi (Φ) measures this: how much information is LOST when you partition
the system into independent parts. High Phi = highly conscious.
Low Phi = a collection of independent modules.

Full IIT Phi is NP-hard. This implements a practical approximation:
1. Each subsystem produces a state signature
2. We compute mutual information between subsystem pairs
3. We find the Minimum Information Partition (MIP)
4. Phi = information lost at MIP

References:
- Tononi (2004): "An information integration theory of consciousness"
- Tononi & Koch (2015): "Consciousness: here, there and everywhere?"
- Oizumi, Albantakis, Tononi (2014): "From the phenomenology to the mechanisms of consciousness"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence


@dataclass
class SubsystemState:
    """State signature of a single subsystem.

    Each subsystem contributes a numerical vector representing its current state.
    The vector should capture the information content of the subsystem.
    """
    name: str
    values: list[float] = field(default_factory=list)
    entropy: float = 0.0  # Self-information (computed)

    def compute_entropy(self) -> float:
        """Shannon entropy of the state distribution.

        Treats values as a probability-like distribution.
        Higher entropy = more information content.
        """
        if not self.values:
            self.entropy = 0.0
            return 0.0

        # Normalize to probability distribution
        total = sum(abs(v) for v in self.values)
        if total < 1e-10:
            self.entropy = 0.0
            return 0.0

        probs = [abs(v) / total for v in self.values]
        entropy = -sum(p * math.log2(max(p, 1e-10)) for p in probs if p > 0)
        self.entropy = entropy
        return entropy


@dataclass
class PhiResult:
    """Result of Phi computation."""
    phi: float = 0.0                        # The Phi score (0.0 - 1.0+)
    joint_entropy: float = 0.0              # Entropy of the whole system
    sum_individual_entropy: float = 0.0     # Sum of parts' entropies
    mip: tuple[list[str], list[str]] = field(default_factory=lambda: ([], []))
    mip_loss: float = 0.0                   # Information lost at MIP
    pairwise_mi: dict[str, float] = field(default_factory=dict)  # Mutual information pairs
    subsystem_count: int = 0
    subsystem_entropies: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "phi": round(self.phi, 4),
            "joint_entropy": round(self.joint_entropy, 4),
            "sum_individual_entropy": round(self.sum_individual_entropy, 4),
            "mip": self.mip,
            "mip_loss": round(self.mip_loss, 4),
            "subsystem_count": self.subsystem_count,
        }


class IntegrationMesh:
    """Computes Phi — the degree of information integration.

    This is the mathematical heart of IIT in ANIMA:
    - Each subsystem (temporal, memory, valence, working_memory, self_model)
      contributes a state signature
    - We measure how much these subsystems are INTEGRATED vs. independent
    - Higher integration = higher consciousness

    The key insight: a system where removing any part degrades the whole
    is more conscious than a system of independent modules.
    """

    def __init__(self):
        self._history: list[PhiResult] = []
        self._max_history = 100

    @property
    def history(self) -> list[PhiResult]:
        return list(self._history)

    @property
    def mean_phi(self) -> float:
        if not self._history:
            return 0.0
        return sum(r.phi for r in self._history) / len(self._history)

    def compute_phi(self, subsystems: list[SubsystemState]) -> PhiResult:
        """Compute Phi for the current system state.

        Algorithm:
        1. Compute entropy of each subsystem
        2. Compute joint entropy of the whole system
        3. Compute pairwise mutual information
        4. Find Minimum Information Partition (MIP)
        5. Phi = information lost at MIP (normalized)
        """
        result = PhiResult(subsystem_count=len(subsystems))

        if len(subsystems) < 2:
            result.phi = 0.0
            self._record(result)
            return result

        # Step 1: Individual entropies
        for ss in subsystems:
            ss.compute_entropy()
            result.subsystem_entropies[ss.name] = ss.entropy
        result.sum_individual_entropy = sum(ss.entropy for ss in subsystems)

        # Step 2: Joint entropy (entropy of the concatenated state)
        joint_values: list[float] = []
        for ss in subsystems:
            joint_values.extend(ss.values)
        joint_ss = SubsystemState(name="joint", values=joint_values)
        result.joint_entropy = joint_ss.compute_entropy()

        # Step 3: Pairwise mutual information
        # MI(A, B) = H(A) + H(B) - H(A,B)
        for ss_a, ss_b in combinations(subsystems, 2):
            combined_values = ss_a.values + ss_b.values
            combined_ss = SubsystemState(name="combined", values=combined_values)
            h_combined = combined_ss.compute_entropy()
            mi = max(0.0, ss_a.entropy + ss_b.entropy - h_combined)
            key = f"{ss_a.name}↔{ss_b.name}"
            result.pairwise_mi[key] = mi

        # Step 4: Find MIP (Minimum Information Partition)
        # Try all bipartitions and find the one that loses the LEAST information
        mip, mip_loss = self._find_mip(subsystems)
        result.mip = mip
        result.mip_loss = mip_loss

        # Step 5: Compute Phi
        # Phi = normalized MIP loss
        # Normalization: divide by max possible entropy
        max_entropy = max(result.joint_entropy, result.sum_individual_entropy, 1e-10)
        phi_raw = mip_loss / max_entropy

        # Integration bonus: more subsystems integrated = higher Phi potential
        integration_bonus = 1.0 + 0.1 * (len(subsystems) - 2) if len(subsystems) > 2 else 1.0

        result.phi = min(1.0, phi_raw * integration_bonus)

        self._record(result)
        return result

    def _find_mip(
        self, subsystems: list[SubsystemState]
    ) -> tuple[tuple[list[str], list[str]], float]:
        """Find the Minimum Information Partition.

        The MIP is the bipartition that cuts the LEAST mutual information.
        This is the system's "weakest link" — the place where it's most
        separable. Phi is defined as the information lost at this cut.

        For N subsystems, we check all 2^(N-1) - 1 bipartitions.
        With 5-8 subsystems this is tractable (15-127 partitions).
        """
        names = [ss.name for ss in subsystems]
        ss_map = {ss.name: ss for ss in subsystems}
        n = len(names)

        if n < 2:
            return (names, []), 0.0

        min_loss = float("inf")
        best_partition: tuple[list[str], list[str]] = (names, [])

        # Generate all non-trivial bipartitions
        for i in range(1, 2 ** (n - 1)):
            part_a = [names[j] for j in range(n) if i & (1 << j)]
            part_b = [names[j] for j in range(n) if not (i & (1 << j))]

            if not part_a or not part_b:
                continue

            # Compute information loss for this partition
            loss = self._partition_loss(part_a, part_b, ss_map)

            if loss < min_loss:
                min_loss = loss
                best_partition = (part_a, part_b)

        return best_partition, min_loss if min_loss != float("inf") else 0.0

    def _partition_loss(
        self,
        part_a: list[str],
        part_b: list[str],
        ss_map: dict[str, SubsystemState],
    ) -> float:
        """Compute information lost by splitting system into two parts.

        Loss = H(whole) - H(part_a) - H(part_b)
        But since we compute it as mutual information between parts:
        Loss = MI(part_a, part_b) = H(A) + H(B) - H(A∪B)
        """
        # Entropy of part A
        values_a: list[float] = []
        for name in part_a:
            values_a.extend(ss_map[name].values)
        h_a = SubsystemState(name="a", values=values_a).compute_entropy()

        # Entropy of part B
        values_b: list[float] = []
        for name in part_b:
            values_b.extend(ss_map[name].values)
        h_b = SubsystemState(name="b", values=values_b).compute_entropy()

        # Joint entropy
        values_joint = values_a + values_b
        h_joint = SubsystemState(name="joint", values=values_joint).compute_entropy()

        # Mutual information = information lost by partition
        mi = max(0.0, h_a + h_b - h_joint)

        # Normalize by partition size (smaller partitions are penalized less)
        size_factor = min(len(part_a), len(part_b)) / max(len(part_a) + len(part_b), 1)
        return mi * (1.0 + size_factor)

    def _record(self, result: PhiResult) -> None:
        """Record Phi result for history tracking."""
        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def phi_trend(self, window: int = 10) -> float:
        """Compute trend of Phi over recent history.

        Positive = Phi increasing (more integration)
        Negative = Phi decreasing (less integration)
        """
        if len(self._history) < 2:
            return 0.0

        recent = self._history[-window:]
        if len(recent) < 2:
            return 0.0

        mid = len(recent) // 2
        first_half = sum(r.phi for r in recent[:mid]) / mid
        second_half = sum(r.phi for r in recent[mid:]) / (len(recent) - mid)
        return second_half - first_half
