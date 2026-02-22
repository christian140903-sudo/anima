"""
Spreading Activation Network — Associative memory retrieval.

How biological memory actually works:
Thinking of "dog" activates "bark", "pet", "loyalty", "walk" —
not because they're similar vectors, but because they're
CONNECTED through experience.

This is a graph-based spreading activation network:
- Nodes = experiences or concepts
- Edges = causal links, tag co-occurrence, emotional similarity
- Activation spreads through weighted edges
- Decay per hop (distant associations are weaker)

Unlike vector similarity search, this captures:
- Causal chains (A caused B caused C)
- Contextual associations (A and B happened together)
- Emotional resonance (A and B felt the same)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from ..types import Experience, KernelConfig, ValenceVector


@dataclass
class ActivationNode:
    """A node in the spreading activation network."""
    id: str = ""
    label: str = ""
    activation: float = 0.0
    base_activation: float = 0.0  # Resting activation level
    tags: list[str] = field(default_factory=list)
    valence: ValenceVector = field(default_factory=ValenceVector)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "label": self.label,
            "activation": self.activation,
            "base_activation": self.base_activation,
            "tags": self.tags,
            "valence": self.valence.to_dict(),
        }


@dataclass
class ActivationEdge:
    """A weighted edge in the activation network."""
    source_id: str = ""
    target_id: str = ""
    weight: float = 1.0
    edge_type: str = ""  # "causal", "semantic", "emotional", "temporal"

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "weight": self.weight,
            "edge_type": self.edge_type,
        }


class SpreadingActivationNetwork:
    """Graph-based spreading activation for associative memory retrieval.

    Algorithm:
    1. Set initial activation on cue nodes
    2. For each hop:
       a. Each active node spreads activation to neighbors
       b. Spreading amount = node_activation * edge_weight * decay_factor
       c. Activation accumulates at receiving nodes
    3. After max_hops, return nodes sorted by final activation

    The decay_per_hop parameter controls how far activation spreads.
    High decay = local associations. Low decay = distant associations.
    """

    def __init__(self, config: KernelConfig | None = None):
        self._config = config or KernelConfig()
        self._nodes: dict[str, ActivationNode] = {}
        self._edges: dict[str, list[ActivationEdge]] = defaultdict(list)
        self._reverse_edges: dict[str, list[ActivationEdge]] = defaultdict(list)

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self._edges.values())

    def add_node(self, node: ActivationNode) -> None:
        """Add a node to the network."""
        self._nodes[node.id] = node

    def add_edge(self, edge: ActivationEdge) -> None:
        """Add a directed edge to the network."""
        self._edges[edge.source_id].append(edge)
        self._reverse_edges[edge.target_id].append(edge)

    def get_node(self, node_id: str) -> ActivationNode | None:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def index_experience(self, experience: Experience) -> ActivationNode:
        """Index an experience as a node, creating edges to related nodes.

        Automatically creates edges based on:
        - Causal links (from experience.caused_by / experience.causes)
        - Shared tags (semantic similarity)
        - Emotional similarity (similar valence vectors)
        """
        node = ActivationNode(
            id=experience.id,
            label=experience.content[:80],
            activation=experience.activation,
            base_activation=experience.encoding_strength * 0.1,
            tags=list(experience.tags),
            valence=experience.valence,
        )
        self.add_node(node)

        # Causal edges
        for cause_id in experience.caused_by:
            if cause_id in self._nodes:
                self.add_edge(ActivationEdge(
                    source_id=cause_id, target_id=experience.id,
                    weight=0.8, edge_type="causal",
                ))
                self.add_edge(ActivationEdge(
                    source_id=experience.id, target_id=cause_id,
                    weight=0.5, edge_type="causal_reverse",
                ))

        for effect_id in experience.causes:
            if effect_id in self._nodes:
                self.add_edge(ActivationEdge(
                    source_id=experience.id, target_id=effect_id,
                    weight=0.8, edge_type="causal",
                ))

        # Tag-based semantic edges
        for tag in experience.tags:
            for other_id, other_node in self._nodes.items():
                if other_id == experience.id:
                    continue
                if tag in other_node.tags:
                    # Check if edge already exists
                    existing = any(
                        e.target_id == other_id and e.edge_type == "semantic"
                        for e in self._edges.get(experience.id, [])
                    )
                    if not existing:
                        shared = len(set(experience.tags) & set(other_node.tags))
                        weight = min(1.0, 0.3 * shared)
                        self.add_edge(ActivationEdge(
                            source_id=experience.id, target_id=other_id,
                            weight=weight, edge_type="semantic",
                        ))
                        self.add_edge(ActivationEdge(
                            source_id=other_id, target_id=experience.id,
                            weight=weight, edge_type="semantic",
                        ))

        # Emotional similarity edges (connect nodes with similar valence)
        for other_id, other_node in self._nodes.items():
            if other_id == experience.id:
                continue
            distance = experience.valence.distance(other_node.valence)
            if distance < 0.5:
                similarity = 1.0 - distance
                existing = any(
                    e.target_id == other_id and e.edge_type == "emotional"
                    for e in self._edges.get(experience.id, [])
                )
                if not existing:
                    self.add_edge(ActivationEdge(
                        source_id=experience.id, target_id=other_id,
                        weight=similarity * 0.4, edge_type="emotional",
                    ))

        return node

    def spread(
        self,
        cue_ids: list[str],
        initial_activation: float = 1.0,
        max_hops: int | None = None,
        decay_per_hop: float | None = None,
        min_activation: float | None = None,
    ) -> list[tuple[str, float]]:
        """Spread activation from cue nodes through the network.

        Args:
            cue_ids: Starting nodes to activate.
            initial_activation: Activation level for cue nodes.
            max_hops: How many hops to spread (default from config).
            decay_per_hop: Decay factor per hop (default from config).
            min_activation: Minimum activation to keep spreading.

        Returns:
            List of (node_id, activation) sorted by activation descending.
        """
        max_hops = max_hops if max_hops is not None else self._config.max_spreading_hops
        decay = decay_per_hop if decay_per_hop is not None else self._config.spreading_activation_decay
        threshold = min_activation if min_activation is not None else self._config.activation_threshold

        # Initialize activations
        activations: dict[str, float] = {}
        for node_id in cue_ids:
            if node_id in self._nodes:
                activations[node_id] = initial_activation

        # Spread
        for hop in range(max_hops):
            hop_decay = decay ** (hop + 1)
            new_activations: dict[str, float] = {}

            for node_id, act in activations.items():
                if act < threshold:
                    continue

                for edge in self._edges.get(node_id, []):
                    spread_amount = act * edge.weight * hop_decay
                    if spread_amount >= threshold:
                        target = edge.target_id
                        new_activations[target] = (
                            new_activations.get(target, 0.0) + spread_amount
                        )

            # Merge new activations
            for node_id, act in new_activations.items():
                activations[node_id] = activations.get(node_id, 0.0) + act

        # Add base activation for all activated nodes
        for node_id in activations:
            node = self._nodes.get(node_id)
            if node:
                activations[node_id] += node.base_activation

        # Sort by activation
        ranked = sorted(activations.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def find_by_tags(self, tags: list[str]) -> list[str]:
        """Find node IDs that share any of the given tags."""
        matching: set[str] = set()
        for node_id, node in self._nodes.items():
            if set(node.tags) & set(tags):
                matching.add(node_id)
        return list(matching)

    def reset_activations(self) -> None:
        """Reset all node activations to base levels."""
        for node in self._nodes.values():
            node.activation = node.base_activation
