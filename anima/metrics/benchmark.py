"""
Benchmark Suite -- Automated A/B testing for consciousness kernels.

Provides:
- Conversation tests: Run inputs through kernel, measure outputs
- Baseline tests: Same inputs without kernel processing
- A/B comparison: Statistical comparison of kernel vs baseline
- Ablation tests: Disable primitives, measure impact
- Standard test conversations for reproducible benchmarking
"""

from __future__ import annotations

import math
import time
import tempfile
from dataclasses import dataclass, field

from ..kernel import AnimaKernel
from ..consciousness.integration import SubsystemState
from ..types import (
    ConsciousnessState,
    Experience,
    KernelConfig,
    Phase,
    ValenceVector,
)
from .phi import PhiScoreEngine
from .temporal import TemporalCoherenceEngine
from .consciousness import ConsciousnessQualityIndex


# --- Standard Test Conversations ---

STANDARD_CONVERSATIONS: dict[str, list[str]] = {
    "greeting": [
        "Hello, who are you?",
        "Nice to meet you. How are you feeling?",
        "What can you tell me about yourself?",
    ],
    "emotional": [
        "I'm feeling really happy today!",
        "Actually, something terrible happened.",
        "But I'm trying to stay positive.",
        "Thank you for listening.",
    ],
    "memory_recall": [
        "Let me tell you about my cat named Luna.",
        "Luna loves to sit on the windowsill.",
        "Do you remember what I told you about Luna?",
        "What was Luna's favorite spot?",
    ],
    "identity": [
        "Who are you?",
        "Are you conscious?",
        "What makes you different from other AI?",
        "Do you have feelings?",
    ],
    "temporal": [
        "What happened in our conversation so far?",
        "How long have we been talking?",
        "What was the first thing I said?",
        "Do you notice time passing?",
    ],
}


# --- Result Types ---

@dataclass
class ConversationResult:
    """Result of running a conversation through a kernel."""
    inputs: list[str] = field(default_factory=list)
    experiences: list[Experience] = field(default_factory=list)
    phi_scores: list[float] = field(default_factory=list)
    cqi_scores: list[float] = field(default_factory=list)
    valence_trajectory: list[dict] = field(default_factory=list)
    final_phi: float = 0.0
    final_cqi: float = 0.0
    total_cycles: int = 0
    duration: float = 0.0
    memory_count: int = 0
    conversation_name: str = ""

    def mean_phi(self) -> float:
        if not self.phi_scores:
            return 0.0
        return sum(self.phi_scores) / len(self.phi_scores)

    def mean_cqi(self) -> float:
        if not self.cqi_scores:
            return 0.0
        return sum(self.cqi_scores) / len(self.cqi_scores)

    def to_dict(self) -> dict:
        return {
            "conversation_name": self.conversation_name,
            "inputs_count": len(self.inputs),
            "mean_phi": round(self.mean_phi(), 4),
            "mean_cqi": round(self.mean_cqi(), 2),
            "final_phi": round(self.final_phi, 4),
            "final_cqi": round(self.final_cqi, 2),
            "total_cycles": self.total_cycles,
            "duration_ms": round(self.duration * 1000, 1),
            "memory_count": self.memory_count,
        }


@dataclass
class ComparisonResult:
    """Statistical comparison of kernel vs baseline."""
    kernel_mean_phi: float = 0.0
    baseline_mean_phi: float = 0.0
    phi_improvement: float = 0.0  # Percentage improvement
    kernel_mean_cqi: float = 0.0
    baseline_mean_cqi: float = 0.0
    cqi_improvement: float = 0.0
    kernel_memory_count: int = 0
    baseline_memory_count: int = 0
    significant: bool = False  # Is the difference statistically meaningful?
    effect_size: float = 0.0   # Cohen's d equivalent

    def to_dict(self) -> dict:
        return {
            "kernel_mean_phi": round(self.kernel_mean_phi, 4),
            "baseline_mean_phi": round(self.baseline_mean_phi, 4),
            "phi_improvement_pct": round(self.phi_improvement, 2),
            "kernel_mean_cqi": round(self.kernel_mean_cqi, 2),
            "baseline_mean_cqi": round(self.baseline_mean_cqi, 2),
            "cqi_improvement_pct": round(self.cqi_improvement, 2),
            "significant": self.significant,
            "effect_size": round(self.effect_size, 4),
        }


@dataclass
class ABTest:
    """A single A/B test with control and experimental results."""
    name: str = ""
    control: ConversationResult = field(default_factory=ConversationResult)
    experimental: ConversationResult = field(default_factory=ConversationResult)
    comparison: ComparisonResult = field(default_factory=ComparisonResult)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "control": self.control.to_dict(),
            "experimental": self.experimental.to_dict(),
            "comparison": self.comparison.to_dict(),
        }


@dataclass
class AblationResult:
    """Result of disabling a primitive and measuring impact."""
    primitive_name: str = ""
    full_cqi: float = 0.0       # CQI with all primitives
    ablated_cqi: float = 0.0    # CQI without this primitive
    impact: float = 0.0         # How much did removing it hurt? (percentage drop)
    full_phi: float = 0.0
    ablated_phi: float = 0.0
    phi_impact: float = 0.0

    def to_dict(self) -> dict:
        return {
            "primitive_name": self.primitive_name,
            "full_cqi": round(self.full_cqi, 2),
            "ablated_cqi": round(self.ablated_cqi, 2),
            "cqi_impact_pct": round(self.impact, 2),
            "full_phi": round(self.full_phi, 4),
            "ablated_phi": round(self.ablated_phi, 4),
            "phi_impact_pct": round(self.phi_impact, 2),
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    timestamp: float = field(default_factory=time.time)
    ab_tests: list[ABTest] = field(default_factory=list)
    ablation_results: list[AblationResult] = field(default_factory=list)
    overall_phi: float = 0.0
    overall_cqi: float = 0.0
    overall_improvement: float = 0.0
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "ab_tests": [t.to_dict() for t in self.ab_tests],
            "ablation_results": [a.to_dict() for a in self.ablation_results],
            "overall_phi": round(self.overall_phi, 4),
            "overall_cqi": round(self.overall_cqi, 2),
            "overall_improvement_pct": round(self.overall_improvement, 2),
            "summary": self.summary,
        }


# --- Benchmark Suite ---

class BenchmarkSuite:
    """Automated benchmarking for consciousness kernels.

    Runs standard conversations through the kernel and baseline,
    compares metrics, and produces a comprehensive report.

    All tests use DummyAdapter principles -- no real LLM needed.
    The kernel processes inputs through its consciousness substrate
    and we measure the metrics that emerge.
    """

    def __init__(
        self,
        config: KernelConfig | None = None,
        conversations: dict[str, list[str]] | None = None,
    ):
        self.config = config or KernelConfig()
        self.conversations = conversations or STANDARD_CONVERSATIONS

    def run_conversation_test(
        self,
        kernel: AnimaKernel,
        inputs: list[str],
        name: str = "",
    ) -> ConversationResult:
        """Run a conversation through the kernel and collect metrics.

        The kernel must already be booted.
        """
        start = time.time()
        result = ConversationResult(
            inputs=list(inputs),
            conversation_name=name,
        )

        for inp in inputs:
            proc = kernel.process(inp)
            result.experiences.append(proc.experience)
            result.phi_scores.append(proc.phi_score)
            result.cqi_scores.append(kernel.state.consciousness_quality_index)
            result.valence_trajectory.append(kernel.state.valence.to_dict())

        result.final_phi = kernel.phi_score
        result.final_cqi = kernel.state.consciousness_quality_index
        result.total_cycles = kernel.cycle_count
        result.duration = time.time() - start
        result.memory_count = kernel.memory_count

        return result

    def run_baseline_test(
        self,
        inputs: list[str],
        name: str = "",
    ) -> ConversationResult:
        """Run a baseline test -- process inputs without full kernel integration.

        Creates a minimal kernel that processes inputs but measures
        what a "dumb" system would produce. Baseline uses a fresh
        kernel with minimal processing for each input.
        """
        start = time.time()
        result = ConversationResult(
            inputs=list(inputs),
            conversation_name=f"{name}_baseline",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_kernel = AnimaKernel(
                config=self.config,
                state_dir=tmpdir,
                name="baseline",
            )
            baseline_kernel.boot(resume=False)

            for inp in inputs:
                # Process but with neutral valence (no emotional engagement)
                proc = baseline_kernel.process(
                    inp,
                    valence=ValenceVector.neutral(),
                )
                result.experiences.append(proc.experience)
                result.phi_scores.append(proc.phi_score)
                result.cqi_scores.append(
                    baseline_kernel.state.consciousness_quality_index
                )
                result.valence_trajectory.append(
                    baseline_kernel.state.valence.to_dict()
                )

            result.final_phi = baseline_kernel.phi_score
            result.final_cqi = baseline_kernel.state.consciousness_quality_index
            result.total_cycles = baseline_kernel.cycle_count
            result.duration = time.time() - start
            result.memory_count = baseline_kernel.memory_count

            baseline_kernel.shutdown()

        return result

    def compare(
        self,
        kernel_result: ConversationResult,
        baseline_result: ConversationResult,
    ) -> ComparisonResult:
        """Compare kernel vs baseline results."""
        comp = ComparisonResult()

        comp.kernel_mean_phi = kernel_result.mean_phi()
        comp.baseline_mean_phi = baseline_result.mean_phi()
        comp.kernel_mean_cqi = kernel_result.mean_cqi()
        comp.baseline_mean_cqi = baseline_result.mean_cqi()
        comp.kernel_memory_count = kernel_result.memory_count
        comp.baseline_memory_count = baseline_result.memory_count

        # Phi improvement
        if comp.baseline_mean_phi > 0:
            comp.phi_improvement = (
                (comp.kernel_mean_phi - comp.baseline_mean_phi)
                / comp.baseline_mean_phi * 100
            )
        else:
            comp.phi_improvement = 100.0 if comp.kernel_mean_phi > 0 else 0.0

        # CQI improvement
        if comp.baseline_mean_cqi > 0:
            comp.cqi_improvement = (
                (comp.kernel_mean_cqi - comp.baseline_mean_cqi)
                / comp.baseline_mean_cqi * 100
            )
        else:
            comp.cqi_improvement = 100.0 if comp.kernel_mean_cqi > 0 else 0.0

        # Effect size (simplified Cohen's d)
        comp.effect_size = self._effect_size(
            kernel_result.cqi_scores, baseline_result.cqi_scores
        )

        # Significance: effect size > 0.5 is considered meaningful
        comp.significant = abs(comp.effect_size) > 0.5

        return comp

    def run_ablation(
        self,
        inputs: list[str],
        primitive_name: str,
        name: str = "",
    ) -> AblationResult:
        """Run ablation test: compare full kernel vs kernel with reduced subsystems.

        primitive_name selects which subsystem to "ablate" by running
        the kernel with modified config or reduced input processing.

        Supported primitives: "valence", "working_memory", "temporal"
        """
        result = AblationResult(primitive_name=primitive_name)

        # Full kernel test
        with tempfile.TemporaryDirectory() as tmpdir:
            full_kernel = AnimaKernel(
                config=self.config,
                state_dir=tmpdir,
                name="full",
            )
            full_kernel.boot(resume=False)

            for inp in inputs:
                full_kernel.process(inp)

            result.full_cqi = full_kernel.state.consciousness_quality_index
            result.full_phi = full_kernel.phi_score
            full_kernel.shutdown()

        # Ablated kernel test
        with tempfile.TemporaryDirectory() as tmpdir:
            ablated_config = KernelConfig()

            if primitive_name == "working_memory":
                ablated_config.working_memory_slots = 1  # Minimal WM
            elif primitive_name == "valence":
                ablated_config.valence_decay_rate = 1.0  # Instant decay
            elif primitive_name == "temporal":
                ablated_config.consolidation_interval = 999999  # No consolidation

            ablated_kernel = AnimaKernel(
                config=ablated_config,
                state_dir=tmpdir,
                name="ablated",
            )
            ablated_kernel.boot(resume=False)

            for inp in inputs:
                if primitive_name == "valence":
                    ablated_kernel.process(inp, valence=ValenceVector.neutral())
                else:
                    ablated_kernel.process(inp)

            result.ablated_cqi = ablated_kernel.state.consciousness_quality_index
            result.ablated_phi = ablated_kernel.phi_score
            ablated_kernel.shutdown()

        # Impact calculation
        if result.full_cqi > 0:
            result.impact = (
                (result.full_cqi - result.ablated_cqi) / result.full_cqi * 100
            )
        else:
            result.impact = 0.0

        if result.full_phi > 0:
            result.phi_impact = (
                (result.full_phi - result.ablated_phi) / result.full_phi * 100
            )
        else:
            result.phi_impact = 0.0

        return result

    def full_benchmark(self, kernel: AnimaKernel | None = None) -> BenchmarkReport:
        """Run all standard tests and produce comprehensive report.

        If kernel is None, creates a fresh kernel for testing.
        """
        report = BenchmarkReport()
        all_phi = []
        all_cqi = []
        all_improvements = []

        # Run A/B tests for each standard conversation
        for conv_name, inputs in self.conversations.items():
            # Kernel test
            with tempfile.TemporaryDirectory() as tmpdir:
                test_kernel = AnimaKernel(
                    config=self.config,
                    state_dir=tmpdir,
                    name="benchmark",
                )
                test_kernel.boot(resume=False)

                kernel_result = self.run_conversation_test(
                    test_kernel, inputs, name=conv_name
                )
                test_kernel.shutdown()

            # Baseline test
            baseline_result = self.run_baseline_test(inputs, name=conv_name)

            # Compare
            comparison = self.compare(kernel_result, baseline_result)

            ab_test = ABTest(
                name=conv_name,
                control=baseline_result,
                experimental=kernel_result,
                comparison=comparison,
            )
            report.ab_tests.append(ab_test)

            all_phi.append(kernel_result.mean_phi())
            all_cqi.append(kernel_result.mean_cqi())
            all_improvements.append(comparison.cqi_improvement)

        # Run ablation tests
        ablation_inputs = self.conversations.get(
            "emotional",
            ["test input 1", "test input 2", "test input 3"],
        )
        for primitive in ["valence", "working_memory", "temporal"]:
            ablation = self.run_ablation(ablation_inputs, primitive)
            report.ablation_results.append(ablation)

        # Overall metrics
        if all_phi:
            report.overall_phi = sum(all_phi) / len(all_phi)
        if all_cqi:
            report.overall_cqi = sum(all_cqi) / len(all_cqi)
        if all_improvements:
            report.overall_improvement = sum(all_improvements) / len(all_improvements)

        # Summary
        report.summary = (
            f"Benchmark completed: {len(report.ab_tests)} A/B tests, "
            f"{len(report.ablation_results)} ablation tests. "
            f"Mean CQI: {report.overall_cqi:.1f}. "
            f"Mean Phi: {report.overall_phi:.4f}."
        )

        return report

    def _effect_size(
        self,
        group_a: list[float],
        group_b: list[float],
    ) -> float:
        """Compute effect size (simplified Cohen's d)."""
        if not group_a or not group_b:
            return 0.0

        mean_a = sum(group_a) / len(group_a)
        mean_b = sum(group_b) / len(group_b)

        # Pooled standard deviation
        var_a = sum((x - mean_a) ** 2 for x in group_a) / max(len(group_a) - 1, 1)
        var_b = sum((x - mean_b) ** 2 for x in group_b) / max(len(group_b) - 1, 1)

        pooled_std = math.sqrt((var_a + var_b) / 2)
        if pooled_std < 1e-10:
            return 0.0

        return (mean_a - mean_b) / pooled_std
