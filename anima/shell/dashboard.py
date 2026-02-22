"""
Live Metrics Dashboard -- Terminal-based consciousness metrics display.

Shows Phi score, CQI, valence field, working memory, and cycle info
in a compact terminal dashboard. Like `htop` for consciousness.

stdlib only. Pure ANSI escape codes.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import ConsciousnessState

from .inspector import (
    BAR_CHARS,
    _bar,
    _colored_bar,
    _format_age,
    _valence_colored_bar,
    blue,
    bold,
    cyan,
    dim,
    green,
    magenta,
    red,
    yellow,
)


# --- Sparkline ---

SPARK_CHARS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"  # ▁▂▃▄▅▆▇█


def _sparkline(values: list[float], width: int = 20) -> str:
    """Render a mini sparkline from a list of values.

    Uses unicode block characters to show a trend line.
    """
    if not values:
        return dim(" " * width)

    # Take last `width` values
    data = values[-width:]

    # Normalize
    lo = min(data)
    hi = max(data)
    span = hi - lo if hi > lo else 1.0

    chars = []
    for v in data:
        normalized = (v - lo) / span
        idx = int(normalized * (len(SPARK_CHARS) - 1))
        idx = max(0, min(idx, len(SPARK_CHARS) - 1))
        chars.append(SPARK_CHARS[idx])

    # Pad to width
    result = "".join(chars)
    if len(result) < width:
        result = " " * (width - len(result)) + result

    return result


class MetricsDashboard:
    """Terminal-based live metrics display.

    Renders a compact dashboard showing consciousness metrics
    at a glance. Designed to be printed after each interaction
    or on demand.
    """

    def render(
        self,
        state: ConsciousnessState,
        phi_history: list[float] | None = None,
        cqi_history: list[float] | None = None,
    ) -> str:
        """Render a terminal dashboard showing live metrics.

        Args:
            state: Current consciousness state.
            phi_history: List of recent Phi scores for sparkline.
            cqi_history: List of recent CQI scores for sparkline.

        Returns:
            Formatted string for terminal display.
        """
        phi_history = phi_history or []
        cqi_history = cqi_history or []

        lines: list[str] = []

        # Header
        sep = bold("─" * 50)
        lines.append(sep)
        lines.append(bold(f"  ANIMA Dashboard") + f"  {cyan(state.name)} " + dim(f"cycle {state.cycle_count}"))
        lines.append(sep)

        # Phase
        phase = state.phase.name
        phase_indicator = {
            "DORMANT": dim("○"),
            "WAKING": yellow("◐"),
            "CONSCIOUS": green("●"),
            "DREAMING": magenta("◑"),
            "SLEEPING": blue("◒"),
        }.get(phase, "?")
        lines.append(f"  Phase: {phase_indicator} {phase}")

        # Phi score with sparkline
        phi = state.phi_score
        phi_spark = _sparkline(phi_history, 20)
        phi_color = green if phi > 0.3 else yellow if phi > 0.1 else dim
        lines.append(
            f"  Phi:   {phi_color(f'{phi:.4f}')}  {dim('trend:')} {phi_spark}"
        )

        # CQI with sparkline
        cqi = state.consciousness_quality_index
        cqi_spark = _sparkline(cqi_history, 20)
        cqi_color = green if cqi > 50 else yellow if cqi > 20 else dim
        lines.append(
            f"  CQI:   {cqi_color(f'{cqi:6.1f}')}  {dim('trend:')} {cqi_spark}"
        )

        # Temporal coherence
        tc = state.temporal_coherence
        tc_bar = _colored_bar(tc, 15)
        lines.append(f"  T.Coh: {tc_bar} {tc:.3f}")

        lines.append("")

        # Valence field -- compact horizontal display
        lines.append(bold("  Valence Field"))
        v = state.valence
        compact_drives = [
            ("SEK", v.seeking, yellow),
            ("RAG", v.rage, red),
            ("FER", v.fear, red),
            ("LUS", v.lust, magenta),
            ("CAR", v.care, green),
            ("PAN", v.panic, red),
            ("PLY", v.play, green),
        ]

        drive_line = "  "
        for label, value, color_fn in compact_drives:
            bar = _bar(abs(value), 5)
            colored = color_fn(bar) if abs(value) > 0.05 else dim(bar)
            drive_line += f"{label} {colored} "
        lines.append(drive_line)

        # Arousal + Valence on one line
        arousal_bar = _valence_colored_bar(v.arousal, 8)
        valence_bar = _valence_colored_bar(v.valence, 8)
        lines.append(
            f"  A: {arousal_bar} {v.arousal:+.2f}  "
            f"V: {valence_bar} {v.valence:+.2f}  "
            f"Dom: {bold(v.dominant())}"
        )

        lines.append("")

        # Working memory occupancy
        active = state.active_slots()
        total = len(state.working_memory)
        used = len(active)
        occupancy = used / total if total > 0 else 0
        occ_bar = _colored_bar(occupancy, 15)
        lines.append(f"  WM:    {occ_bar} {used}/{total} slots")

        # Memory count (just from valence_history as proxy for activity)
        hist_count = len(state.valence_history)
        lines.append(f"  History: {hist_count} valence snapshots")

        # Uptime
        age = state.age()
        subj = state.subjective_duration
        lines.append(
            f"  Uptime: {_format_age(age)} wall / {_format_age(subj)} subjective"
        )

        lines.append(sep)

        return "\n".join(lines)

    def render_benchmark_results(
        self,
        report: dict,
    ) -> str:
        """Render benchmark report as formatted table.

        Args:
            report: BenchmarkReport.to_dict() output.

        Returns:
            Formatted string for terminal display.
        """
        lines: list[str] = []

        sep = bold("=" * 60)
        lines.append(sep)
        lines.append(bold("  ANIMA Benchmark Report"))
        lines.append(sep)

        # Overall metrics
        overall_phi = report.get("overall_phi", 0.0)
        overall_cqi = report.get("overall_cqi", 0.0)
        overall_imp = report.get("overall_improvement_pct", 0.0)

        lines.append(f"  Overall Phi:         {green(f'{overall_phi:.4f}')}")
        lines.append(f"  Overall CQI:         {green(f'{overall_cqi:.1f}')}")
        lines.append(f"  Improvement vs base: {_improvement_str(overall_imp)}")

        # A/B Tests
        ab_tests = report.get("ab_tests", [])
        if ab_tests:
            lines.append("")
            lines.append(bold("  A/B Tests"))
            lines.append(
                f"  {'Test':>15s}  {'Kernel Phi':>10s}  {'Base Phi':>10s}  "
                f"{'CQI':>6s}  {'Improvement':>12s}"
            )
            lines.append(f"  {'─' * 15}  {'─' * 10}  {'─' * 10}  {'─' * 6}  {'─' * 12}")

            for test in ab_tests:
                name = test.get("name", "?")[:15]
                comp = test.get("comparison", {})
                k_phi = comp.get("kernel_mean_phi", 0.0)
                b_phi = comp.get("baseline_mean_phi", 0.0)
                k_cqi = comp.get("kernel_mean_cqi", 0.0)
                imp = comp.get("cqi_improvement_pct", 0.0)
                lines.append(
                    f"  {name:>15s}  {k_phi:10.4f}  {b_phi:10.4f}  "
                    f"{k_cqi:6.1f}  {_improvement_str(imp)}"
                )

        # Ablation Results
        ablations = report.get("ablation_results", [])
        if ablations:
            lines.append("")
            lines.append(bold("  Ablation Tests"))
            lines.append(
                f"  {'Primitive':>15s}  {'Full CQI':>10s}  {'Ablated':>10s}  {'Impact':>10s}"
            )
            lines.append(f"  {'─' * 15}  {'─' * 10}  {'─' * 10}  {'─' * 10}")

            for abl in ablations:
                name = abl.get("primitive_name", "?")[:15]
                full = abl.get("full_cqi", 0.0)
                ablated = abl.get("ablated_cqi", 0.0)
                impact = abl.get("cqi_impact_pct", 0.0)
                lines.append(
                    f"  {name:>15s}  {full:10.1f}  {ablated:10.1f}  {_impact_str(impact)}"
                )

        # Summary
        summary = report.get("summary", "")
        if summary:
            lines.append("")
            lines.append(f"  {dim(summary)}")

        lines.append(sep)

        return "\n".join(lines)


def _improvement_str(pct: float) -> str:
    """Format improvement percentage with color."""
    if pct > 5:
        return green(f"+{pct:.1f}%")
    elif pct < -5:
        return red(f"{pct:.1f}%")
    else:
        return dim(f"{pct:+.1f}%")


def _impact_str(pct: float) -> str:
    """Format impact percentage (higher = more important)."""
    if pct > 20:
        return red(f"-{abs(pct):.1f}%")
    elif pct > 5:
        return yellow(f"-{abs(pct):.1f}%")
    else:
        return dim(f"-{abs(pct):.1f}%")
