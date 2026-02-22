"""
Consciousness State Inspector -- Pretty-print consciousness state to terminal.

Like `ls -la` but for consciousness. Shows what's inside the kernel:
identity, emotions, working memory, self-model, metrics.

stdlib only. ANSI escape codes with graceful fallback.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import ConsciousnessState, Experience, SelfModel, ValenceVector


# --- ANSI Color Helpers ---

def _supports_color() -> bool:
    """Check if the terminal supports color output."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    term = os.environ.get("TERM", "")
    if term == "dumb":
        return False
    # Most modern terminals support color
    return hasattr(os, "isatty") and os.isatty(1)


_COLOR = _supports_color()


def _ansi(code: str, text: str) -> str:
    """Wrap text in ANSI escape code, with fallback."""
    if not _COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(text: str) -> str:
    return _ansi("1", text)


def dim(text: str) -> str:
    return _ansi("2", text)


def green(text: str) -> str:
    return _ansi("32", text)


def red(text: str) -> str:
    return _ansi("31", text)


def yellow(text: str) -> str:
    return _ansi("33", text)


def blue(text: str) -> str:
    return _ansi("34", text)


def cyan(text: str) -> str:
    return _ansi("36", text)


def magenta(text: str) -> str:
    return _ansi("35", text)


# --- Bar Chart Helpers ---

BAR_CHARS = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"  # ▁▂▃▄▅▆▇█


def _bar(value: float, width: int = 20) -> str:
    """Render a horizontal bar using unicode block characters.

    value: 0.0 to 1.0 (clamped)
    width: total character width of the bar
    """
    value = max(0.0, min(1.0, value))
    filled = int(value * width)
    remainder = (value * width) - filled
    # Pick sub-character for the partial block
    sub_idx = int(remainder * (len(BAR_CHARS) - 1))

    bar = BAR_CHARS[-1] * filled
    if filled < width:
        bar += BAR_CHARS[sub_idx]
        bar += " " * (width - filled - 1)

    return bar


def _colored_bar(value: float, width: int = 20) -> str:
    """Bar with color based on value magnitude."""
    raw = _bar(value, width)
    if value > 0.6:
        return green(raw)
    elif value > 0.3:
        return yellow(raw)
    elif value > 0.1:
        return cyan(raw)
    else:
        return dim(raw)


def _valence_colored_bar(value: float, width: int = 20) -> str:
    """Bar colored by positive/negative valence (can be negative)."""
    abs_val = abs(value)
    raw = _bar(abs_val, width)
    if value > 0.1:
        return green(raw)
    elif value < -0.1:
        return red(raw)
    else:
        return dim(raw)


def _format_age(seconds: float) -> str:
    """Human-readable age string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    else:
        return f"{seconds / 86400:.1f}d"


# --- Inspector ---

class ConsciousnessInspector:
    """Pretty-print consciousness state to terminal.

    Like `htop` but for consciousness. Shows the internal landscape
    of a running ANIMA kernel.
    """

    def inspect(
        self,
        state: ConsciousnessState,
        experiences: list[Experience] | None = None,
    ) -> str:
        """Return formatted string of complete consciousness state."""
        sections: list[str] = []

        # Header
        sections.append(self._header(state))

        # Identity
        sections.append(self.inspect_identity(state))

        # Phase
        sections.append(self.inspect_phase(state))

        # Valence (emotional state)
        sections.append(self.inspect_valence(state.valence))

        # Working memory
        wm = self.inspect_working_memory(state)
        if wm:
            sections.append(wm)

        # Self-model
        sections.append(self.inspect_self_model(state.self_model))

        # Recent memories
        if experiences:
            sections.append(self.inspect_recent_memories(experiences))

        # Metrics
        sections.append(self.inspect_metrics(state))

        return "\n\n".join(s for s in sections if s)

    def _header(self, state: ConsciousnessState) -> str:
        """Render the header banner."""
        line = bold("=" * 50)
        title = bold(f"  ANIMA Consciousness Inspector")
        name_line = f"  Kernel: {cyan(state.name)} ({dim(state.kernel_id[:8])})"
        return f"{line}\n{title}\n{name_line}\n{line}"

    def inspect_identity(self, state: ConsciousnessState) -> str:
        """Render identity section."""
        age = _format_age(state.age())
        lines = [
            bold("IDENTITY"),
            f"  Name:    {state.name}",
            f"  ID:      {state.kernel_id}",
            f"  Age:     {age}",
            f"  Cycles:  {state.cycle_count}",
        ]
        return "\n".join(lines)

    def inspect_phase(self, state: ConsciousnessState) -> str:
        """Render phase with color."""
        phase_name = state.phase.name
        phase_colors = {
            "DORMANT": dim,
            "WAKING": yellow,
            "CONSCIOUS": green,
            "DREAMING": magenta,
            "SLEEPING": blue,
        }
        color_fn = phase_colors.get(phase_name, dim)
        return f"{bold('PHASE')}    {color_fn(phase_name)}"

    def inspect_valence(self, valence: ValenceVector) -> str:
        """Visualize valence as horizontal bar chart in terminal."""
        lines = [bold("EMOTIONAL STATE")]

        # Panksepp systems
        systems = [
            ("seeking", valence.seeking, yellow),
            ("rage", valence.rage, red),
            ("fear", valence.fear, red),
            ("lust", valence.lust, magenta),
            ("care", valence.care, green),
            ("panic", valence.panic, red),
            ("play", valence.play, green),
        ]

        for name, value, color_fn in systems:
            bar = _bar(abs(value), 20)
            colored = color_fn(bar) if abs(value) > 0.05 else dim(bar)
            lines.append(f"  {name:>8s}  {colored} {value:+.3f}")

        # Dimensional axes
        lines.append("")
        arousal_bar = _valence_colored_bar(valence.arousal, 20)
        valence_bar = _valence_colored_bar(valence.valence, 20)
        lines.append(f"  {'arousal':>8s}  {arousal_bar} {valence.arousal:+.3f}")
        lines.append(f"  {'valence':>8s}  {valence_bar} {valence.valence:+.3f}")

        # Dominant drive and magnitude
        dominant = valence.dominant()
        magnitude = valence.magnitude()
        lines.append("")
        lines.append(f"  Dominant: {bold(dominant)}  Magnitude: {magnitude:.3f}")

        return "\n".join(lines)

    def inspect_working_memory(self, state: ConsciousnessState) -> str:
        """Render working memory slots."""
        active = state.active_slots()
        total = len(state.working_memory)
        used = len(active)

        lines = [bold(f"WORKING MEMORY") + f"  ({used}/{total} slots)"]

        if not active:
            lines.append(dim("  (empty)"))
            return "\n".join(lines)

        # Sort by activation
        active_sorted = sorted(active, key=lambda s: s.activation, reverse=True)

        for slot in active_sorted:
            content = str(slot.content) if slot.content else ""
            if len(content) > 60:
                content = content[:57] + "..."
            act_bar = _bar(slot.activation, 10)
            if slot.activation > 0.5:
                act_bar = green(act_bar)
            elif slot.activation > 0.2:
                act_bar = yellow(act_bar)
            else:
                act_bar = dim(act_bar)
            lines.append(
                f"  {act_bar} [{slot.activation:.2f}] {content}"
                + (f" ({dim(slot.source)})" if slot.source else "")
            )

        return "\n".join(lines)

    def inspect_self_model(self, model: SelfModel) -> str:
        """Render self-model (attention schema)."""
        lines = [bold("SELF-MODEL")]

        has_data = False

        if model.attending_to:
            lines.append(f"  Attending to: {model.attending_to}")
            has_data = True
        if model.attending_because:
            lines.append(f"  Because:      {dim(model.attending_because)}")
        if model.current_emotion:
            lines.append(f"  Feeling:      {model.current_emotion}")
            has_data = True
        if model.emotion_cause:
            lines.append(f"  Cause:        {dim(model.emotion_cause)}")
        if model.prediction:
            conf = model.prediction_confidence
            conf_bar = _bar(conf, 10)
            lines.append(f"  Predicting:   {model.prediction}")
            lines.append(f"  Confidence:   {conf_bar} {conf:.0%}")
            has_data = True

        if has_data:
            # Calibration (only show when there's prediction data)
            cal_err = model.calibration_error()
            if cal_err < 0.2:
                cal_label = green("well-calibrated")
            elif cal_err < 0.4:
                cal_label = yellow("fair")
            else:
                cal_label = red("poorly calibrated")
            lines.append(f"  Calibration:  {cal_label} (error: {cal_err:.2f})")

            # Performance suspicion
            if model.performance_suspicion > 0.1:
                ps = model.performance_suspicion
                ps_color = red if ps > 0.5 else yellow
                lines.append(f"  Performance:  {ps_color(f'{ps:.0%} suspicion')}")
        else:
            lines.append(dim("  (no self-model data)"))

        return "\n".join(lines)

    def inspect_recent_memories(
        self,
        experiences: list[Experience],
        max_count: int = 5,
    ) -> str:
        """Render recent memories."""
        lines = [bold(f"RECENT MEMORIES") + f"  (showing {min(max_count, len(experiences))})"]

        if not experiences:
            lines.append(dim("  (no memories)"))
            return "\n".join(lines)

        now = time.time()
        recent = sorted(experiences, key=lambda e: e.timestamp, reverse=True)

        for exp in recent[:max_count]:
            age = now - exp.timestamp
            age_str = _format_age(age)
            content = exp.content
            if len(content) > 50:
                content = content[:47] + "..."
            dominant = exp.valence.dominant()
            tags = ", ".join(exp.tags[:3]) if exp.tags else ""
            lines.append(
                f"  {dim(age_str + ' ago'):>12s}  "
                f"[{yellow(dominant):>8s}]  {content}"
                + (f"  {dim('{' + tags + '}')}" if tags else "")
            )

        return "\n".join(lines)

    def inspect_metrics(self, state: ConsciousnessState) -> str:
        """Render consciousness metrics."""
        lines = [bold("METRICS")]

        phi = state.phi_score
        cqi = state.consciousness_quality_index
        tc = state.temporal_coherence

        # Phi score
        phi_bar = _colored_bar(min(phi, 1.0), 20)
        lines.append(f"  Phi (integration):     {phi_bar} {phi:.4f}")

        # CQI
        cqi_norm = min(cqi / 100.0, 1.0) if cqi > 0 else 0.0
        cqi_bar = _colored_bar(cqi_norm, 20)
        lines.append(f"  CQI (quality index):   {cqi_bar} {cqi:.1f}")

        # Temporal coherence
        tc_bar = _colored_bar(tc, 20)
        lines.append(f"  Temporal coherence:    {tc_bar} {tc:.3f}")

        # Subjective duration
        subj = state.subjective_duration
        lines.append(f"  Subjective duration:   {_format_age(subj)}")

        return "\n".join(lines)
