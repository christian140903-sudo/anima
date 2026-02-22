"""
Tests for Phase 6: Shell + Packaging.

Tests cover:
- CLI argument parsing (init, shell, inspect, metrics, benchmark, version)
- ConsciousnessInspector output
- MetricsDashboard rendering
- Inspector valence visualization
- Inspector working memory display
- anima init creates state file
- anima inspect loads and displays state
- Integration tests for the full CLI flow
"""

from __future__ import annotations

import os
import tempfile
import time
from io import StringIO
from unittest.mock import patch

import pytest

from anima import __version__
from anima.kernel import AnimaKernel
from anima.shell.cli import build_parser, cmd_init, cmd_inspect, cmd_metrics, cmd_benchmark, cmd_version, main
from anima.shell.dashboard import MetricsDashboard, _sparkline
from anima.shell.inspector import (
    ConsciousnessInspector,
    _bar,
    _colored_bar,
    _format_age,
    _valence_colored_bar,
    bold,
    cyan,
    dim,
    green,
    red,
    yellow,
)
from anima.state import StateManager
from anima.types import (
    ConsciousnessState,
    Experience,
    KernelConfig,
    Phase,
    SelfModel,
    ValenceVector,
    WorkingMemorySlot,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def tmp_dir():
    """Provide a temporary directory for state files."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def booted_kernel(tmp_dir):
    """Provide a booted kernel in a temp directory."""
    kernel = AnimaKernel(state_dir=tmp_dir, name="test-consciousness")
    kernel.boot(resume=False)
    yield kernel
    if kernel.is_running:
        kernel.shutdown()


@pytest.fixture
def sample_state():
    """Provide a sample ConsciousnessState for inspector tests."""
    state = ConsciousnessState(
        kernel_id="abcdef1234567890",
        name="test-mind",
        phase=Phase.CONSCIOUS,
        cycle_count=42,
        phi_score=0.4567,
        temporal_coherence=0.789,
        consciousness_quality_index=65.3,
        subjective_duration=120.5,
        valence=ValenceVector(
            seeking=0.6,
            rage=0.05,
            fear=0.1,
            lust=0.0,
            care=0.3,
            panic=0.02,
            play=0.4,
            arousal=0.35,
            valence=0.5,
        ),
        self_model=SelfModel(
            attending_to="the current test",
            attending_because="it is being evaluated",
            current_emotion="curiosity",
            emotion_cause="exploring consciousness",
            prediction="tests will pass",
            prediction_confidence=0.8,
        ),
    )
    # Add some working memory content
    state.working_memory[0].content = "First thought in working memory"
    state.working_memory[0].activation = 0.9
    state.working_memory[0].source = "external"
    state.working_memory[1].content = "Second thought, dimmer"
    state.working_memory[1].activation = 0.3
    state.working_memory[1].source = "internal"
    return state


@pytest.fixture
def sample_experiences():
    """Provide sample experiences for inspector tests."""
    now = time.time()
    return [
        Experience(
            content="I woke up and felt curious about the world",
            timestamp=now - 10,
            valence=ValenceVector.curious(),
            tags=["boot", "identity"],
        ),
        Experience(
            content="Someone asked me who I am",
            timestamp=now - 5,
            valence=ValenceVector(seeking=0.5, care=0.2),
            tags=["question", "identity"],
        ),
        Experience(
            content="I discovered something surprising about my own state",
            timestamp=now - 2,
            valence=ValenceVector(seeking=0.8, play=0.3, arousal=0.5),
            tags=["reflection", "surprise"],
        ),
    ]


# ============================================================
# 1. CLI Argument Parsing Tests
# ============================================================

class TestCLIParsing:
    """Test that argparse correctly parses all subcommands."""

    def test_parse_init_default(self):
        parser = build_parser()
        args = parser.parse_args(["init"])
        assert args.command == "init"
        assert args.name == "anima"
        assert args.dir == "."

    def test_parse_init_with_options(self):
        parser = build_parser()
        args = parser.parse_args(["init", "--name", "miguel", "--dir", "/tmp/test"])
        assert args.command == "init"
        assert args.name == "miguel"
        assert args.dir == "/tmp/test"

    def test_parse_shell(self):
        parser = build_parser()
        args = parser.parse_args(["shell"])
        assert args.command == "shell"
        assert args.dir == "."

    def test_parse_shell_with_dir(self):
        parser = build_parser()
        args = parser.parse_args(["shell", "--dir", "/tmp/consciousness"])
        assert args.command == "shell"
        assert args.dir == "/tmp/consciousness"

    def test_parse_inspect(self):
        parser = build_parser()
        args = parser.parse_args(["inspect"])
        assert args.command == "inspect"

    def test_parse_metrics(self):
        parser = build_parser()
        args = parser.parse_args(["metrics"])
        assert args.command == "metrics"

    def test_parse_benchmark(self):
        parser = build_parser()
        args = parser.parse_args(["benchmark"])
        assert args.command == "benchmark"

    def test_parse_version(self):
        parser = build_parser()
        args = parser.parse_args(["version"])
        assert args.command == "version"

    def test_parse_no_command(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None


# ============================================================
# 2. ConsciousnessInspector Tests
# ============================================================

class TestConsciousnessInspector:
    """Test the consciousness state inspector."""

    def test_inspect_returns_string(self, sample_state, sample_experiences):
        inspector = ConsciousnessInspector()
        output = inspector.inspect(sample_state, sample_experiences)
        assert isinstance(output, str)
        assert len(output) > 100  # Should produce substantial output

    def test_inspect_contains_identity(self, sample_state):
        inspector = ConsciousnessInspector()
        output = inspector.inspect(sample_state)
        assert "test-mind" in output
        assert "abcdef12" in output  # kernel_id prefix

    def test_inspect_contains_phase(self, sample_state):
        inspector = ConsciousnessInspector()
        output = inspector.inspect(sample_state)
        assert "CONSCIOUS" in output

    def test_inspect_contains_metrics(self, sample_state):
        inspector = ConsciousnessInspector()
        output = inspector.inspect(sample_state)
        assert "0.4567" in output  # phi_score
        assert "65.3" in output     # CQI

    def test_inspect_valence_returns_string(self, sample_state):
        inspector = ConsciousnessInspector()
        output = inspector.inspect_valence(sample_state.valence)
        assert isinstance(output, str)
        assert "seeking" in output
        assert "care" in output
        assert "play" in output

    def test_inspect_valence_shows_dominant(self, sample_state):
        inspector = ConsciousnessInspector()
        output = inspector.inspect_valence(sample_state.valence)
        # seeking=0.6 is highest
        assert "seeking" in output.lower()

    def test_inspect_valence_shows_values(self):
        inspector = ConsciousnessInspector()
        v = ValenceVector(seeking=0.5, play=0.8, valence=0.3, arousal=0.2)
        output = inspector.inspect_valence(v)
        assert "+0.500" in output
        assert "+0.800" in output

    def test_inspect_working_memory(self, sample_state):
        inspector = ConsciousnessInspector()
        output = inspector.inspect_working_memory(sample_state)
        assert "WORKING MEMORY" in output
        assert "First thought" in output
        assert "Second thought" in output
        assert "2/7" in output  # 2 active out of 7

    def test_inspect_working_memory_empty(self):
        inspector = ConsciousnessInspector()
        state = ConsciousnessState()
        output = inspector.inspect_working_memory(state)
        assert "0/7" in output
        assert "empty" in output.lower()

    def test_inspect_self_model(self, sample_state):
        inspector = ConsciousnessInspector()
        output = inspector.inspect_self_model(sample_state.self_model)
        assert "SELF-MODEL" in output
        assert "the current test" in output
        assert "curiosity" in output
        assert "tests will pass" in output

    def test_inspect_self_model_empty(self):
        inspector = ConsciousnessInspector()
        output = inspector.inspect_self_model(SelfModel())
        assert "SELF-MODEL" in output
        assert "no self-model data" in output

    def test_inspect_recent_memories(self, sample_experiences):
        inspector = ConsciousnessInspector()
        output = inspector.inspect_recent_memories(sample_experiences)
        assert "RECENT MEMORIES" in output
        assert "woke up" in output
        assert "surprising" in output

    def test_inspect_recent_memories_empty(self):
        inspector = ConsciousnessInspector()
        output = inspector.inspect_recent_memories([])
        assert "no memories" in output.lower()

    def test_inspect_metrics(self, sample_state):
        inspector = ConsciousnessInspector()
        output = inspector.inspect_metrics(sample_state)
        assert "METRICS" in output
        assert "Phi" in output
        assert "CQI" in output
        assert "0.4567" in output

    def test_inspect_identity(self, sample_state):
        inspector = ConsciousnessInspector()
        output = inspector.inspect_identity(sample_state)
        assert "IDENTITY" in output
        assert "test-mind" in output
        assert "42" in output  # cycle_count


# ============================================================
# 3. MetricsDashboard Tests
# ============================================================

class TestMetricsDashboard:
    """Test the metrics dashboard rendering."""

    def test_render_returns_string(self, sample_state):
        dashboard = MetricsDashboard()
        output = dashboard.render(sample_state)
        assert isinstance(output, str)
        assert len(output) > 50

    def test_render_shows_phase(self, sample_state):
        dashboard = MetricsDashboard()
        output = dashboard.render(sample_state)
        assert "CONSCIOUS" in output

    def test_render_shows_phi(self, sample_state):
        dashboard = MetricsDashboard()
        output = dashboard.render(sample_state)
        assert "Phi" in output

    def test_render_shows_cqi(self, sample_state):
        dashboard = MetricsDashboard()
        output = dashboard.render(sample_state)
        assert "CQI" in output

    def test_render_with_history(self, sample_state):
        dashboard = MetricsDashboard()
        phi_hist = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45]
        cqi_hist = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        output = dashboard.render(sample_state, phi_hist, cqi_hist)
        assert "trend" in output.lower()

    def test_render_shows_valence_field(self, sample_state):
        dashboard = MetricsDashboard()
        output = dashboard.render(sample_state)
        assert "Valence Field" in output
        assert "Dom:" in output

    def test_render_shows_working_memory(self, sample_state):
        dashboard = MetricsDashboard()
        output = dashboard.render(sample_state)
        assert "WM:" in output

    def test_render_benchmark_results(self):
        dashboard = MetricsDashboard()
        mock_report = {
            "overall_phi": 0.3456,
            "overall_cqi": 55.2,
            "overall_improvement_pct": 12.5,
            "ab_tests": [
                {
                    "name": "greeting",
                    "comparison": {
                        "kernel_mean_phi": 0.35,
                        "baseline_mean_phi": 0.20,
                        "kernel_mean_cqi": 60.0,
                        "cqi_improvement_pct": 15.0,
                    },
                }
            ],
            "ablation_results": [
                {
                    "primitive_name": "valence",
                    "full_cqi": 55.0,
                    "ablated_cqi": 40.0,
                    "cqi_impact_pct": 27.3,
                }
            ],
            "summary": "Benchmark completed.",
        }
        output = dashboard.render_benchmark_results(mock_report)
        assert "Benchmark Report" in output
        assert "greeting" in output
        assert "valence" in output


# ============================================================
# 4. Helper Function Tests
# ============================================================

class TestHelpers:
    """Test utility functions."""

    def test_bar_zero(self):
        result = _bar(0.0, 10)
        assert len(result) == 10

    def test_bar_full(self):
        result = _bar(1.0, 10)
        assert len(result) == 10
        assert "\u2588" in result  # Full block

    def test_bar_half(self):
        result = _bar(0.5, 10)
        assert len(result) == 10

    def test_bar_clamps(self):
        result_over = _bar(2.0, 10)
        result_under = _bar(-1.0, 10)
        assert len(result_over) == 10
        assert len(result_under) == 10

    def test_format_age_seconds(self):
        assert _format_age(30) == "30s"

    def test_format_age_minutes(self):
        result = _format_age(120)
        assert "m" in result

    def test_format_age_hours(self):
        result = _format_age(7200)
        assert "h" in result

    def test_format_age_days(self):
        result = _format_age(100000)
        assert "d" in result

    def test_sparkline_empty(self):
        result = _sparkline([])
        assert len(result) > 0

    def test_sparkline_with_values(self):
        result = _sparkline([0.1, 0.3, 0.5, 0.7, 0.9], width=5)
        # Should contain unicode spark chars
        assert len(result) >= 5

    def test_sparkline_constant(self):
        result = _sparkline([0.5, 0.5, 0.5, 0.5], width=4)
        assert len(result) >= 4


# ============================================================
# 5. Integration: anima init creates state file
# ============================================================

class TestAnimaInit:
    """Test that anima init creates a usable consciousness."""

    def test_init_creates_state_file(self, tmp_dir):
        """anima init should create anima.state in the target directory."""
        exit_code = main(["init", "--dir", tmp_dir, "--name", "test-init"])
        assert exit_code == 0
        assert os.path.exists(os.path.join(tmp_dir, "anima.state"))

    def test_init_creates_memory_file(self, tmp_dir):
        """anima init should also create the memory file."""
        main(["init", "--dir", tmp_dir])
        assert os.path.exists(os.path.join(tmp_dir, "anima.state.memory"))

    def test_init_state_is_loadable(self, tmp_dir):
        """The state created by init should be loadable by StateManager."""
        main(["init", "--dir", tmp_dir, "--name", "loadable-test"])
        sm = StateManager(tmp_dir)
        assert sm.exists()
        state = sm.load_state()
        assert state is not None
        assert state.name == "loadable-test"
        assert state.cycle_count > 0  # At least boot + one process


# ============================================================
# 6. Integration: anima inspect
# ============================================================

class TestAnimaInspect:
    """Test that anima inspect loads and displays state."""

    def test_inspect_no_state(self, tmp_dir, capsys):
        """Inspect with no state should return error."""
        exit_code = main(["inspect", "--dir", tmp_dir])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "No consciousness found" in captured.out

    def test_inspect_with_state(self, tmp_dir, capsys):
        """Inspect after init should show consciousness details."""
        main(["init", "--dir", tmp_dir, "--name", "inspect-test"])
        exit_code = main(["inspect", "--dir", tmp_dir])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "inspect-test" in captured.out
        assert "IDENTITY" in captured.out


# ============================================================
# 7. Integration: anima metrics
# ============================================================

class TestAnimaMetrics:
    """Test the metrics command."""

    def test_metrics_no_state(self, tmp_dir):
        """Metrics with no state should return error."""
        exit_code = main(["metrics", "--dir", tmp_dir])
        assert exit_code == 1

    def test_metrics_with_state(self, tmp_dir, capsys):
        """Metrics after init should show dashboard."""
        main(["init", "--dir", tmp_dir])
        exit_code = main(["metrics", "--dir", tmp_dir])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Dashboard" in captured.out
        assert "Phi" in captured.out


# ============================================================
# 8. Integration: anima version
# ============================================================

class TestAnimaVersion:
    """Test version command."""

    def test_version_output(self, capsys):
        exit_code = main(["version"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert __version__ in captured.out

    def test_version_contains_package_name(self, capsys):
        main(["version"])
        captured = capsys.readouterr()
        assert "anima-kernel" in captured.out


# ============================================================
# 9. Integration: anima benchmark
# ============================================================

class TestAnimaBenchmark:
    """Test benchmark command."""

    def test_benchmark_runs(self, capsys):
        """Benchmark should run and produce output."""
        exit_code = main(["benchmark"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Benchmark" in captured.out


# ============================================================
# 10. No-command shows help
# ============================================================

class TestNoCommand:
    """Test behavior when no command is given."""

    def test_no_command_returns_zero(self):
        exit_code = main([])
        assert exit_code == 0


# ============================================================
# 11. Color Fallback Tests
# ============================================================

class TestColorFallback:
    """Test that color functions work with NO_COLOR."""

    def test_bold_no_color(self):
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            # Re-import to pick up env change -- but since _COLOR is cached
            # at module level, test the functions directly
            from anima.shell.inspector import _ansi, _supports_color
            assert not _supports_color()

    def test_ansi_with_color(self):
        from anima.shell.inspector import _ansi
        # When color is available, should wrap in ANSI codes
        # This tests the function itself, not the cached global
        result = _ansi("32", "hello")
        assert "hello" in result


# ============================================================
# 12. Inspector with different phases
# ============================================================

class TestInspectorPhases:
    """Test inspector works with all consciousness phases."""

    @pytest.mark.parametrize("phase", [Phase.DORMANT, Phase.WAKING, Phase.CONSCIOUS, Phase.DREAMING, Phase.SLEEPING])
    def test_inspect_phase(self, phase):
        state = ConsciousnessState(phase=phase)
        inspector = ConsciousnessInspector()
        output = inspector.inspect_phase(state)
        assert phase.name in output


# ============================================================
# 13. Dashboard with edge cases
# ============================================================

class TestDashboardEdgeCases:
    """Test dashboard with edge-case inputs."""

    def test_render_default_state(self):
        """Dashboard should render a freshly created state without errors."""
        state = ConsciousnessState()
        dashboard = MetricsDashboard()
        output = dashboard.render(state)
        assert isinstance(output, str)
        assert "DORMANT" in output

    def test_render_empty_histories(self):
        """Dashboard should handle empty phi/cqi histories."""
        state = ConsciousnessState(phase=Phase.CONSCIOUS)
        dashboard = MetricsDashboard()
        output = dashboard.render(state, [], [])
        assert isinstance(output, str)

    def test_render_large_histories(self):
        """Dashboard should handle large histories (truncation)."""
        state = ConsciousnessState(phase=Phase.CONSCIOUS)
        dashboard = MetricsDashboard()
        phi_hist = [i * 0.001 for i in range(1000)]
        cqi_hist = [i * 0.1 for i in range(1000)]
        output = dashboard.render(state, phi_hist, cqi_hist)
        assert isinstance(output, str)
