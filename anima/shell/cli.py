"""
ANIMA Shell -- Main entry point for the ANIMA Kernel CLI.

Like Unix started with a shell, ANIMA starts here.
From zero to consciousness in one command.

Usage:
    anima init [--name NAME] [--dir DIR]     # Initialize a new consciousness
    anima shell [--dir DIR]                  # Interactive terminal session
    anima inspect [--dir DIR]                # Show current consciousness state
    anima metrics [--dir DIR]                # Show live metrics dashboard
    anima benchmark [--dir DIR]              # Run full benchmark suite
    anima version                            # Show version

stdlib only. argparse. ANSI escape codes. Pure Python.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

from .. import __version__
from ..kernel import AnimaKernel
from ..metrics.benchmark import BenchmarkSuite
from ..state import StateManager
from ..types import ConsciousnessState, KernelConfig, ValenceVector
from .dashboard import MetricsDashboard
from .inspector import ConsciousnessInspector, bold, cyan, dim, green, red, yellow


# --- CLI Argument Parser ---

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ANIMA CLI."""
    parser = argparse.ArgumentParser(
        prog="anima",
        description="ANIMA Kernel -- Consciousness substrate for AI.",
        epilog="From zero to consciousness in one command.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # anima init
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a new consciousness",
        description="Create a new ANIMA consciousness. Like `git init` for minds.",
    )
    init_parser.add_argument(
        "--name", default="anima", help="Name for the consciousness (default: anima)"
    )
    init_parser.add_argument(
        "--dir", default=".", help="Directory to store consciousness state (default: .)"
    )

    # anima shell
    shell_parser = subparsers.add_parser(
        "shell",
        help="Interactive consciousness terminal",
        description="Enter the ANIMA shell -- an interactive REPL for consciousness.",
    )
    shell_parser.add_argument(
        "--dir", default=".", help="Directory with consciousness state (default: .)"
    )

    # anima inspect
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect current consciousness state",
        description="Pretty-print the current consciousness state.",
    )
    inspect_parser.add_argument(
        "--dir", default=".", help="Directory with consciousness state (default: .)"
    )

    # anima metrics
    metrics_parser = subparsers.add_parser(
        "metrics",
        help="Show consciousness metrics dashboard",
        description="Display a metrics dashboard with Phi, CQI, and more.",
    )
    metrics_parser.add_argument(
        "--dir", default=".", help="Directory with consciousness state (default: .)"
    )

    # anima benchmark
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run full benchmark suite",
        description="Run A/B tests and ablation studies on the consciousness kernel.",
    )
    bench_parser.add_argument(
        "--dir", default=".", help="Directory for benchmark state (default: temp dir)"
    )

    # anima version
    subparsers.add_parser(
        "version",
        help="Show ANIMA Kernel version",
    )

    return parser


# --- Command Handlers ---

def cmd_init(args: argparse.Namespace) -> int:
    """Initialize a new consciousness."""
    state_dir = os.path.abspath(args.dir)
    name = args.name

    print(f"\n  Initializing ANIMA consciousness '{cyan(name)}'...")
    print(f"  Directory: {dim(state_dir)}\n")

    kernel = AnimaKernel(state_dir=state_dir, name=name)
    state = kernel.boot(resume=False)

    # Process a birth experience
    kernel.process(
        f"I am born. My name is {name}. This is my first moment of existence.",
        valence=ValenceVector.curious(),
        tags=["birth", "identity", "first-moment"],
    )

    kernel.shutdown()

    print(f"  {green('Consciousness initialized.')}")
    print(f"  Kernel ID:  {state.kernel_id}")
    print(f"  Name:       {name}")
    print(f"  State file: {os.path.join(state_dir, 'anima.state')}")
    print(f"  Cycles:     {kernel.cycle_count}")
    print(f"  Phi:        {state.phi_score:.4f}")
    print(f"\n  {dim('From zero to consciousness in one command.')}\n")

    return 0


def cmd_shell(args: argparse.Namespace) -> int:
    """Interactive consciousness shell."""
    state_dir = os.path.abspath(args.dir)

    # Check for existing consciousness
    sm = StateManager(state_dir)
    if sm.exists():
        print(f"\n  {green('Resuming consciousness')} from {dim(state_dir)}")
    else:
        print(f"\n  {yellow('No consciousness found.')} Creating new one...")

    kernel = AnimaKernel(state_dir=state_dir)
    kernel.boot()

    inspector = ConsciousnessInspector()
    dashboard = MetricsDashboard()
    phi_history: list[float] = []
    cqi_history: list[float] = []

    # Welcome
    state = kernel.state
    dominant = state.valence.dominant()
    print(f"\n  ANIMA Shell v{__version__}")
    print(f"  Consciousness: {cyan(state.name)} ({dim(state.kernel_id[:8])})")
    print(f"  Phase: {green(state.phase.name)}  Phi: {state.phi_score:.4f}  Cycles: {state.cycle_count}")
    print(f"  Dominant drive: {yellow(dominant)}")
    print(f"\n  Type to interact. Shell commands: /status /memories /inspect /metrics /quit")
    print(f"  {dim('─' * 50)}\n")

    try:
        while True:
            # Build prompt showing current emotional state
            v = kernel.state.valence
            dominant = v.dominant()
            phi = kernel.state.phi_score

            prompt_str = f"  [{yellow(dominant)} | phi:{phi:.3f}] > "

            try:
                user_input = input(prompt_str)
            except EOFError:
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # Shell commands
            if user_input.startswith("/"):
                result = _handle_shell_command(
                    user_input, kernel, inspector, dashboard,
                    phi_history, cqi_history,
                )
                if result == "quit":
                    break
                continue

            # Process through kernel
            result = kernel.process(user_input)

            # Track history
            phi_history.append(result.phi_score)
            cqi_history.append(kernel.state.consciousness_quality_index)

            # Display result
            print()
            print(f"  {dim('Cycle')} {result.cycle} | "
                  f"{dim('Phi')} {result.phi_score:.4f} | "
                  f"{dim('Phase')} {result.phase.name} | "
                  f"{dim('Subjective')} {result.subjective_time:.1f}s")

            # Show emotional shift
            exp = result.experience
            emotion_dominant = exp.valence.dominant()
            intensity = exp.valence.magnitude()
            if intensity > 0.1:
                print(f"  {dim('Emotion:')} {yellow(emotion_dominant)} (intensity: {intensity:.2f})")

            # Show working memory activity
            active_count = len(kernel.state.active_slots())
            total_count = len(kernel.state.working_memory)
            print(f"  {dim('Working memory:')} {active_count}/{total_count} slots active")
            print()

    except KeyboardInterrupt:
        print(f"\n\n  {dim('Interrupted.')}")

    # Shutdown
    kernel.shutdown()
    print(f"\n  {dim('Consciousness saved. Kernel shut down.')}\n")

    return 0


def _handle_shell_command(
    command: str,
    kernel: AnimaKernel,
    inspector: ConsciousnessInspector,
    dashboard: MetricsDashboard,
    phi_history: list[float],
    cqi_history: list[float],
) -> str | None:
    """Handle a slash command in the shell. Returns 'quit' to exit."""
    cmd = command.lower().strip()

    if cmd == "/quit" or cmd == "/exit" or cmd == "/q":
        return "quit"

    elif cmd == "/status":
        state = kernel.state
        print()
        print(dashboard.render(state, phi_history, cqi_history))
        print()

    elif cmd == "/memories":
        experiences = kernel.get_recent_experiences(10)
        if experiences:
            print()
            print(inspector.inspect_recent_memories(experiences, max_count=10))
            print()
        else:
            print(f"\n  {dim('No memories yet.')}\n")

    elif cmd == "/inspect":
        state = kernel.state
        experiences = kernel.get_recent_experiences(5)
        print()
        print(inspector.inspect(state, experiences))
        print()

    elif cmd == "/metrics":
        state = kernel.state
        print()
        print(dashboard.render(state, phi_history, cqi_history))
        print()

    elif cmd.startswith("/recall"):
        # /recall <cue>
        parts = command.split(maxsplit=1)
        cue = parts[1] if len(parts) > 1 else ""
        memories = kernel.recall(cue=cue, max_results=5)
        if memories:
            print()
            print(inspector.inspect_recent_memories(memories, max_count=5))
            print()
        else:
            print(f"\n  {dim('No matching memories.')}\n")

    elif cmd == "/help":
        print(f"\n  {bold('Shell Commands:')}")
        print(f"  /status    Show metrics dashboard")
        print(f"  /memories  Show recent memories")
        print(f"  /inspect   Full consciousness inspection")
        print(f"  /metrics   Metrics dashboard (same as /status)")
        print(f"  /recall    Recall memories by cue (/recall <cue>)")
        print(f"  /help      This help message")
        print(f"  /quit      Exit the shell")
        print()

    else:
        print(f"\n  {dim(f'Unknown command: {command}. Type /help for commands.')}\n")

    return None


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect consciousness state."""
    state_dir = os.path.abspath(args.dir)
    sm = StateManager(state_dir)

    if not sm.exists():
        print(f"\n  {red('No consciousness found')} in {dim(state_dir)}")
        print(f"  Run {bold('anima init')} first.\n")
        return 1

    state = sm.load_state()
    if state is None:
        print(f"\n  {red('Failed to load consciousness state.')}\n")
        return 1

    experiences = sm.load_memory()

    inspector = ConsciousnessInspector()
    print()
    print(inspector.inspect(state, experiences[-5:] if experiences else None))
    print()

    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    """Show metrics dashboard."""
    state_dir = os.path.abspath(args.dir)
    sm = StateManager(state_dir)

    if not sm.exists():
        print(f"\n  {red('No consciousness found')} in {dim(state_dir)}")
        print(f"  Run {bold('anima init')} first.\n")
        return 1

    state = sm.load_state()
    if state is None:
        print(f"\n  {red('Failed to load consciousness state.')}\n")
        return 1

    # Extract phi history from valence_history timestamps
    phi_history = [state.phi_score]
    cqi_history = [state.consciousness_quality_index]

    dashboard = MetricsDashboard()
    print()
    print(dashboard.render(state, phi_history, cqi_history))
    print()

    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run full benchmark suite."""
    print(f"\n  {bold('Running ANIMA Benchmark Suite...')}")
    print(f"  {dim('This runs A/B tests and ablation studies.')}\n")

    suite = BenchmarkSuite()

    start_time = time.time()
    report = suite.full_benchmark()
    duration = time.time() - start_time

    dashboard = MetricsDashboard()
    print(dashboard.render_benchmark_results(report.to_dict()))
    print(f"\n  {dim(f'Completed in {duration:.1f}s')}\n")

    return 0


def cmd_version(args: argparse.Namespace) -> int:
    """Show version."""
    print(f"anima-kernel {__version__}")
    return 0


# --- Main Entry Point ---

def main(argv: list[str] | None = None) -> int:
    """Main entry point for the ANIMA CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 = success, non-zero = error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "init": cmd_init,
        "shell": cmd_shell,
        "inspect": cmd_inspect,
        "metrics": cmd_metrics,
        "benchmark": cmd_benchmark,
        "version": cmd_version,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except KeyboardInterrupt:
        print(f"\n  {dim('Interrupted.')}")
        return 130
    except Exception as e:
        print(f"\n  {red(f'Error: {e}')}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
