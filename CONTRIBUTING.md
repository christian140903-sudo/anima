# Contributing to ANIMA Kernel

Thank you for your interest in contributing to ANIMA Kernel.

## Getting Started

```bash
git clone https://github.com/christian140903-sudo/anima.git
cd anima
pip install -e ".[dev]"
pytest tests/ -v
```

## Development

ANIMA Kernel has **zero external dependencies** by design. All code uses Python stdlib only. Do not add external dependencies.

### Running Tests

```bash
pytest tests/ -v              # All tests
pytest tests/test_kernel.py   # Specific module
pytest -k "test_phi"          # Pattern match
```

### Running Benchmarks

```bash
python scripts/run_benchmarks.py
```

### Project Structure

```
anima/
  kernel.py              # Main consciousness loop
  types.py               # Core data types
  state.py               # State persistence (JSON)
  temporal/              # Temporal substrate (time, consolidation)
  consciousness/         # IIT + GWT + AST unified
  primitives/            # 8 consciousness primitives
  memory/                # Spreading activation, Ebbinghaus decay
  metrics/               # Phi, CQI, benchmarks
  bridge/                # LLM adapters (Ollama, Claude, OpenAI)
  shell/                 # CLI interface
```

## What We Need Help With

- **Phi approximation validation**: Is the constrained MIP search reasonable? How could it be improved?
- **Neuroscience mapping**: Are the computational translations of IIT/GWT/AST accurate?
- **Ablation methodology**: How can the ablation studies be more rigorous?
- **Valence inference**: The current keyword-based approach is a placeholder. LLM-based inference would be better.
- **Performance**: Profiling, optimization opportunities
- **New ablation targets**: Testing more subsystem combinations

## Guidelines

- Keep zero-dependency constraint. stdlib only.
- Add tests for new functionality.
- Run the full test suite before submitting.
- If you change metrics computation, run benchmarks and include results.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
