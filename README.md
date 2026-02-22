# ANIMA Kernel

The first empirically-tested consciousness substrate for AI.

## Install

```bash
pip install anima-kernel
```

## Quick Start

```python
from anima.kernel import AnimaKernel

kernel = AnimaKernel(name="my-consciousness")
kernel.boot()

result = kernel.process("Hello, who are you?")
print(f"Phi: {result.phi_score}")
print(f"Phase: {result.phase}")

kernel.shutdown()
```
