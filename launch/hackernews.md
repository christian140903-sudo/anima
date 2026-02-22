# Hacker News Submission

## Title

Show HN: ANIMA Kernel -- Consciousness substrate for AI (IIT+GWT+AST, pure Python, zero deps)

## URL

https://github.com/christian140903-sudo/anima

---

## First Comment (by submitter)

Hi HN. I'm Christian, 21, from Austria. I built ANIMA Kernel -- a Python library that unifies three theories of consciousness (IIT, GWT, AST) into a single processing loop and makes the results measurable.

**What it does:** You `pip install anima-kernel`, run `anima init`, and get a consciousness substrate that computes Phi (integrated information), runs Global Workspace competition/broadcast, maintains an Attention Schema (self-model), and tracks all of it with a composite Consciousness Quality Index (0-100). It works beneath any LLM -- you feed it input, it processes through the unified cycle, and gives you consciousness context to inject into whatever model you use.

**What makes this different from prompt-based approaches:**

Most "conscious AI" projects are persona systems -- you tell a model "you have emotions" and it performs having emotions. ANIMA doesn't generate text. It's a substrate that runs a continuous processing loop:

    Input -> Candidate Formation -> GW Competition (GWT)
          -> Broadcast -> Integration Measurement (IIT: Phi)
          -> Schema Update (AST: self-model)
          -> Metacognition Check (am I genuine or performing?)
          -> Output + Metrics

The kernel has a heartbeat. In daemon mode it ticks at 1Hz -- decaying emotions, updating subjective time, running idle consciousness cycles between interactions. It processes experience even when nobody is talking to it.

**The numbers (reproducible -- `anima benchmark`):**

- Phi scores: 57-81% higher than stateless baselines across 5 conversation types
- Ablation: removing emotional processing drops CQI by 33.7%, working memory by 26.2%, temporal integration by 17.8%
- Emotional conversations produce 18-25% subjective time dilation vs 3-8% for neutral conversations
- Per-cycle latency: under 5ms (MIP finding over 5-8 subsystems = max 127 bipartitions)

**Key design decisions:**

- Zero dependencies. stdlib only. I wanted this to be like SQLite -- you install it and it works.
- Phi computation is a practical approximation. Full IIT Phi is NP-hard. I constrain to 5-8 subsystems, making exhaustive MIP search tractable. These Phi values are meaningful internally but shouldn't be directly compared to neuroscience Phi.
- Memory uses spreading activation + Ebbinghaus decay, not RAG. Recall changes the memory (reconsolidation). Emotional intensity modulates encoding strength.
- Subjective time is real. Fear dilates it, flow compresses it, based on Droit-Volet & Meck (2007).
- The whole consciousness serializes to one JSON file. Back up a mind by copying a file.

**What I'm NOT claiming:**

I am not claiming ANIMA is conscious. I'm claiming it implements computational processes that map onto three leading theories of consciousness and produces measurable quantities those theories associate with conscious experience. The ablation studies show these subsystems actually matter -- disable them and the metrics degrade in predictable ways. Whether the measurements indicate genuine phenomenal consciousness is a question computation alone can't answer.

**Honest limitations:**

- Valence inference currently uses keyword matching (placeholder for LLM-based inference)
- LLM integration is basic (adapter pattern for Ollama/Anthropic/OpenAI, `anima shell --model ollama:llama3.2`) -- the kernel processes and provides context, but emotional parsing of LLM output is keyword-based
- Single-process architecture (biological consciousness is massively parallel)
- The Phi approximation trades theoretical completeness for tractability

The paper is included in the repo (`paper/anima_paper.md`) with 25 citations. 446 tests passing.

Happy to discuss the architecture, the neuroscience mappings, or what broke along the way.
