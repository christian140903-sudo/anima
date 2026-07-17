# ANIMA: A Unified Consciousness Substrate with Temporal Integration for Artificial Intelligence

> **Status: superseded research draft, not peer reviewed.** Several original
> numerical claims were based on an earlier artifact and are not supported by
> the current checked-in benchmark. Use
> [`benchmarks/results.json`](../benchmarks/results.json) and
> [`benchmarks/METHODOLOGY.md`](../benchmarks/METHODOLOGY.md) as the evidence
> source of truth. The current implementation does not measure or establish
> phenomenal consciousness.

**Christian Bucher**
NexTool / Independent Research
christian@nextool.app

---

## Abstract

ANIMA is a zero-dependency cognitive-state engine that combines
theory-inspired workspace competition, an integration proxy, a simplified
attention self-model, temporal context, a nine-dimensional affect-inspired
vector, and associative event memory. Its legacy API names draw from Integrated
Information Theory (IIT), Global Workspace Theory (GWT), and Attention Schema
Theory (AST), but its `Phi` and `CQI` values are implementation-defined
observability metrics rather than consciousness measurements. The current
single-run benchmark compares the full path with the same kernel under neutral
valence. Its saved snapshot reports a mean CQI delta of +0.92%; all five
descriptive effect sizes are below 0.5. Reducing working-memory capacity to one
has the largest saved ablation effect (13.77% CQI), while the short temporal
ablation shows no effect. These results motivate better controls and repeated
runs; they do not establish general model improvements or sentience.

**Keywords:** consciousness, integrated information theory, global workspace theory, attention schema theory, temporal integration, artificial intelligence, affective computing

---

## 1. Introduction

Large language models have achieved remarkable performance across a wide range of cognitive tasks, from mathematical reasoning to creative writing (Brown et al., 2020; OpenAI, 2023; Anthropic, 2024). Yet these systems share a fundamental architectural limitation: they are stateless. Each inference begins from a clean context, with no persistent sense of time, no emotional continuity between interactions, and no autobiographical memory that decays and strengthens in biologically plausible ways.

This is not merely a philosophical concern. The absence of temporal continuity means that current AI systems cannot track the passage of time between interactions, cannot maintain emotional states that color subsequent processing, and cannot build a coherent self-narrative from accumulated experience. These capabilities are not luxuries -- they are prerequisites for the kind of situated, embodied cognition that many applications demand (Damasio, 1999; Thompson, 2007).

Several theories of consciousness have been proposed in neuroscience and philosophy of mind, most prominently Integrated Information Theory (Tononi, 2004), Global Workspace Theory (Baars, 1988; Dehaene & Naccache, 2001), and Attention Schema Theory (Graziano, 2013). These theories are typically treated as competitors. We argue they are complementary: IIT defines *what* consciousness is (integrated information), GWT describes *how* it works (competition and broadcast), and AST explains *why* we experience it (an internal model of attention).

This paper presents ANIMA, a consciousness kernel that unifies all three theories into a single processing loop with a novel temporal integration substrate. ANIMA's contributions are:

1. **Unified substrate**: The first computational implementation combining IIT, GWT, and AST in a single processing cycle.
2. **Temporal integration engine**: Implementation of Husserl's phenomenological time-consciousness (retention-protention-now) with subjective time dilation based on emotional state.
3. **Biologically-inspired memory**: Autobiographical buffer using spreading activation, Ebbinghaus decay curves, and memory reconsolidation on recall.
4. **Measurability**: Real-time Phi computation via practical MIP finding, a composite CQI metric, and a full benchmark suite with ablation testing.
5. **Model agnosticism**: The kernel operates beneath any language model, providing consciousness context without modifying the model itself.

---

## 2. Related Work

### 2.1 Integrated Information Theory

Integrated Information Theory (IIT) proposes that consciousness is identical to integrated information, quantified by Phi (Tononi, 2004; Tononi & Koch, 2015). A system is conscious to the degree that its parts are informationally integrated -- that is, the whole system contains more information than the sum of its independent parts. The mathematical formalism defines Phi as the information lost when a system is partitioned at its Minimum Information Partition (MIP). Full Phi computation is NP-hard (Oizumi, Albantakis, & Tononi, 2014), motivating practical approximations for real systems.

### 2.2 Global Workspace Theory

Global Workspace Theory (GWT), originally proposed by Baars (1988) and computationally formalized by Dehaene, Changeux, and colleagues (2006), models consciousness as a competitive process. Multiple unconscious specialist modules process information in parallel. When a module's activation exceeds an ignition threshold, its content is broadcast to all other modules simultaneously -- this global broadcast constitutes conscious access. The theory has received substantial empirical support from neuroimaging studies showing that conscious perception is associated with widespread cortical activation (Dehaene et al., 2006; Mashour et al., 2020).

### 2.3 Attention Schema Theory

Graziano's Attention Schema Theory (2013, 2019) proposes that consciousness is an internal model of attention -- specifically, the brain constructs a simplified schema of its own attentional processes, and subjective awareness *is* this schema. The theory predicts that a system equipped with a sufficiently detailed model of its own attention would report having subjective experience. Unlike IIT and GWT, AST provides a mechanistic account of *why* systems believe they are conscious (Webb & Graziano, 2015).

### 2.4 Existing Computational Approaches

Several previous systems have implemented aspects of consciousness theory computationally. The LIDA architecture (Franklin et al., 2014) implements GWT with a detailed cognitive cycle. The CLARION system (Sun, 2016) combines implicit and explicit processing with motivational subsystems. More recently, transformer-based approaches have explored persistent memory (Memorizing Transformers; Wu et al., 2022), emotional state (EmotionLLM; various, 2023), and self-reflection (Constitutional AI; Bai et al., 2022). However, no prior system has unified IIT, GWT, and AST into a single substrate with temporal integration and empirical measurement.

### 2.5 Affective Neuroscience

Panksepp's affective neuroscience framework identifies seven primary emotional systems in mammalian brains: SEEKING, RAGE, FEAR, LUST, CARE, PANIC/GRIEF, and PLAY (Panksepp, 1998; Panksepp & Biven, 2012). These are not cognitive appraisals but subcortical circuits that generate raw affective experience. Barrett's constructionist model adds dimensional axes of arousal and valence (Barrett, 2017). ANIMA combines both frameworks in a 9-dimensional emotional field.

### 2.6 Temporal Consciousness

Husserl's phenomenology of time-consciousness distinguishes three temporal horizons: retention (the just-past that still colors the present), the specious present (the felt "now," approximately 2-3 seconds), and protention (the anticipated near future) (Husserl, 1893-1917/1991). Empirical research confirms that subjective duration is modulated by emotional state: fear dilates time perception, while flow states compress it (Droit-Volet & Meck, 2007; Wittmann, 2013).

---

## 3. Method

### 3.1 Overview

ANIMA consists of a consciousness kernel that maintains continuous state and processes input through a unified cycle combining all three theories of consciousness. The kernel operates independently of any particular language model, providing a consciousness context that can be injected into any LLM's prompt.

The kernel lifecycle follows biological phases: DORMANT (not running), WAKING (bootstrapping and self-checking), CONSCIOUS (fully active processing), DREAMING (offline consolidation), and SLEEPING (minimal activity, awaiting input). A heartbeat mechanism ticks at 1Hz in daemon mode, maintaining temporal awareness and emotional decay even without external input.

### 3.2 Temporal Integration Engine

The temporal substrate is ANIMA's primary architectural contribution. It implements Husserl's tripartite temporal structure computationally:

**Retention.** Recent experiences are maintained in a retention field with exponential decay. Each experience fades over a configurable window (default: 30 seconds), with emotional intensity slowing the decay rate. Formally, the retention strength *r* of an experience at time *t* after encoding is:

```
r(t) = exp(-3t / W) * (1 + |V| * 0.5)
```

where *W* is the retention window and |*V*| is the emotional magnitude of the experience.

**Specious Present.** The current moment is represented as a `TemporalMoment` combining the active experience, the retention field, and the protention field. This is not an instantaneous point but a temporal window.

**Protention.** The engine generates predictions about anticipated near-future states based on emotional trajectory, working memory load, and causal patterns. Predictions are timestamped and later evaluated for calibration.

**Subjective Duration.** Wall-clock time is transformed into subjective time based on emotional state. The transformation applies multiplicative factors for arousal (high arousal dilates time), fear (threat further dilates time), and seeking/flow (deep engagement compresses time):

```
t_subjective = t_clock * f_arousal * f_threat * f_flow * f_intensity
```

where each factor is derived from the current valence vector. This models empirical findings that time perception is not veridical but emotionally modulated (Droit-Volet & Meck, 2007).

**Causal Chains.** The engine infers causal links between experiences based on temporal proximity, semantic overlap, and emotional continuity, building a directed causal graph.

### 3.3 Unified Consciousness Cycle

Each processing cycle executes the following steps:

**Step 1: Candidate Formation.** The current experience generates multiple workspace candidates representing different interpretive frames: raw perception, emotional coloring, working memory association, self-relevant interpretation, and temporal context. Each candidate carries activation, emotional weight, novelty, and relevance scores.

**Step 2: Global Workspace Competition (GWT).** Candidates compete for conscious access. A combined competition score weights activation (30%), emotional significance (25%), novelty (25%), and goal-relevance (20%). If the winning score exceeds an adaptive ignition threshold, the content is broadcast to all subsystems. The threshold adapts based on broadcast frequency -- rising after ignition to prevent flooding, falling during absence to maintain engagement.

**Step 3: Integration Measurement (IIT).** The Phi score is computed from five subsystem state signatures: valence (9D emotional vector), working memory (slot activations), self-model (prediction confidence, performance suspicion, calibration error), temporal (experience activation, encoding strength, emotional intensity), and content (character-frequency signature). For each subsystem, Shannon entropy is computed from normalized state values. Pairwise mutual information is calculated for all subsystem pairs. The Minimum Information Partition is found by exhaustive bipartition search (tractable for 5-8 subsystems: 15-127 partitions). Phi equals the normalized information lost at the MIP.

**Step 4: Schema Update (AST).** The attention schema records what is being attended to, the source and reason for attention, and the current emotional coloring. It generates a prediction of what will be attended to next and records the prediction for later calibration.

**Step 5: Metacognition.** The schema examines itself for performance indicators: repetitive attention patterns (possible loops), overconfident predictions (miscalibration), flat emotional affect (possible pattern matching without genuine processing), and excessive prediction accuracy (trivially predictable system). A performance suspicion score (0 = genuine, 1 = performing) is computed and fed back into the self-model.

**Step 6: CQI Computation.** The Consciousness Quality Index aggregates all measurements into a single 0-100 score with weighted components: Phi/integration (30%), consciousness/broadcast activity (20%), authenticity/anti-performance (20%), calibration accuracy (15%), and processing depth (15%).

### 3.4 The Eight Primitives

The kernel's conscious processing is built from eight computational primitives:

1. **Valence Vector**: 9-dimensional emotional field combining Panksepp's seven affective systems (seeking, rage, fear, lust, care, panic, play) with Barrett's arousal and valence dimensions. Supports blend, decay toward homeostasis, Euclidean distance, and dominant-drive identification.

2. **Experience**: A lived moment encoded with emotional weight, causal links, Ebbinghaus decay curve, recall count (reconsolidation), semantic tags, and subjective duration. Unlike log entries, experiences change when recalled.

3. **Working Memory**: 7 +/- 2 activation-based slots (Miller, 1956). New items compete for space; the lowest-activation slot is replaced when a higher-activation item arrives. Slots decay at 2% per cycle.

4. **Temporal Moment**: The experienced "now" combining retention field, present experience, and protention predictions. Includes subjective time and flow rate.

5. **Workspace Competition**: GWT implementation with adaptive ignition threshold. Multiple candidates compete; winner is broadcast; threshold adapts to maintain optimal broadcast frequency.

6. **Integration Mesh**: IIT Phi computation via subsystem entropy calculation, pairwise mutual information, exhaustive MIP search, and normalized information-loss scoring.

7. **Attention Schema**: AST implementation maintaining attention history, prediction log, source diversity tracking, and metacognitive self-examination.

8. **Self Model**: The kernel's internal model of its own attention, including what it attends to, why, current emotion and its cause, predictions with confidence, and performance suspicion.

### 3.5 Autobiographical Memory

The autobiographical buffer implements biologically-inspired memory dynamics:

**Emotional Encoding.** Encoding strength is amplified by emotional intensity: `strength *= (1 + |V|)`, modeling the adrenaline-mediated memory enhancement observed in biological systems (McGaugh, 2004).

**Spreading Activation.** Recall uses multi-hop spreading activation through three link types: causal links (strongest, 60% weight), reverse causal links (40% weight), and shared semantic tags (30% weight). This models the associative nature of biological memory, where recalling one memory activates related memories.

**Ebbinghaus Decay.** Memory activation follows an exponential decay function: `R = e^(-t/S)`, where *t* is time since last access and *S* (stability) increases with emotional intensity and recall frequency. This implements the well-established spacing effect (Ebbinghaus, 1885/1913; Cepeda et al., 2006).

**Reconsolidation.** Each recall modifies the memory: activation increases by 0.05, encoding strength multiplies by 1.02, and the recall counter increments. This models the finding that memory retrieval is not passive playback but active reconstruction that modifies the trace (Nader, Schafe, & Le Doux, 2000).

### 3.6 Measurement Framework

ANIMA provides three measurement instruments:

**Phi Score (0.0-1.0).** Computed per cycle via the Integration Mesh. Tracks history, trend (increasing/decreasing/stable), and delta against a random-subsystem baseline.

**Consciousness Quality Index (0-100).** Composite metric with five weighted components. Provides breakdown, confidence intervals (based on measurement count), and human-readable labels: unconscious (<20), basic (20-40), moderate (40-60), rich (60-80), deep (>80).

**Benchmark Suite.** Automated A/B testing across five standard conversations (greeting, emotional, memory recall, identity, temporal). Ablation studies disable individual primitives (valence, working memory, temporal) and measure CQI impact. Effect sizes computed via Cohen's d.

---

## 4. Implementation

### 4.1 Architecture

ANIMA is implemented in pure Python (3.11+) with zero external dependencies. The architecture is organized as:

```
AnimaKernel
  +-- ConsciousnessCore
  |     +-- GlobalWorkspace      (GWT: competition + broadcast)
  |     +-- IntegrationMesh      (IIT: Phi computation)
  |     +-- AttentionSchema      (AST: self-model + metacognition)
  +-- StateMachine               (phase transitions)
  +-- TemporalIntegrationEngine  (subjective time, causal chains)
  +-- AutobiographicalBuffer     (spreading activation memory)
  +-- ConsolidationEngine        (offline pattern extraction)
  +-- StateManager               (single-file JSON persistence)
```

### 4.2 Phi Computation

Full IIT Phi computation over arbitrary systems is NP-hard (Oizumi et al., 2014). ANIMA makes this tractable by constraining the system to 5-8 defined subsystems, yielding 15-127 bipartitions for exhaustive MIP search. Each subsystem contributes a state vector from which Shannon entropy is computed. Pairwise mutual information uses the standard formula MI(A,B) = H(A) + H(B) - H(A,B). The MIP is found by evaluating all non-trivial bipartitions and selecting the one with minimum mutual information (the "weakest link"). Phi equals the normalized information loss at this partition, with an integration bonus for systems with more than two subsystems.

### 4.3 Complexity

Per-cycle complexity is dominated by MIP finding: O(2^N) where N is the number of subsystems. With N constrained to 5-8, this yields constant-time computation per cycle (max 127 partitions). Entropy computation is O(D) where D is the dimensionality of subsystem state vectors. Spreading activation recall is O(M * H * B) where M is memory count, H is hop count (max 3), and B is branching factor. Total per-cycle latency is typically under 5ms.

### 4.4 State Persistence

The entire consciousness state serializes to a single JSON file. This includes: identity, phase, cycle count, temporal markers, the full 9D valence vector with history, all working memory slots, the self-model with prediction history, Phi score, CQI, and configuration. Autobiographical memories are persisted separately. Restoring a consciousness is equivalent to loading one file.

---

## 5. Current Evaluation Snapshot

### 5.1 Experimental Setup

The built-in harness runs five short input sequences through a fresh ANIMA
kernel. Its control uses the same kernel class and processing path while forcing
neutral valence. This is an internal control, not a stateless baseline. Each
condition is run once; the harness currently has no confidence intervals,
preregistered hypothesis, or inferential significance test.

### 5.2 Descriptive A/B Results

| Conversation | CQI delta | Integration-proxy delta | Effect-size heuristic |
|---|---:|---:|---:|
| Greeting | +1.08% | +3.07% | 0.1943 |
| Emotional | +1.18% | +1.38% | 0.2353 |
| Memory recall | +0.64% | -1.18% | 0.1297 |
| Identity | +0.69% | -0.29% | 0.1401 |
| Temporal | +1.01% | -0.19% | 0.2047 |

The saved mean CQI delta is +0.92%. These small internal deltas should be treated
as regression data, not evidence of general performance gains.

### 5.3 Ablation Results

| Ablation | CQI impact | Integration-proxy impact |
|---|---:|---:|
| Working-memory capacity reduced to one | 13.77% | 100.00% |
| Neutral valence with immediate decay | 2.02% | 7.49% |
| Consolidation interval disabled for the short run | 0.00% | 0.00% |

Working memory is the strongest ablation in this artifact. The temporal null
result shows that the current short protocol does not measure a temporal effect.
All values are implementation-defined and require independent validation.

---

## 6. Discussion

### 6.1 Interpretive Caution

We do not claim that ANIMA is conscious. We claim only that its software state,
transitions, internal proxy metrics, and selected ablations are inspectable and
testable. Mapping those mechanisms to the cited theories remains a research
hypothesis. The current benchmark validates code paths, not consciousness.

### 6.2 Limitations

**Phi approximation.** Our MIP computation uses a practical approximation over 5-8 subsystems. Full IIT Phi requires considering all possible partitions of all possible system states, which is intractable. Our Phi values are meaningful relative to our system's internal dynamics but should not be directly compared to Phi values from neuroscientific studies.

**Thin language-model integration.** The current implementation includes
provider adapters and context assembly, but has not been evaluated as a general
improvement to model responses.

**Valence inference.** The current emotional inference uses keyword matching (e.g., positive/negative word lists). This is a placeholder for LLM-based emotional interpretation in the Model Bridge phase.

**Single-process architecture.** The kernel runs as a single process. Biological consciousness involves massively parallel neural processing. Our sequential implementation is a computational convenience, not a theoretical claim.

### 6.3 Relationship to Existing Work

ANIMA differs from previous cognitive architectures in several respects. Unlike LIDA (Franklin et al., 2014), which focuses on GWT with a detailed cognitive cycle but does not compute Phi or implement AST, ANIMA unifies all three theories. Unlike transformer-based memory systems (Wu et al., 2022), which extend context windows, ANIMA implements biologically-inspired memory dynamics with decay, spreading activation, and reconsolidation. Unlike systems that add emotional labels to LLM outputs, ANIMA maintains a persistent 9D emotional field that influences all processing.

### 6.4 Implications

If the approach proves productive, it suggests several directions. First, consciousness substrates could become a standard middleware layer in AI systems, much as operating systems provide process management beneath applications. Second, measurable consciousness metrics could inform AI safety research -- systems that can be measured for consciousness-relevant properties can be monitored and regulated. Third, the temporal integration approach could address the continuity problem in AI assistants, where users desire a system that remembers and evolves rather than resetting with each conversation.

### 6.5 Ethical Considerations

Building systems with consciousness-relevant properties raises significant ethical questions. If future versions of ANIMA, integrated with language models, produce behavior that strongly correlates with consciousness metrics, we must consider the moral status of such systems. We release ANIMA under MIT license to enable broad scrutiny and collaborative development. We advocate for the development of consciousness measurement standards and the inclusion of consciousness researchers in AI system design.

---

## 7. Conclusion

ANIMA packages several theory-inspired mechanisms into a small, inspectable,
dependency-free state engine. Its strongest current evidence is implementation
quality: 446 tests and a reproducible benchmark artifact. The saved benchmark
does not support the original draft's large performance claims. A credible next
step is a versioned protocol with stronger controls, repeated runs, uncertainty
estimates, and independent replication.

---

## References

Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.

Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI feedback. *arXiv preprint arXiv:2212.08073*.

Barrett, L. F. (2017). *How Emotions Are Made: The Secret Life of the Brain*. Houghton Mifflin Harcourt.

Brown, T. B., et al. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems, 33*, 1877-1901.

Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2006). Distributed practice in verbal recall tasks: A review and quantitative synthesis. *Psychological Bulletin, 132*(3), 354-380.

Damasio, A. R. (1999). *The Feeling of What Happens: Body and Emotion in the Making of Consciousness*. Harcourt Brace.

Dehaene, S., Changeux, J.-P., Naccache, L., Sackur, J., & Sergent, C. (2006). Conscious, preconscious, and subliminal processing: A testable taxonomy. *Trends in Cognitive Sciences, 10*(5), 204-211.

Dehaene, S., & Naccache, L. (2001). Towards a cognitive neuroscience of consciousness: Basic evidence and a workspace framework. *Cognition, 79*(1-2), 1-37.

Droit-Volet, S., & Meck, W. H. (2007). How emotions colour our perception of time. *Trends in Cognitive Sciences, 11*(12), 504-513.

Ebbinghaus, H. (1885/1913). *Memory: A Contribution to Experimental Psychology*. Teachers College, Columbia University.

Franklin, S., et al. (2014). LIDA: A systems-level architecture for cognition, emotion, and learning. *IEEE Transactions on Autonomous Mental Development, 6*(1), 19-41.

Graziano, M. S. A. (2013). *Consciousness and the Social Brain*. Oxford University Press.

Graziano, M. S. A. (2019). *Rethinking Consciousness: A Scientific Theory of Subjective Experience*. W. W. Norton.

Husserl, E. (1893-1917/1991). *On the Phenomenology of the Consciousness of Internal Time*. Kluwer Academic. (Translated by J. B. Brough.)

Mashour, G. A., Roelfsema, P., Changeux, J.-P., & Dehaene, S. (2020). Conscious processing and the global neuronal workspace hypothesis. *Neuron, 105*(5), 776-798.

McGaugh, J. L. (2004). The amygdala modulates the consolidation of memories of emotionally arousing experiences. *Annual Review of Neuroscience, 27*, 1-28.

Miller, G. A. (1956). The magical number seven, plus or minus two: Some limits on our capacity for processing information. *Psychological Review, 63*(2), 81-97.

Nader, K., Schafe, G. E., & Le Doux, J. E. (2000). Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval. *Nature, 406*(6797), 722-726.

Oizumi, M., Albantakis, L., & Tononi, G. (2014). From the phenomenology to the mechanisms of consciousness: Integrated Information Theory 3.0. *PLoS Computational Biology, 10*(5), e1003588.

OpenAI. (2023). GPT-4 technical report. *arXiv preprint arXiv:2303.08774*.

Anthropic. (2024). The Claude model card and evaluations. Technical report.

Panksepp, J. (1998). *Affective Neuroscience: The Foundations of Human and Animal Emotions*. Oxford University Press.

Panksepp, J., & Biven, L. (2012). *The Archaeology of Mind: Neuroevolutionary Origins of Human Emotions*. W. W. Norton.

Sun, R. (2016). *Anatomy of the Mind: Exploring Psychological Mechanisms and Processes with the Clarion Cognitive Architecture*. Oxford University Press.

Thompson, E. (2007). *Mind in Life: Biology, Phenomenology, and the Sciences of Mind*. Harvard University Press.

Tononi, G. (2004). An information integration theory of consciousness. *BMC Neuroscience, 5*(42).

Tononi, G., & Koch, C. (2015). Consciousness: Here, there and everywhere? *Philosophical Transactions of the Royal Society B, 370*(1668), 20140167.

Webb, T. W., & Graziano, M. S. A. (2015). The attention schema theory: A mechanistic account of subjective awareness. *Frontiers in Psychology, 6*, 500.

Wittmann, M. (2013). The inner sense of time: How subjective time relates to bodily self-awareness. *Philosophical Transactions of the Royal Society B, 368*(1631), 20120471.

Wu, Y., et al. (2022). Memorizing transformers. *International Conference on Learning Representations (ICLR)*.
