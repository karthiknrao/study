# Deep Dive: PDDL-Based Planning with LLMs/Agents

## Background: What is PDDL?

**Planning Domain Definition Language (PDDL)** is the standard formal language for representing planning problems in AI. A PDDL specification consists of:

- **Domain file**: Defines predicates, actions, preconditions, and effects
- **Problem file**: Specifies initial state, objects, and goal conditions
- **Planner**: A symbolic solver that finds valid action sequences

LLMs struggle with direct plan generation for complex, multi-constraint problems—they produce "plausible" but often invalid plans. The **LLM+PDDL paradigm** addresses this by having LLMs translate natural language to formal PDDL, then using sound symbolic planners for guaranteed correctness.

---

## Key Paradigms for LLM + PDDL Integration

### 1. **LLM as Formalizer** (Translation to PDDL)

The dominant paradigm—LLMs convert natural language to PDDL, symbolic planners solve it.

| Framework | Key Contribution | Source |
|-----------|-----------------|--------|
| **LLM+P** | First architecture combining LLM translation with PDDL planners | [ICAPS 2024](https://ojs.aaai.org/index.php/ICAPS/article/download/31502/33662) |
| **L2P Library** | Tools for NL→PDDL generation, domain-agnostic | [GitHub: AI-Planning/l2p](https://github.com/AI-Planning/l2p) |
| **LLM-Modulo** | RAG-augmented PDDL generation with iterative feedback | [Stuttgart](https://elib.uni-stuttgart.de/items/8d5f0a2e-f7c2-474a-b8c7-0f13db4a96e6) |
| **Planetarium** | Rigorous benchmark for NL→PDDL translation | [NAACL 2025](https://arxiv.org/abs/2407.03321) |

**Limitation identified**: [On the Limit of Language Models as Planning Formalizers (ACL 2025)](https://aclanthology.org/2025.acl-long.242.pdf) systematically evaluates this methodology and finds persistent challenges in semantic correctness.

### 2. **LLM + Formal Verification Tools**

Combining LLMs with satisfiability solvers (SAT/SMT) for provably correct plans.

| Paper | Method | Performance |
|-------|--------|-------------|
| **[NAACL 2025: LLMs + Formal Verification](https://aclanthology.org/2025.naacl-long.176/)** | LLM formalizes problems as constrained SAT, solver finds solutions | **93.9% success** on TravelPlanner vs 10% for o1-preview |

This work tackles the TravelPlanner benchmark:
- **Problem**: Multi-constraint travel planning (flights, meals, hotels, attractions)
- **LLM-only**: GPT-4 achieves 0.6%, o1-preview achieves 10%
- **LLM+Formal**: 93.9% success rate with zero-shot generalization

### 3. **LLM-Generated Heuristics for Classical Planners**

LLMs generate domain-specific heuristics to guide symbolic planners.

| Paper | Approach | Venue |
|-------|----------|-------|
| [Classical Planning with LLM-Generated Heuristics](https://openreview.net/forum?id=UCV21BsuqA) | LLM reasons about PDDL domain to produce heuristic functions | OpenReview |

### 4. **Neuro-Symbolic Robot Task Planning**

Hybrid frameworks combining LLM reasoning with PDDL for robotics.

| Framework | Contribution | Source |
|-----------|-------------|--------|
| **NSLM** | LLM proposes subgoals → PDDL planner solves constrained search | [arXiv 2409.19250](https://arxiv.org/abs/2409.19250) |
| **SPCA** | Sense-Plan-Code-Act: perception→PDDL→planning→execution | [MDPI 2025](https://www.mdpi.com/2504-4990/8/1/22) |
| **Neuro-Symbolic Replanning** | Multimodal LLM extracts semantic/spatial info → PDDL encoding | [OpenReview](https://openreview.net/pdf?id=Xx3rwBDm46) |

### 5. **Agentic PDDL Simulation**

LLMs operate as agents *inside* PDDL simulation environments.

| Framework | Key Idea | Source |
|-----------|----------|--------|
| **PyPDDLEngine** | LLM executes one action at a time, observes state, can reset/retry | [arXiv 2603.06064](https://arxiv.org/pdf/2603.06064) |
| **End-to-End Agentic Framework** | Dynamic multi-agent orchestration for PDDL construction | [arXiv 2512.09629](https://arxiv.org/html/2512.09629v1) |

### 6. **Instruction Tuning for PDDL Reasoning**

Training LLMs to reason symbolically about planning.

| Framework | Method | Results |
|-----------|--------|---------|
| **[PDDL-INSTRUCT (MIT CSAIL)](https://arxiv.org/abs/2509.13351)** | Logical CoT + external plan validation (VAL) | **94% valid plans** on Blocksworld (Llama-3-8B), 64x improvement |

---

## Major Benchmarks

### PlanBench ([NeurIPS 2023](https://arxiv.org/abs/2206.10498))
- **Purpose**: Evaluate LLMs on classical planning capabilities
- **Domains**: Blocksworld, Mystery Blocksworld, and more
- **Key Finding**: LLMs struggle even with SOTA models; performance degrades with obfuscated PDDL
- **Repository**: [github.com/harshakokel/PlanBench](https://github.com/harshakokel/PlanBench)

### TravelPlanner ([ICML 2024 Spotlight](https://arxiv.org/abs/2402.01622))
- **Purpose**: Real-world multi-constraint planning
- **Dataset**: 1,225 curated travel planning queries, ~4M data records
- **Challenge**: GPT-4 achieves only **0.6%** success rate
- **Key insight**: Language agents struggle with task persistence, tool use, and constraint tracking

### Planetarium ([NAACL 2025](https://arxiv.org/abs/2407.03321))
- **Purpose**: Benchmark NL→PDDL translation
- **Dataset**: **145,918** text-to-PDDL pairs across 73 state combinations
- **Innovation**: Graph isomorphism-based PDDL equivalence checking
- **Finding**: Zero-shot LLMs perform poorly; fine-tuning helps significantly

---

## Performance Insights

| Approach | Task | Performance |
|----------|------|-------------|
| LLM-only (GPT-4) | TravelPlanner | 0.6% |
| LLM-only (o1-preview) | TravelPlanner | 10% |
| **LLM + Formal Verification** | TravelPlanner | **93.9%** |
| LLM-only (untuned) | Blocksworld | ~30% |
| **PDDL-INSTRUCT (tuned 8B)** | Blocksworld | **94%** |
| LLM-only (obfuscated PDDL) | PlanBench | Severe degradation |

---

## Key Architectural Patterns

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM + PDDL Planning Pipeline                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Natural    │───▶│     LLM      │───▶│    PDDL      │       │
│  │   Language   │    │  Formalizer  │    │   Domain +   │       │
│  │   Query      │    │              │    │   Problem    │       │
│  └──────────────┘    └──────────────┘    └──────┬───────┘       │
│                                                  │               │
│                    ┌──────────────┐              ▼               │
│                    │   Feedback   │◀─────┌──────────────┐        │
│                    │   / Repair   │      │   Symbolic   │        │
│                    │   (LLM)      │      │   Planner    │        │
│                    └──────────────┘      │  (FastDown-  │        │
│                          │               │   ward, etc) │        │
│                          ▼               └──────┬───────┘        │
│                    ┌──────────────┐              │               │
│                    │   Validated  │◀─────────────┘               │
│                    │     Plan     │                              │
│                    └──────────────┘                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Research Papers (2024-2025)

| Paper | Venue | Focus |
|-------|-------|-------|
| [LLMs as Planning Formalizers: A Survey](https://aclanthology.org/2025.findings-acl.1291.pdf) | ACL 2025 Findings | Comprehensive survey |
| [On the Limit of LLMs as Planning Formalizers](https://aclanthology.org/2025.acl-long.242.pdf) | ACL 2025 | Systematic evaluation, limitations |
| [Planetarium Benchmark](https://aclanthology.org/2025.naacl-long.560/) | NAACL 2025 | NL→PDDL evaluation |
| [LLMs + Formal Verification](https://aclanthology.org/2025.naacl-long.176/) | NAACL 2025 | 93.9% on TravelPlanner |
| [PDDL-INSTRUCT](https://arxiv.org/abs/2509.13351) | arXiv 2025 | Instruction tuning for planning |
| [Agentic LLM Planning via PDDL Simulation](https://arxiv.org/pdf/2603.06064) | arXiv 2026 | Step-wise PDDL simulation |
| [End-to-End Agentic Planning Framework](https://arxiv.org/html/2512.09629v1) | arXiv 2024 | Dynamic multi-agent orchestration |
| [Neuro-Symbolic Robot Planning (NSLM)](https://arxiv.org/abs/2409.19250) | IEEE 2024 | LLM proposals + PDDL verification |

---

## Open Challenges

1. **Semantic Correctness**: LLMs often generate syntactically valid but semantically wrong PDDL
2. **Domain Coverage**: Generalizing to unseen domains remains difficult
3. **Infeasibility Detection**: Identifying when user requests are unsatisfiable
4. **Long-Horizon Tasks**: Exponential search space for complex environments
5. **Real-Time Execution**: Balancing LLM inference cost with planning needs

---

## Practical Resources

| Resource | Link |
|----------|------|
| L2P Library (NL→PDDL) | [github.com/AI-Planning/l2p](https://github.com/AI-Planning/l2p) |
| PlanBench Benchmark | [github.com/harshakokel/PlanBench](https://github.com/harshakokel/PlanBench) |
| TravelPlanner Benchmark | [github.com/OSU-NLP-Group/TravelPlanner](https://github.com/OSU-NLP-Group/TravelPlanner) |
| Planetarium Benchmark | [github.com/BatsResearch/planetarium](https://github.com/BatsResearch/planetarium) |
| PDDL-INSTRUCT Framework | [github.com/ParallelLLC/symbolic-planning](https://github.com/ParallelLLC/symbolic-planning) |
| PyPDDLEngine | [arXiv 2603.06064](https://arxiv.org/pdf/2603.06064) |

---

## Related Topics

- **LLM + Formal Verification**: See pattern in `MEMORY.md` under "LLM+Formal Verification Papers"
- **Game Agent Planning**: Memory-augmented agents for strategic games (MEMO, Suspicion-Agent)
- **ML Engineering Agents**: MLE-STAR for automated pipeline development

---

*Compiled: 2026-03-15*
