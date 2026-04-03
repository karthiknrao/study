# MLEvolve-Type Agents: Deep Dive (April 2026)

## Table of Contents
1. [MLEvolve — The Core System](#1-mlevolve)
2. [Taxonomy: Three Categories](#2-taxonomy)
3. [MLE-bench Leaderboard Landscape](#3-mle-bench-landscape)
4. [Direct Competitors — Deep Profiles](#4-competitor-profiles)
5. [Evolutionary Code Discovery Systems](#5-evolutionary-code-discovery)
6. [Agent Evolution Frameworks](#6-agent-evolution-frameworks)
7. [Shared Technical DNA](#7-shared-technical-dna)
8. [Key Distinctions & Tradeoffs](#8-key-distinctions)
9. [Master Comparison Table](#9-master-comparison)

---

## 1. MLEvolve

**Team:** InternScience (Shangheng Du, Xiangchao Yan, et al.)
**Repo:** https://github.com/InternScience/MLEvolve
**Predecessor:** AutoMLGen (arXiv:2510.08511)
**Parent System:** InternAgent 1.5 (arXiv:2602.08990)
**Status:** Open-source (Feb 2026), #1 on MLE-bench at 12h budget

### Architecture
MLEvolve is an agentic MLE system that solves Kaggle-style ML competitions through:

1. **Progressive MCGS (Monte Carlo Graph Search)**
   - Extends vanilla UCT with piecewise exploration decay
   - Time-aware explore-exploit switching
   - Automatic stagnation detection
   - Multiple solution branches evolve in parallel

2. **Multi-Mode Planning & Code Generation**
   - Base (single-shot) and memory-enhanced (two-stage retrieval-augmented) planning
   - Three code generation strategies: single-pass, stepwise multi-agent pipeline, incremental SEARCH/REPLACE diff patching
   - Adaptive mode dispatch based on search state

3. **Experience-Driven Memory**
   - Global memory layer recording plan, code, metrics, and success/failure labels for every node
   - Retrieval via BM25 + FAISS hybrid search
   - Different agents query memory differently to encourage novel approaches

4. **Cross-Branch Fusion**
   - When progress stalls, merges insights from top-performing nodes across different branches
   - Trajectory-aware evolution leveraging each branch's full improvement history

### Key Insight
MLEvolve treats ML pipeline design as **structured graph search** rather than evolutionary mutation. Its MCGS with UCT is fundamentally different from the evolutionary paradigms (generate → evaluate → select → mutate) used by most competitors.

---

## 2. Taxonomy: Three Categories

### Category A: MLE Competition Agents
Systems designed for automated ML engineering on Kaggle-style tasks. Benchmarked on MLE-bench.

### Category B: Evolutionary Code Discovery Systems
General-purpose evolutionary agents that discover/optimize algorithms and code. Originated with AlphaEvolve.

### Category C: Agent Evolution Frameworks
Meta-frameworks that evolve agent configurations/prompts/code across any domain.

---

## 3. MLE-bench Leaderboard Landscape

MLE-bench (OpenAI, ICLR 2025) is the primary benchmark: 75 real Kaggle competitions. Scores are "Any Medal %" (mean ± SEM).

### Main Leaderboard (as of April 2026)

| Rank | Agent | LLM | All Medal % | Time |
|------|-------|-----|------------|------|
| 1 | **Famou-Agent 2.0** | Gemini-3-Pro-Preview | **64.44%** | 24h |
| 2 | AIBuildAI | Claude-Opus-4.6 | 63.11% | 24h |
| 3 | CAIR MARS+ | Gemini-3-Pro-Preview | 62.67% | 24h |
| 4 | **MLEvolve** | Gemini-3-Pro-Preview | **61.33%** | **12h** |
| 5 | PiEvolve | Gemini-3-Pro-Preview | 61.33% | 24h |
| 6 | Famou-Agent 2.0 | Gemini-2.5-Pro | 59.56% | 24h |
| 7 | ML-Master 2.0 | Deepseek-V3.2-Speciale | 56.44% | 24h |
| 8 | CAIR MARS | Gemini-3-Pro-Preview | 56.0% | 24h |
| 9 | PiEvolve (12h) | Gemini-3-Pro-Preview | 52.0% | 12h |
| 10 | Leeroo | Gemini-3-Pro-Preview | 50.67% | 24h |

### Additional Leaderboard (uses test-set feedback)

| Agent | Score | Notes |
|-------|-------|-------|
| Disarray (ensemble) | 90.91% | Claude+GPT+Gemini ensemble |
| LoongFlow | 77.27% | Gemini-3-Flash-Preview |

**Key observation:** MLEvolve achieves 61.33% in just 12 hours — half the budget of most competitors at the same score level.

---

## 4. Direct Competitor Deep Profiles

### 4.1 AIDE (AI-Driven Exploration in the Space of Code)
**Team:** Weco AI | **Repo:** https://github.com/WecoAI/aideml | **Paper:** arXiv:2502.13138

**Architecture:**
- **Solution Space Tree Search** — tree-structured exploration (not MCTS, but greedy tree search with LLM-guided branching)
- Three coding operators: **Drafting** (from scratch), **Debugging** (fix errors), **Improving** (atomic changes)
- **Summarization operator** keeps prompts concise regardless of tree depth
- Search policy decides whether to draft/debug/improve next

**Key Innovation:** Maintains full tree of all explored solutions with structured summaries, enabling backtracking. The paradigm-setting system that many later agents built upon.

**MLE-bench:** 17.12% (o1-preview, 24h) — lower score but pioneered the tree-search-over-code paradigm.

### 4.2 ML-Master / EvoMaster
**Team:** SJTU (Shanghai Jiao Tong) | **Repo:** https://github.com/sjtu-sai-agents/ML-Master

**Architecture:**
- **MCTS framework** with dual-LLM reasoning
- DeepSeek-R1 for code generation (with `<think/>` reasoning tags)
- GPT-4o for structured evaluation feedback
- **Adaptive Memory Mechanism** — selectively captures insights from both successes and failures

**ML-Master 2.0 Innovation — Hierarchical Cognitive Caching (HCC):**
- Dynamically distills transient execution traces into stable long-term knowledge
- Decouples tactical execution from strategic planning
- 92.7% improvement over ML-Master 1.0 (29.3% → 56.44%)

**Evolution:** ML-Master 2.0 generalized into **EvoMaster** (open-sourced Mar 2026), which also powers X-Master and Browse-Master.

**Papers:**
- ML-Master 1.0: arXiv:2506.16499
- ML-Master 2.0: arXiv:2601.10402

### 4.3 PiEvolve
**Team:** Fractal Analytics

**Architecture:**
- **Graph-structured search** over solution trajectories
- Builds a **directed acyclic graph (DAG)** of solution chains for lineage tracking and credit assignment
- **Adaptive agent orchestration** — dynamically coordinates specialized LLM agents via probabilistic action selection
- Balances exploration and exploitation
- Also applied to NP-hard optimization beyond ML (vehicle routing, bin packing, portfolio optimization)

**Paper:** "PiML: Automated Machine Learning Workflow Optimization using LLM Agents"

### 4.4 LoongFlow
**Team:** Baidu Baige | **Repo:** https://github.com/baidu-baige/LoongFlow | **Paper:** arXiv:2512.24077

**Architecture — Plan-Execute-Summarize (PES) Paradigm:**
- **Planner**: strategic reasoning and domain priors
- **Executor**: code generation and contract verification
- **Summary**: abductive reflection (analyzing why scores improved/dropped)
- **Evolution Tree** + MAP-Elites for diversity preservation
- Adaptive Boltzmann selection for exploration/exploitation balance

**ML-Evolve Agent:** Specialized agent with canonical 6-stage ML structure (Load → Cross Val → Feature Eng → Train → Ensemble → Workflow)

**Key Innovation:** "Causal reasoning" instead of random mutation. The Summary module analyzes why previous attempts failed, avoiding repeated mistakes.

**Results:** 4x fewer generation calls than OpenEvolve (258 vs 927). 100% success rate on Circle Packing vs OpenEvolve's instability. 22 gold medals on MLE-bench.

### 4.5 CAIR MARS / MARS+
**Team:** CAIR (Concentric AI Research)

**Architecture:**
- **Modular Agent with Reflective Search**
- Structured exploration and refinement for computationally intensive ML tasks
- MARS+ improved from 56.0% to 62.67%

**Paper:** "MARS: Modular Agent with Reflective Search for Automated AI Research"

### 4.6 MLE-STAR
**Team:** Google Research | **Paper:** arXiv:2506.15692 (NeurIPS 2025)

**Architecture:**
- **Search and Targeted Refinement** approach
- Leverages external knowledge via web search for ML pipeline design
- Ensemble construction and safety verification
- CAIR derivative (MLE-STAR-Pro) achieved 44.0% on MLE-bench

### 4.7 AutoKaggle
**Team:** ByteDance/Doubao Seed | **Paper:** arXiv:2410.20424 (ICLR 2025)

**Architecture:**
- Phase-based workflow: Background Analysis → Data Cleaning → Feature Engineering → Model Training → Prediction
- Multi-agent collaboration with iterative development
- Comprehensive unit testing (30+ tests)
- Code interpretation, debugging, and validation

### 4.8 Other Notable Agents

| Agent | Team | Score | Notes |
|-------|------|-------|-------|
| **Leeroo** | Startup | 50.67% | Continuous learning AI agents, open-source "superml" and "kapso" |
| **Thesis** | — | 48.44% | Referenced in "Reasoning as Gradient" paper (arXiv:2603.01692) |
| **R&D-Agent** | Microsoft | 35.11% | Research-then-development pattern, open-source |
| **AIRA-dojo** | Meta | 31.60% | Framework for developing/benchmarking AI research agents (NeurIPS 2025) |
| **InternAgent-1.5** | InternScience | 36.44% | Broader scientific discovery (not just ML); MLEvolve powers its coding module |

---

## 5. Evolutionary Code Discovery Systems (Category B)

### 5.1 AlphaEvolve (Google DeepMind)
**Paper:** arXiv:2506.13131 (Jun 2025)

**Architecture:**
1. **Prompt Sampler** — samples parent + "inspiration" programs, constructs rich prompts with scores, rendered results, stochastic formatting, literature context
2. **LLM Ensemble** — Gemini Flash (breadth/throughput) + Gemini Pro (depth/quality) generate SEARCH/REPLACE diffs
3. **Evaluators** — automated, parallelized, with cascade filtering and LLM-generated feedback
4. **Program Database** — MAP-Elites + island-based population model

**Key Innovations:**
- Evolves entire codebases (hundreds of lines), not just single functions
- Meta-prompt evolution: prompts themselves are co-evolved
- Multi-objective optimization improves even single-target metrics through diversity
- Only thousands of LLM samples needed (vs millions in predecessor FunSearch)

**Results:**
- Broke 56-year-old record: 48 multiplications for 4×4 matrices (down from 49)
- Applied to 50+ open mathematical problems: 75% matched SOTA, 20% discovered new better solutions
- Kissing number in dimension 11: 593 (up from 592)
- **Production impact:** Recovered 0.7% of Google's worldwide compute (Borg scheduler), 1% reduction in Gemini training time, 32.5% FlashAttention speedup

### 5.2 OpenEvolve
**Team:** Algorithmic Superintelligence | **Repo:** https://github.com/algorithmicsuperintelligence/openevolve

**Architecture (5 components):**
1. **Prompt Sampler** — context-rich prompts from parent programs + evidence sets
2. **LLM Ensemble** — weighted ensemble of models for candidate generation
3. **Evaluator** — cascade evaluation (stage 1/2/3 with thresholds) + artifact capture
4. **Program Database** — MAP-Elites per island, binning along feature dimensions
5. **Controller** — orchestration, seeding, logging, parallelism

**Key Algorithms:**
- Island-based evolution with lazy migration (ring topology)
- MAP-Elites for quality-diversity preservation
- Double-selection strategy (parent biased toward fitness; inspiration from diverse sources)
- Diff-based editing (SEARCH/REPLACE blocks) by default

**Results:** Replicated AlphaEvolve's circle packing (2.634 for n=26), GPU kernel optimizations (12.5% speedup on transformer attention), 321x speedup via JAX JIT discovery on AlgoTune

### 5.3 CodeEvolve
**Paper:** arXiv:2510.14150

Open-source evolutionary coding agent with independent architecture choices, inspired by AlphaEvolve's vision.

### 5.4 AVO (Agentic Variation Operators)
**Paper:** arXiv:2603.24517

Proposes self-directed coding agents as replacement for mutation and crossover in evolutionary search.

---

## 6. Agent Evolution Frameworks (Category C)

### 6.1 A-Evolve
**Team:** Amazon / A-EVO-Lab | **Repo:** https://github.com/A-EVO-Lab/a-evolve

**Architecture — "The PyTorch for Agentic AI":**
- **Agent Workspace**: standardized directory defining an agent's "DNA":
  - `manifest.yaml` (config)
  - `prompts/` (system messages)
  - `skills/` (reusable code snippets)
  - `tools/` (external API configs)
  - `memory/` (episodic data)
- **Mutation Engine**: directly modifies workspace files via LLM-driven operations
- **Five-stage loop**: Solve → Observe → Evolve → Gate → Reload
- Every mutation is **git-tagged** (`evo-1`, `evo-2`, ...) for reproducibility and rollback
- **BYOA/BYOE/BYO-Algo**: Bring Your Own Agent/Environment/Algorithm

**Results:** #1 on MCP-Atlas (79.4%), ~#5 on SWE-bench Verified (76.8%), ~#7 on Terminal-Bench 2.0 (76.5%)

### 6.2 AgentEvolver
**Paper:** arXiv:2511.10395

Self-evolving agent system leveraging LLM semantic understanding for autonomous agent learning. Addresses high data-construction costs and low exploration efficiency of traditional RL-based approaches.

---

## 7. Shared Technical DNA

### 7.1 Search Strategies
| Strategy | Systems |
|----------|---------|
| MCTS/MCGS (Graph Search) | MLEvolve, ML-Master |
| Greedy Tree Search | AIDE |
| Evolutionary Population + MAP-Elites | AlphaEvolve, OpenEvolve, LoongFlow |
| DAG-based Search | PiEvolve |
| Hybrid (Tree + Evolution) | MLEvolve (cross-branch fusion), LoongFlow |
| File-based Mutation Loop | A-Evolve |

### 7.2 Memory / Experience Mechanisms
| Mechanism | Systems |
|-----------|---------|
| BM25 + FAISS retrieval | MLEvolve |
| Hierarchical Cognitive Caching | ML-Master 2.0 |
| Tree summary (structured condensation) | AIDE |
| Evolution Tree + MAP-Elites | LoongFlow, OpenEvolve |
| Island-based program database | OpenEvolve, AlphaEvolve |
| DAG lineage tracking | PiEvolve |
| Git-tagged workspace mutations | A-Evolve |

### 7.3 Code Generation Methods
| Method | Systems |
|--------|---------|
| SEARCH/REPLACE diff patching | MLEvolve, OpenEvolve, AlphaEvolve |
| Full code generation | AIDE, ML-Master |
| Stepwise multi-agent pipeline | MLEvolve, LoongFlow |
| Mixed (adaptive) | MLEvolve |

### 7.4 Common Evaluation Pipeline
All systems: (1) execute generated code in sandboxed environments, (2) capture performance metrics, (3) feed results back into search loop, (4) handle errors/debugging as part of the loop.

---

## 8. Key Distinctions & Tradeoffs

### Graph Search vs. Evolutionary Population
- **Graph Search (MLEvolve, ML-Master):** Preserves full exploration history, enables backtracking. Better for avoiding local optima. MCGS with UCT provides principled explore-exploit balance.
- **Evolutionary (OpenEvolve, AlphaEvolve):** Population-based, naturally parallel. Culling weaker solutions saves memory. MAP-Elites preserves diversity.
- **Hybrid (LoongFlow):** Evolution Tree + MAP-Elites + Boltzmann selection for best of both.

### Reasoning Depth
- **Shallow (OpenEvolve):** LLM generates mutations with minimal structured reasoning. Fitness scores only.
- **Medium (MLEvolve, ML-Master):** Memory-driven reasoning with experience retrieval. Learns from past.
- **Deep (LoongFlow):** Dedicated Planner/Executor/Summary roles. Explicit abductive reasoning about why changes worked/failed.

### Domain Specificity
- **ML-Specific (MLEvolve, ML-Master, AutoKaggle):** Built-in domain knowledge (cold-start models, canonical ML pipeline stages, feature engineering templates).
- **General (AlphaEvolve, OpenEvolve, A-Evolve):** Domain-agnostic. Requires user-provided evaluation functions.
- **Hybrid (LoongFlow):** General framework with specialized agents (General-Evolve + ML-Evolve).

### LLM Backbone
Most top MLE-bench agents converge on **Gemini-3-Pro-Preview**. ML-Master uniquely uses dual-LLM (DeepSeek + GPT-4o). AlphaEvolve uses Gemini Flash + Pro ensemble.

---

## 9. Master Comparison Table

| System | Team | Cat. | Search | Memory | LLM | Open Source | MLE-bench |
|--------|------|------|--------|--------|-----|-------------|-----------|
| **MLEvolve** | InternScience | A | MCGS | BM25+FAISS | Gemini-3-Pro | Yes | 61.33% (12h) |
| **AIDE** | Weco AI | A | Greedy Tree | Tree Summary | Any (OpenAI) | Yes | 17.12% |
| **ML-Master 2.0** | SJTU | A | MCTS | HCC | DeepSeek+GPT-4o | Yes | 56.44% |
| **PiEvolve** | Fractal | A | DAG Search | DAG Lineage | Gemini-3-Pro | No | 61.33% (24h) |
| **LoongFlow** | Baidu | A+B | Evo Tree+MAP-Elites | Evolution Tree | Any | Yes | 77.27%* |
| **CAIR MARS+** | CAIR | A | Reflective Search | Modular | Gemini-3-Pro | No | 62.67% |
| **MLE-STAR** | Google | A | Search+Refinement | External KB | Unknown | No | 44.0% (CAIR) |
| **AutoKaggle** | ByteDance | A | Phase-based | Testing | Any | Yes | N/A |
| **Famou-Agent** | — | A | Unknown | Unknown | Gemini-3-Pro | No | 64.44% |
| **AlphaEvolve** | DeepMind | B | Island Evo+MAP-Elites | Program DB | Gemini Flash+Pro | Cloud | N/A |
| **OpenEvolve** | ASI | B | Island Evo+MAP-Elites | Per-island DB | Any | Yes | N/A |
| **A-Evolve** | Amazon | C | File Mutation | Filesystem+Git | Any | Yes | N/A |
| **AgentEvolver** | — | C | Self-evolving | LLM-driven | Any | Partial | N/A |

*LoongFlow score is on the additional leaderboard (uses test-set feedback).

---

## Key References

1. MLEvolve / AutoMLGen: arXiv:2510.08511
2. InternAgent 1.5: arXiv:2602.08990
3. AIDE: arXiv:2502.13138
4. ML-Master 1.0: arXiv:2506.16499
5. ML-Master 2.0: arXiv:2601.10402
6. AlphaEvolve: arXiv:2506.13131
7. LoongFlow: arXiv:2512.24077
8. OpenEvolve: github.com/algorithmicsuperintelligence/openevolve
9. MLE-STAR: arXiv:2506.15692
10. AutoKaggle: arXiv:2410.20424
11. MLE-bench: arXiv:2410.07095
12. A-Evolve: github.com/A-EVO-Lab/a-evolve
13. AgentEvolver: arXiv:2511.10395
14. AVO: arXiv:2603.24517
15. Reasoning as Gradient: arXiv:2603.01692
16. AIRA-dojo: NeurIPS 2025
