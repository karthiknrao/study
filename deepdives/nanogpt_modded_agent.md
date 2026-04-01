# Papers Using Agents to Replicate Modded-NanoGPT Improvements

## 1. The Automated LLM Speedrunning Benchmark (Meta FAIR / U. Edinburgh) — The Main One

**Paper**: [arxiv.org/abs/2506.22419](https://arxiv.org/abs/2506.22419) | **Venue**: NeurIPS 2025
**Code**: [github.com/facebookresearch/llm-speedrunner](https://github.com/facebookresearch/llm-speedrunner)

This is **the** paper that directly turns the modded-nanoGPT speedrun into an agent benchmark. It evaluates whether LLM agents can **sequentially reproduce** the cumulative improvements from the NanoGPT speedrun (training GPT-2 124M in the shortest time).

**Setup**:
- Agents are given a description of each innovation from the speedrun and must implement it
- Uses **Fraction of Speedup Recovered (FSR)** as the metric — how much of the human-achieved speedup can the agent reproduce
- Tests agents under different hint levels (no hints → pseudocode → detailed descriptions)
- Uses **AIDE** and **Multi-AIDE** as search scaffolds for iterative improvement

**Key findings**:
- Even recent reasoning LLMs **struggle** to re-implement known improvements
- Large gaps remain between AI agents and human researchers in reproducing innovations
- Giving pseudocode helps but doesn't close the gap
- Sequential reproduction (each improvement builds on the last) is especially challenging

---

## 2. AIDE: AI-Driven Exploration in the Space of Code (WecoAI / Meta)

**Paper**: [arxiv.org/abs/2502.13138](https://arxiv.org/abs/2502.13138) | [GitHub](https://github.com/WecoAI/aideml) | [aide.ml](https://www.aide.ml/)

The **agent scaffold used in the LLM Speedrunning Benchmark** above. Not nanoGPT-specific but directly applied to it:
- LLM-based agent that drafts, debugs, and refines ML solutions in a **tree-structured search**
- Reuses promising approaches and quickly discards failing ones
- SOTA on Kaggle evaluations, OpenAI MLE-bench, and METR RE-bench
- The primary search scaffold used in the nanoGPT speedrun reproduction experiments

---

## 3. PACEvolve: Long-Horizon Progress-Aware Consistent Evolution

**Paper**: [arxiv.org/abs/2601.10657](https://arxiv.org/abs/2601.10657)

**Surpassed the human record on Modded NanoGPT** — the only system to do so autonomously:
- An evolutionary search framework using LLMs as operators
- Specifically designed for **long-horizon self-improvement** (iterative optimization over many steps)
- Progress-aware mechanism that maintains consistent improvement trajectory
- SOTA on LLM-SR and KernelBench, and **beat the modded-nanoGPT record**
- This is arguably the most impressive result — an agent didn't just replicate, it **surpassed** human achievements

---

## 4. RE-Bench (METR)

**Paper**: [arxiv.org/abs/2411.15114](https://arxiv.org/abs/2411.15114) | [GitHub](https://github.com/METR/RE-Bench)

Not nanoGPT-specific, but one of the 7 benchmark tasks is **directly based on the nanoGPT speedrun**:
- 7 challenging ML research engineering environments
- Includes 71 human expert 8-hour attempts as baselines
- Evaluates frontier models (Claude 3.5 Sonnet, o1-preview) against humans
- One task: optimize a GPT-2 training setup (inspired by modded-nanoGPT)

---

## 5. PaperBench (OpenAI)

**Paper**: [arxiv.org/abs/2504.01848](https://arxiv.org/abs/2504.01848)

Broader than nanoGPT, but tests the same core capability — **can agents replicate AI research from scratch**:
- Agents must replicate 20 ICML 2024 Spotlight/Oral papers end-to-end
- Includes understanding papers, developing codebases, running experiments
- Hierarchical rubrics with thousands of gradable sub-tasks
- Relevant because modded-nanoGPT improvements are exactly the kind of research this benchmark evaluates

---

## 6. CORE-Bench (Princeton)

**Paper**: [arxiv.org/abs/2409.11363](https://arxiv.org/abs/2409.11363) | [GitHub](https://github.com/siegelz/core-bench)

Evaluates **computational reproducibility** — a prerequisite for replicating modded-nanoGPT:
- 270 tasks from 90 scientific papers across CS, social science, medicine
- Measures ability to reproduce results using provided code and data
- Simpler than the LLM Speedrunning Benchmark (reproduce with code vs. implement from description)
- Shows even basic reproduction is surprisingly hard for current agents

---

## Summary

| Paper | Relationship to modded-nanoGPT | Key Result |
|-------|-------------------------------|------------|
| [LLM Speedrunning Benchmark](https://arxiv.org/abs/2506.22419) | Directly benchmarks agent reproduction of nanoGPT speedrun improvements | Agents struggle even with hints |
| [AIDE](https://arxiv.org/abs/2502.13138) | The agent scaffold used in the speedrun benchmark | Tree-structured search for ML solutions |
| [PACEvolve](https://arxiv.org/abs/2601.10657) | **Beat the human modded-nanoGPT record** autonomously | Long-horizon evolutionary self-improvement |
| [RE-Bench](https://arxiv.org/abs/2411.15114) | Includes nanoGPT optimization as one of 7 tasks | Humans still outperform agents at 8h time scale |
| [PaperBench](https://arxiv.org/abs/2504.01848) | Tests same replication capability, broader scope | Agents can't yet replicate ICML papers from scratch |
| [CORE-Bench](https://arxiv.org/abs/2409.11363) | Tests computational reproducibility (simpler prerequisite) | Even basic reproduction is hard |

The standout result is **PACEvolve** — it's the only system that didn't just replicate but actually **surpassed** human achievements on modded-nanoGPT. The LLM Speedrunning Benchmark from Meta is the most systematic evaluation of agent reproduction capability on this specific task.
