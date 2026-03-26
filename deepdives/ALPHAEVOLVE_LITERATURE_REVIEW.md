# AlphaEvolve and Similar Methods: A Literature Review

## Table of Contents
1. [AlphaEvolve Overview](#alphaevolve-overview)
2. [Direct Predecessors from DeepMind](#direct-predecessors-from-deepmind)
3. [Open-Source Implementations](#open-source-implementations)
4. [LLM-Guided Evolutionary Methods](#llm-guided-evolutionary-methods)
5. [Scientific Discovery with LLMs](#scientific-discovery-with-llms)
6. [Program Synthesis & Algorithm Discovery](#program-synthesis--algorithm-discovery)
7. [LLM for Reward Design & RL](#llm-for-reward-design--rl)
8. [Key Themes & Comparison](#key-themes--comparison)
9. [References](#references)

---

## AlphaEvolve Overview

**Paper:** *AlphaEvolve: A coding agent for scientific and algorithmic discovery*
**Authors:** Novikov et al. (Google DeepMind)
**Date:** June 2025
**Link:** https://arxiv.org/abs/2506.13131

### Core Concept
AlphaEvolve is an **evolutionary coding agent** that uses LLMs (specifically Gemini models) for general-purpose algorithm discovery and optimization. It pairs:
- **LLM-based code generation** (creative problem-solving)
- **Automated evaluators** (verification and scoring)
- **Evolutionary framework** (iterative improvement)

### Key Innovations
1. **Direct code modification**: LLMs make changes directly to code
2. **Multi-evaluator feedback**: Continuous feedback from automated evaluators
3. **Evolutionary approach**: Maintains population of solutions, evolves over time
4. **General-purpose**: Not domain-specific like AlphaFold or AlphaTensor

### Notable Achievements
- Found procedure to multiply 4×4 complex matrices using **48 scalar multiplications** (first improvement over Strassen's algorithm in 56 years)
- Developed more efficient **data center scheduling algorithm**
- Found **hardware accelerator circuit simplifications**
- Accelerated training of the LLM underpinning AlphaEvolve itself
- Solved open mathematical problems (kissing number problem, etc.)

---

## Direct Predecessors from DeepMind

### 1. FunSearch (2023)
**Paper:** *Mathematical discoveries from program search with large language models*
**Published:** Nature, December 2023
**Link:** https://nature.com/articles/s41586-023-06924-6

**Concept:** Searches for "functions" written in code. Pairs a pretrained LLM with a systematic evaluator.

**Key Differences from AlphaEvolve:**
- More limited scope (primarily mathematical functions)
- Simpler evolutionary mechanism
- Less sophisticated code evolution

**Achievements:**
- New solutions for cap set problem
- Improved bin packing algorithms

### 2. AlphaTensor (2022)
**Paper:** *Discovering faster matrix multiplication algorithms with AlphaTensor*
**Published:** Nature, October 2022
**Link:** https://nature.com/articles/s41586-022-05172-4

**Concept:** Uses **reinforcement learning** (AlphaZero-based) to discover matrix multiplication algorithms by treating it as a game (TensorGame).

**Key Differences:**
- RL-based, not LLM-based
- Domain-specific (matrix multiplication only)
- Uses Monte Carlo Tree Search (MCTS)

---

## Open-Source Implementations

### 1. OpenEvolve
**Repository:** https://github.com/algorithmicsuperintelligence/openevolve
**PyPI:** `pip install openevolve`

**Features:**
- Most advanced open-source evolutionary coding agent
- Distributed architecture
- Multi-language support
- Integration with various LLM providers
- MAP-Elites quality-diversity optimization
- Island model with migration for parallel exploration

**Achievements:**
- Replicated AlphaEvolve's mathematical optimization results
- GPU kernel optimizations with 12.5% performance improvements
- 2.04x speedups on various benchmarks

### 2. CodeEvolve
**Paper:** *CodeEvolve: An open source evolutionary coding agent for algorithmic discovery and optimization*
**Link:** https://arxiv.org/abs/2510.14150

**Features:**
- Open-source implementation inspired by AlphaEvolve
- Proposes new approaches for core components
- Reproducible framework

---

## LLM-Guided Evolutionary Methods

### 1. LLM Guided Evolution (GE)
**Paper:** *LLM Guided Evolution -- The Automation of Models Advancing Models*
**Link:** https://arxiv.org/abs/2403.11446

**Concept:** Uses LLMs to directly modify code for more intelligent, supervised evolutionary process. LLMs guide mutations and crossovers.

### 2. LLaMEA (Large Language Model Evolutionary Algorithm)
**Paper:** *LLaMEA: A Large Language Model Evolutionary Algorithm for Automatically Generating Algorithms*
**Link:** https://ieeexplore.ieee.org/document/10752628
**GitHub:** https://github.com/XAI-liacs/LLaMEA

**Concept:** Framework leveraging GPT models for automated generation and refinement of algorithms given criteria and task definition.

### 3. LEO (Large Language-Model-Based Evolutionary Optimizer)
**Paper:** *Large language model-based evolutionary optimizer: Reasoning with...*
**Link:** https://www.sciencedirect.com/science/article/pii/S0925231224020435

**Concept:** Population-based method for numerical optimization using LLMs. Claims zero-shot optimization across diverse scenarios.

### 4. LLM_GP
**Paper:** *Evolving code with a large language model*
**Link:** https://link.springer.com/article/10.1007/s10710-024-09494-2

**Concept:** General LLM-based evolutionary algorithm designed to evolve code. Uses evolutionary operators that enlist LLMs via prompting.

### 5. EvoTune
**Paper:** *Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning*
**Link:** https://arxiv.org/html/2504.05108v4
**GitHub:** https://github.com/CLAIRE-Labo/EvoTune

**Concept:** Combines evolutionary search over LLM-generated Python programs with RL to fine-tune the LLM search operator based on discovered algorithm performance.

**Key Innovation:** Makes the LLM itself a learnable search operator (not static like FunSearch).

### 6. BAG (Benchmark-Assisted Guided Evolutionary Approach)
**Paper:** *Automated Algorithm Design with LLMs: A Benchmark-Assisted Approach*
**Link:** https://openreview.net/forum?id=ad9BPW5bI6

**Concept:** High-quality benchmark code examples enhance prompt effectiveness for LLM-driven black-box optimization.

---

## Scientific Discovery with LLMs

### 1. LLM-SR (Scientific Equation Discovery)
**Paper:** *LLM-SR: Scientific Equation Discovery via Programming with Large Language Models*
**Venue:** ICLR 2025 (Oral)
**Link:** https://arxiv.org/abs/2404.18400
**GitHub:** https://github.com/deep-symbolic-mathematics/LLM-SR

**Concept:** Uses LLMs to discover scientific equations from data. Treats equations as programs with mathematical operators. Combines LLMs' scientific priors with evolutionary search.

**Achievements:** Outperforms state-of-the-art symbolic regression baselines, especially in out-of-domain settings.

### 2. The AI Scientist
**Organization:** Sakana AI
**Link:** https://sakana.ai/ai-scientist/

**Concept:** Fully automated open-ended scientific discovery. Performs:
- Idea generation
- Literature search
- Experiment planning and execution
- Figure generation
- Manuscript writing
- Automated peer review

### 3. SciAgents
**Paper:** *SciAgents: Automating Scientific Discovery Through Bioinspired Multi-Agent Intelligent Graph Reasoning*
**Link:** https://advanced.onlinetextbook.wiley.com/doi/10.1002/adma.202413523

**Concept:** Multi-agent approach for automated scientific discovery using knowledge graphs.

### 4. CodeScientist
**Organization:** Allen AI
**Link:** https://allenai.org/blog/codescientist

**Concept:** Automated scientific discovery building on AI Scientist, Agent Laboratory, and data-to-paper approaches.

---

## Program Synthesis & Algorithm Discovery

### 1. AlphaResearch
**Paper:** *Accelerating New Algorithm Discovery with Language Models*
**Link:** https://arxiv.org/html/2511.08522v1

**Concept:** Autonomous research agent with dual research approach for discovering new algorithms on open-ended problems.

### 2. Evo-MCTS
**Paper:** *Automated Algorithmic Discovery for Scientific Computing through Evolutionary Monte Carlo Tree Search*
**Link:** https://arxiv.org/html/2508.03661v3

**Concept:** Integrates LLMs with tree-structured Monte Carlo Tree Search for algorithm discovery.

### 3. ResearchCodeAgent
**Paper:** *ResearchCodeAgent: An LLM Multi-Agent System for Automated Codification of Research Methodologies*
**Link:** https://openreview.net/forum?id=XCFHgkaFJZ

**Concept:** Multi-agent system for automating codification of research methodologies from ML literature.

---

## LLM for Reward Design & RL

### 1. Eureka
**Paper:** *Eureka: Human-Level Reward Design via Coding Large Language Models*
**Venue:** ICLR 2024
**Link:** https://arxiv.org/abs/2310.12931
**GitHub:** https://github.com/eureka-research/Eureka

**Concept:** Uses LLMs (GPT-4) for evolutionary optimization over reward code for RL. Takes environment source code as context and generates executable reward functions.

**Key Features:**
- Zero-shot reward function generation
- Evolutionary search over reward programs
- Reward reflection for iterative improvement

**Achievements:** Outperforms human-designed rewards in 83% of tasks with 52% average improvement.

### 2. Text2Reward
**Related to Eureka** - Automated dense reward function generation for RL.

### 3. LaRes
**Paper:** *LaRes: Evolutionary Reinforcement Learning with LLM-based Reward Shaping*
**Link:** https://neurips.cc/virtual/2025/poster/116462

**Concept:** LLMs generate reward function populations to guide RL policy learning.

---

## Key Themes & Comparison

### Core Architecture Pattern
Most AlphaEvolve-like methods share:

```
┌─────────────────────────────────────────────────────────┐
│                    EVOLUTIONARY LOOP                     │
│  ┌─────────┐    ┌─────────┐    ┌─────────────────────┐  │
│  │  LLM    │───▶│  Code   │───▶│    Evaluators       │  │
│  │ Generator│    │ Mutation│    │ (Automated Testing) │  │
│  └─────────┘    └─────────┘    └─────────────────────┘  │
│       ▲                                │                │
│       └────────────────────────────────┘                │
│                    Feedback Loop                         │
└─────────────────────────────────────────────────────────┘
```

### Comparison Table

| Method | Year | Approach | Domain | Code Evolution | Evaluator Type |
|--------|------|----------|--------|----------------|----------------|
| AlphaTensor | 2022 | RL/MCTS | Matrix Mult | No | Game-based |
| FunSearch | 2023 | LLM+EA | Math/CS | Yes | Automated |
| AlphaEvolve | 2025 | LLM+EA | General | Yes | Multi-evaluator |
| Eureka | 2023 | LLM+EA | RL Rewards | Yes | RL Training |
| LLM-SR | 2024 | LLM+EA | Equations | Yes | Data Fitting |
| LLaMEA | 2024 | LLM+EA | Optimization | Yes | Performance |
| OpenEvolve | 2025 | LLM+EA | General | Yes | Configurable |
| EvoTune | 2024 | LLM+EA+RL | General | Yes | Learned |

### Key Design Decisions

1. **Representation**
   - Raw string/code (AlphaEvolve, FunSearch)
   - Abstract syntax trees
   - Domain-specific languages

2. **Mutation Strategy**
   - Diff-based edits (OpenEvolve default)
   - Full program regeneration
   - Guided by LLM context

3. **Selection Mechanism**
   - Tournament selection
   - MAP-Elites (quality-diversity)
   - Island models with migration

4. **Population Management**
   - Single population
   - Island models (OpenEvolve)
   - Archive-based (MAP-Elites)

---

## Resources

### Curated Lists
- **LLM4AlgorithmDesign:** https://github.com/FeiLiu36/LLM4AlgorithmDesign
- **LLM4Opt:** https://github.com/FeiLiu36/LLM4Opt
- **Awesome-LLM-Scientific-Discovery:** https://github.com/HKUST-KnowComp/Awesome-LLM-Scientific-Discovery

### Key Repositories
- AlphaEvolve results: https://github.com/google-deepmind/alphaevolve_results
- FunSearch: https://github.com/google-deepmind/funsearch
- AlphaTensor: https://github.com/google-deepmind/alphatensor
- OpenEvolve: https://github.com/algorithmicsuperintelligence/openevolve
- LLM-SR: https://github.com/deep-symbolic-mathematics/LLM-SR
- EvoTune: https://github.com/CLAIRE-Labo/EvoTune

---

## References

1. Novikov, A., et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv:2506.13131
2. Romera-Paredes, B., et al. (2023). Mathematical discoveries from program search with large language models. Nature.
3. Fawzi, A., et al. (2022). Discovering faster matrix multiplication algorithms with AlphaTensor. Nature.
4. Ma, Y.J., et al. (2024). Eureka: Human-Level Reward Design via Coding Large Language Models. ICLR.
5. Shojaee, P., et al. (2025). LLM-SR: Scientific Equation Discovery via Programming with Large Language Models. ICLR.
6. Liu, F., et al. (2024). LLaMEA: A Large Language Model Evolutionary Algorithm. IEEE TEVC.
7. Chen, A., et al. (2024). Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning. arXiv.
