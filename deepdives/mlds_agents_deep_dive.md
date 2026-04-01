# ML Engineering Agents: Complete Landscape Deep Dive

## MLE-STAR (Google Cloud / KAIST) — The Anchor System

**Paper**: [arxiv.org/abs/2506.15692](https://arxiv.org/abs/2506.15692) | **Venue**: NeurIPS 2025
**Authors**: Jaehyun Nam, Jinsung Yoon, Jiefeng Chen, Jinwoo Shin, Sercan Ö. Arık, Tomas Pfister

**Core Idea**: Most LLM-based MLE agents fail because they (1) rely on internal LLM knowledge (stale/incomplete) and (2) modify entire code at once (shallow exploration). MLE-STAR fixes this with:

1. **Web Search Phase** — Searches Kaggle/forums for effective models and kernels relevant to the task, forming a strong initial solution from real-world code
2. **Targeted Refinement** — Iteratively refines **specific ML pipeline components** (feature engineering, model selection, hyperparams, etc.) guided by **ablation studies** that measure each code block's impact
3. **Novel Ensembling** — Combines the best strategies discovered during refinement

**Results**: **64% medal rate** on MLE-bench Lite (best at time of publication)
**Code**: [Google ADK sample](https://github.com/google/adk-samples/tree/main/python/agents/machine-learning-engineering) | [Unofficial open reimplementation](https://github.com/WalkingDevFlag/MLE-STAR-Open)

---

## The Full Landscape of Similar Systems

### 1. DS-STAR — Data Science Agent (Google Cloud / KAIST)

**Paper**: [arxiv.org/abs/2509.21825](https://arxiv.org/abs/2509.21825) | Same lead author group as MLE-STAR

MLE-STAR's sibling, but for **data science** rather than just ML model building:
- Processes **heterogeneous formats** (CSV, JSON, Markdown, unstructured text)
- Handles **open-ended queries** without ground-truth labels
- Multi-agent framework with **iterative planning and verification**
- Generates comprehensive research reports, not just predictions
- **32+ percentage point** improvement on hard-level DABStep

### 2. ML-Agent — RL-Trained MLE Agent (Shanghai Jiao Tong University)

**Paper**: [arxiv.org/abs/2505.23723](https://arxiv.org/abs/2505.23723) | [GitHub](https://github.com/MASWorks/ML-Agent)

**Key insight**: First to move from prompt-engineering to **learning-based agentic ML**:
- Trains a **7B Qwen-2.5** LLM via online RL — small model beats large prompted ones
- Three innovations: **exploration-enriched fine-tuning**, **step-wise RL**, **agentic ML-specific reward module**
- Models ML engineering as sequential decision-making
- Proves that fine-tuning for the task beats prompt-engineering GPT-4-class models

### 3. AutoMind — Knowledge-Grounded Agent (Zhejiang Univ / Ant Group)

**Paper**: [arxiv.org/abs/2506.10974](https://arxiv.org/abs/2506.10974) | [GitHub](https://github.com/ant-research/automind)

Three-pronged approach that mimics human Kaggle grandmasters:
- **Curated expert knowledge base** — grounds the agent in real domain expertise
- **Agentic knowledgeable tree search** — strategic exploration of solution space
- **Self-adaptive coding** — dynamically adjusts code generation to task difficulty
- **11.0% higher win rate** on MLE-bench vs prior SOTA, with **63% fewer tokens**

### 4. AI Research Agents — Search & Operators (Meta FAIR / UCL)

**Paper**: [arxiv.org/abs/2507.02554](https://arxiv.org/abs/2507.02554)

**Current SOTA on MLE-bench** at **47.7% medal rate** (on full MLE-bench, not Lite):
- Formalizes MLE agents as **search policies + operators** (not monolithic)
- Studies **Greedy, MCTS, and Evolutionary Search** strategies
- Developed enhanced operator set + **AIRA-dojo** framework for scalable experimentation
- Key finding: the **generalization gap** (validation vs test divergence) is a major bottleneck

### 5. MLZero — End-to-End Multi-Agent (Amazon / AutoGluon)

**Paper**: [arxiv.org/abs/2505.13941](https://arxiv.org/abs/2505.13941) | [GitHub](https://github.com/autogluon/autogluon-assistant)

A **9-agent system** for zero-intervention ML automation:
- **Cognitive perception module** transforms raw multimodal inputs into perceptual context
- **Dual memory modules** (semantic + episodic) for iterative improvement
- Covers diverse data modalities (tabular, image, text, multimodal)
- Built on top of AutoGluon, leveraging existing AutoML tooling
- End-to-end: raw data → trained model → predictions, zero human input

### 6. Agent0 — Self-Evolving Agents (UNC / Salesforce / Stanford)

**Paper**: [arxiv.org/abs/2511.16043](https://arxiv.org/abs/2511.16043) | [GitHub](https://github.com/aiming-lab/Agent0)

Takes the concept further — agents that **evolve from scratch**:
- Two agents co-evolve: **Curriculum Agent** (proposes harder tasks) + **Executor Agent** (solves them)
- Zero external human-curated data needed
- Self-reinforcing improvement cycle
- Broader scope than just MLE, but directly applicable

### 7. MLR-Bench / MLR-Agent — Full Research Lifecycle (NUS)

**Paper**: [arxiv.org/abs/2505.19955](https://arxiv.org/abs/2505.19955) | NeurIPS 2025

Goes beyond "build a model" to **conduct ML research**:
- **MLR-Agent**: idea generation → proposal → experimentation → paper writing
- **MLR-Judge**: automated evaluation (high agreement with human experts)
- 201 research tasks from NeurIPS, ICLR, ICML
- The most ambitious scope of any system here

### 8. MLE-Agent — Practical Pairing Tool (MLSysOps)

**GitHub**: [MLSysOps/MLE-agent](https://github.com/MLSysOps/MLE-agent) | [PyPI](https://pypi.org/project/mle-agent/)

A developer-focused **pair programming agent** for ML:
- Autonomous baseline building from task descriptions
- End-to-end Kaggle competition participation
- Arxiv + Papers with Code integration for latest methods
- Practical tool, not a research system

---

## Training Environments & Task Generators

| System | Role | Key Detail |
|--------|------|------------|
| [MLE-Dojo](https://arxiv.org/abs/2505.07782) | **Training gym** for MLE agents | 200+ Kaggle tasks, supports SFT + RL, Gym-style API |
| [MLE-Smith](https://arxiv.org/abs/2510.07307) | **Task generator** at scale | Auto-creates competition-style MLE tasks from raw datasets |
| [AIRA-dojo](https://arxiv.org/abs/2507.02554) | **Experimentation framework** | Scalable operator evaluation for Meta's research agents |

---

## Benchmark Ecosystem

| Benchmark | Origin | Focus | Scale |
|-----------|--------|-------|-------|
| [MLE-bench](https://arxiv.org/abs/2410.07095) | OpenAI | Kaggle-style MLE tasks | 75 competitions |
| [MLE-bench Lite](https://github.com/openai/mle-bench) | OpenAI | Lightweight subset | 45 competitions |
| [MLAgentBench](https://arxiv.org/abs/2310.03302) | Stanford | ML research experimentation | 13 tasks |
| [ML-Bench](https://arxiv.org/abs/2311.09835) | ML-Bench team | Repository-level ML code | 150+ tasks |
| [MLE-Dojo](https://arxiv.org/abs/2505.07782) | GaTech/Stanford | Interactive MLE training | 200+ competitions |
| [MLR-Bench](https://arxiv.org/abs/2505.19955) | NUS | Open-ended ML research | 201 tasks |
| [FML-bench](https://arxiv.org/abs/2510.10472) | — | Automatic ML research | — |

---

## Performance Comparison on MLE-bench

| Agent | Medal Rate | Approach | Backing |
|-------|-----------|----------|---------|
| MLE-STAR | 64% (Lite) | Web search + targeted refinement | Google Cloud |
| Meta AI Research Agents | 47.7% (full) | MCTS/evolutionary search + operators | Meta FAIR |
| AutoMind | +11% over prior SOTA | Knowledge base + tree search + adaptive coding | Ant Group |
| ML-Agent | Competitive | RL-trained 7B model | SJTU |

---

## Key Takeaways Across the Ecosystem

1. **External knowledge > internal LLM knowledge**: MLE-STAR's web search, AutoMind's knowledge base, MLE-Agent's Papers-with-Code integration all beat relying on what the LLM already knows
2. **Targeted refinement > whole-code rewriting**: Modifying specific pipeline components (feature engineering, hyperparams) consistently outperforms rewriting entire solutions
3. **Learning-based > prompt-based**: ML-Agent's 7B RL-trained model beats prompted models 10x its size — suggests fine-tuning for MLE is the future
4. **Search strategy matters**: Meta FAIR showed that HOW you search (MCTS > evolutionary > greedy) is as important as WHAT you search
5. **Training environments are emerging**: MLE-Dojo, AIRA-dojo provide "gyms" — the field is shifting from prompted agents to trained agents
6. **Scope is expanding rapidly**: Build a model (MLE-STAR) → do data science (DS-STAR) → conduct research (MLR-Agent) → evolve autonomously (Agent0)
7. **AutoML is converging with LLM agents**: MLZero (built on AutoGluon) shows traditional AutoML + LLM agents = more capable than either alone
