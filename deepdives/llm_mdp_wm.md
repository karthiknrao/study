# Deep Dive: Using LLMs to Create, Learn, and Distill MDPs

## The Core Problem

A Markov Decision Process (MDP) is defined by the tuple (S, A, T, R, γ) — states, actions, transition function, reward function, and discount factor. Traditionally, designing an MDP requires domain expertise: you need to identify the right state variables, define meaningful actions, model transition dynamics, and craft reward functions. This is labor-intensive and error-prone. The central question: **can LLMs automate some or all of this?**

The research breaks down into several distinct paradigms:

---

## 1. End-to-End MDP Construction from Natural Language

### A-LAMP (NeurIPS 2025 Workshop)
**Paper:** [A-LAMP: Agentic LLM-Based Framework for Automated MDP Modeling and Policy Generation](https://arxiv.org/abs/2512.11270) — Je-Gal, Yi, Lee (2025)

This is the most direct answer to the question. A-LAMP takes free-form natural language task descriptions and **automatically produces a complete MDP formulation plus a trained policy**. It decomposes the pipeline into three verifiable stages:

1. **MDP Formulation** — the LLM translates NL descriptions into formal MDP components (states, actions, rewards, transitions)
2. **Environment Implementation** — generates executable code for the RL environment
3. **Policy Training** — trains and evaluates a policy

Key insight: By decomposing into stages with verification at each step, A-LAMP catches modeling errors that a single-prompt approach would miss. It outperforms single-model LLM approaches across classic control and custom RL domains.

### Automated Generation of MDPs Using Logic Programming and LLMs (IEEE 2025)
**Paper:** [Automated Generation of MDPs Using Logic Programming and LLMs](https://ieeexplore.ieee.org/iel8/7083369/11293803/11297748.pdf)

A related approach using logic programming + LLMs for scalable MDP model generation for real-time policy generation.

---

## 2. LLMs as World Model Builders (Learning Transition Dynamics T)

### WorldCoder (NeurIPS 2024)
**Paper:** [WorldCoder, a Model-Based LLM Agent: Building World Models by Writing Code](https://proceedings.neurips.cc/paper_files/paper/2024/hash/820c61a0cd419163ccbd2c33b268816e-Abstract-Conference.html) — Tang, Key, Ellis (Cornell, NeurIPS 2024)

This is arguably the most elegant approach. WorldCoder:
- **Builds a Python program representing the world model** by interacting with the environment
- The LLM generates code that explains observed interactions (transition dynamics)
- Uses **optimistic exploration** — the agent invents reward functions it optimistically thinks it can achieve, driving exploration
- Plans using the learned code-based world model

Results: **10,000x more sample-efficient than deep RL**, at a fraction of the LLM cost of ReAct-style agents. The code representation means the world model is interpretable, editable, and transferable.

Key insight: Representing world knowledge as *code* (rather than neural network weights) lets the LLM's prior knowledge directly inform the transition model. The code is both a model and an explanation.

### DECKARD (2023)
**Paper:** [DECKARD](https://arxiv.org/pdf/2301.12050) — Nottingham et al.

Built for Minecraft. DECKARD uses an LLM to generate an **Abstract World Model (AWM)** — a DAG of subgoals representing task prerequisites. It operates in two phases:
- **Dream phase**: Uses the LLM-predicted AWM to plan which subgoal to pursue
- **Wake phase**: Executes in the environment, collecting experiences that validate or correct the AWM

The key idea: the LLM hypothesizes the world model, and real interaction *corrects* it — closing the learning-planning loop.

### Web Agents with World Models (2024)
**Paper:** [Web Agents with World Models](https://arxiv.org/abs/2410.13232)

Trains LLMs as world models predicting next observations for web navigation. Uses a **transition-focused observation abstraction** — instead of predicting raw HTML, it predicts natural language descriptions of important state differences. This makes the transition model tractable.

---

## 3. Neural World Models (Learning T from Pixels/Observations)

These aren't LLM-based per se, but are the dominant paradigm for learning MDP transition dynamics from data:

### IRIS (2023) & DIAMOND (NeurIPS 2024 Spotlight)
**Paper:** [DIAMOND: Diffusion for World Modeling](https://arxiv.org/abs/2405.12399) — Alonso et al.

- **IRIS** uses a discrete autoencoder + autoregressive transformer to model environment dynamics
- **DIAMOND** uses **diffusion models** to predict future frames directly — achieving strong results on Atari 100k benchmark

These learn the full transition function T(s'|s,a) from interaction data, then train RL agents *inside* the learned world model. No explicit MDP design needed.

### Agentic World Modeling (2025)
**Paper:** [Agentic World Modeling: Foundations, Capabilities, Laws, and Beyond](https://arxiv.org/html/2604.22748v1)

A comprehensive framework treating world models as the bridge between LLM reasoning and environment dynamics — covering prediction, planning, and causal understanding.

---

## 4. LLM-Guided POMDP Learning (When States Are Partially Observable)

### POMDP Coder (AISTATS 2025)
**Paper:** [LLM-Guided Probabilistic Program Induction for POMDP Model Estimation](https://arxiv.org/abs/2505.02216) — Curtis, Tang (2025)

This is one of the most sophisticated approaches. It:
- Collects experiences in a database
- Uses an LLM to **generate probabilistic programs** for each POMDP component (observation model Z, transition model T, reward model R, initial state distribution b₀)
- Uses Bayesian inference to update beliefs about environment dynamics
- **Closes the learning-planning loop**: plans using the current model, collects more data, refines the model

Key finding: LLM-guided POMDP construction outperforms tabular POMDP learning, behavior cloning, AND direct LLM planning. The LLM's prior knowledge (embedded in its weights) provides an inductive bias that makes learning from limited data practical.

### Tru-POMDP (2025)
**Paper:** [Tru-POMDP](https://tru-pomdp.github.io/)

Treats the LLM as a **generator of beliefs** (not actions) — leveraging GPT-4's commonsense knowledge while keeping decision-making grounded in mathematical rigor. Achieves higher success rates than pure LLM planners in complex robot manipulation tasks with ambiguity and occlusion.

---

## 5. Automated Reward Function Design (Learning R)

### Eureka (NVIDIA)
**Paper:** [Eureka: Human-Level Reward Design via Coding Large Language Models](https://eureka-research.github.io/)

Generates reward functions as executable code from natural language task descriptions. Without task-specific prompting or templates, Eureka generates rewards that **outperform expert human-engineered rewards** on dexterous manipulation tasks.

### Text2Reward (ICLR 2024)
**Paper:** [Text2Reward: Reward Shaping with Language Models for RL](https://proceedings.iclr.cc/paper_files/paper/2024/file/9941833e8327910ef25daeb9005e4748-Paper-Conference.pdf)

A data-free framework that automates generation and shaping of **dense reward functions** from LLMs. Given a goal in natural language, it produces dense reward code — critical for making RL tractable.

### CARD (2025)
**Paper:** [CARD: LLM-Driven Reward Design Framework](https://www.sciencedirect.com/science/article/pii/S0950705125011104)

An iterative framework where a Coder generates reward code and an Evaluator provides dynamic feedback — eliminating human feedback from the reward design loop.

### Progress Functions (NeurIPS 2025)
**Paper:** [Automated Rewards via LLM-Generated Progress Functions](https://openreview.net/forum?id=lvDHfy169r)

LLMs generate "progress functions" that measure task completion, achieving SOTA on the Bi-DexHands benchmark with **20x fewer reward samples**.

---

## 6. LLMs for State/Action Abstraction (Discovering S and A)

### LDSC: Option Discovery (2025)
**Paper:** [LDSC: Option Discovery Using LLM-guided Semantic HRL](https://arxiv.org/abs/2503.19007) — Shek, Tokekar

Uses LLMs to decompose tasks into **subgoals and reusable options** (temporal abstractions). The LLM's semantic understanding helps identify meaningful state abstractions and skill boundaries. Outperforms baselines by 55.9% in average reward.

### LLM-Augmented HRL (Nature Scientific Reports 2025)
**Paper:** [LLMs augmented hierarchical RL with action abstraction](https://www.nature.com/articles/s41598-025-20653-y)

Reformulates the MDP through policy distillation with LLM-guided action abstractions — improving data efficiency by letting the LLM define meaningful high-level actions.

---

## 7. LLMs for Planning Domain Definition (Generating Formal MDP Specs)

### NL → PDDL Generation
Multiple papers tackle generating formal planning models from natural language:

- **[LLMs as Planning Domain Generators](https://ojs.aaai.org/index.php/ICAPS/article/download/31502/33662)** (ICAPS 2024) — generates PDDL domain models from textual descriptions
- **[Leveraging Environment Interaction for Automated PDDL Generation](https://neurips.cc/virtual/2024/poster/95137)** (NeurIPS 2024) — iteratively refines PDDL models using environment feedback
- **[End-to-end Planning with Agentic LLMs and PDDL](https://arxiv.org/html/2512.09629v1)** — converts NL specs to PDDL with time constraints and optimality
- **[l2p library](https://github.com/AI-Planning/l2p)** — open-source tools for LLM-driven PDDL model generation

Key insight: PDDL generation is essentially MDP construction — PDDL defines the states, actions, preconditions, and effects. The LLM acts as a translator from human intent to formal specification.

---

## 8. Distillation: From LLMs to Compact MDP-Based Policies

### MiniLLM (2023/2024)
**Paper:** [MiniLLM: On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)

Uses **inverse reinforcement learning** to distill knowledge from large LLMs into smaller ones. Rather than just mimicking outputs, it learns the *reward function* that the teacher is optimizing — effectively distilling the MDP objective.

### LLM-PD (IEEE 2025)
**Paper:** [LLM-PD: Large Language Model-Driven Policy Distillation](https://ieeexplore.ieee.org/document/11359674)

Three-stage paradigm: policy distillation → imitation pretraining → reinforcement-based self-optimization. Combines LLM semantic reasoning with MARL for enhanced policy quality.

### RLKD (AAAI 2025)
**Paper:** [RLKD: Distilling LLMs' Reasoning via Reinforcement Learning](https://ojs.aaai.org/index.php/AAAI/article/view/40710/44671)

Uses RL to teach a student LLM the *reasoning structure* (not just outputs) of a teacher — measured by a Generative Structure Reward Model.

### KALM (NeurIPS 2024)
**Paper:** [KALM: Knowledgeable Agents by Offline RL from LLMs](https://neurips.cc/virtual/2024/poster/93321)

Bridges the semantic gap between LLM tokens and RL action signals. The LLM is fine-tuned on offline trajectories, then generates trajectories "in imagination" — effectively learning an MDP policy from LLM knowledge.

---

## 9. The Voyager Paradigm: LLM Agents That Implicitly Build MDPs

### Voyager (NeurIPS 2023)
**Paper:** [Voyager: An Open-Ended Embodied Agent with LLMs](https://arxiv.org/abs/2305.16291) — Wang et al. (NVIDIA/Caltech)

Voyager doesn't explicitly construct an MDP, but it implicitly learns one through:
1. **Automatic curriculum** — explores the state space systematically
2. **Ever-growing skill library** — builds reusable action abstractions (the "A" in MDP)
3. **Iterative prompting** — refines its understanding of environment dynamics (the "T")

It reached diamond-level Minecraft 15.3x faster than baselines. The skill library is essentially a learned action space that grows over time.

---

## 10. Additional Key Approaches

### STRIDE: LLM Agent with Value Iteration Tools (2024)
**Paper:** [STRIDE: A Tool-Assisted LLM Agent Framework for Strategic and Interactive Decision-Making](https://arxiv.org/abs/2405.16376) — Li, Yang, Li et al. (Yale, CUHK)

Rather than having the LLM solve MDPs directly (which it does poorly due to math limitations), STRIDE gives the LLM tools like `UpdateMDPModel` (updates reward/transition estimates) and `ValueIteration` (computes optimal policy). The LLM orchestrates when to use these classical RL tools. Key insight: **use the LLM as the orchestrator, not the solver**.

### In-Context Policy Iteration (ICPI) (2022)
**Paper:** [Large Language Models can Implement Policy Iteration](https://arxiv.org/abs/2210.03821) — Brooks, Walls, Lewis, Singh

The foundational paper. An LLM (Codex) with NO prior domain knowledge performs policy iteration entirely in-context. The prompt content IS the locus of learning — ICPI iteratively updates the prompt through trial-and-error interaction. No gradients, no expert demonstrations needed.

### Auto-Formulating Dynamic Programming with LLMs (2025)
**Paper:** [Auto-Formulating Dynamic Programming Problems with LLMs](https://arxiv.org/abs/2507.11737) — Zhou, Yang, Xin et al.

Introduces DP-Bench (first benchmark for textbook-level DP problems) and DPLM, a 7B-parameter specialized LLM that auto-formulates DP/MDP models from natural language. Uses DualReflect synthetic data generation (forward for diversity, backward for reliability).

### LESR: LLM-Empowered State Representation (ICML 2024)
**Paper:** [LLM-Empowered State Representation for RL](https://arxiv.org/abs/2407.13237) — Wang et al.

Uses LLMs to generate state representation code from environment observations. The LLM produces code that extracts task-relevant features from raw observations, directly addressing the state abstraction problem in MDPs.

### LLM4Teach (IJCAI 2024)
**Paper:** [Large Language Model as a Policy Teacher for Training RL Agents](https://arxiv.org/abs/2311.13373)

Distills LLM zero-shot policy capabilities into compact RL agents. The LLM generates demonstrations/advice used to train an efficient student. Open source at `ZJLAB-AMMI/LLM4Teach`.

### RLHF as Token-Level MDP (ICML 2025)
**Paper:** [DPO Meets PPO: Reinforced Token Optimization](https://arxiv.org/abs/2404.18922)

Provides theoretical proof that the token-wise MDP formulation of RLHF (where each token is a state-action pair) is superior to the sentence-level bandit formulation. Most practical RLHF implementations still use the simpler bandit formulation.

---

## Open-Source Tools and Frameworks

| Tool | Stars | What it Does | GitHub |
|------|-------|-------------|--------|
| **Atropos** | 1.1k | LM RL environments, trajectory collection/evaluation | `NousResearch/atropos` |
| **EUREKA** | High | LLM generates reward function code for robotics | `eureka-research/eureka` |
| **WorldCoder** | Available | LLM builds code-based world models | `haotang1995/WorldCoder` |
| **STRIDE** | Available | LLM agent with value iteration tools for MDPs | `cyrilli/STRIDE` |
| **CARD** | Available | LLM-driven reward design with dynamic feedback | `ShengjieSun419/CARD` |
| **LLM4Teach** | 54 | Policy distillation from LLM to RL agents | `ZJLAB-AMMI/LLM4Teach` |
| **Voyager** | High | Open-ended embodied agent with skill library | `minedojo/voyager` |
| **l2p** | Available | LLM-driven PDDL model generation | `AI-Planning/l2p` |
| **AgentTrek** | Available | Web tutorial → agent trajectory synthesis | `xlang-ai/AgentTrek` |

---

## Summary: The Landscape

| Approach | MDP Component(s) | Source | Key Mechanism |
|----------|------------------|--------|---------------|
| **A-LAMP** | All (S,A,T,R) | Natural language | Agentic decomposition with verification |
| **WorldCoder** | T (transitions) | Env interaction | LLM writes code explaining dynamics |
| **POMDP Coder** | T,Z,R,b₀ | Env interaction | LLM generates probabilistic programs |
| **Eureka/Text2Reward** | R (rewards) | NL + env feedback | LLM generates reward code |
| **LDSC** | S,A (abstractions) | NL + env | LLM-driven subgoal/option discovery |
| **NL→PDDL** | S,A,T (formal spec) | Natural language | LLM translates NL to formal planning |
| **DECKARD** | T (abstract) | LLM prior + env correction | Hypothesize-then-verify world model |
| **DIAMOND/IRIS** | T (from pixels) | Interaction data | Neural world models (diffusion/transformer) |
| **MiniLLM/RLKD** | π (policy) | Teacher LLM trajectories | Distill LLM reasoning into smaller policy |
| **STRIDE** | T,R (estimation) | Env interaction | LLM orchestrates value iteration tools |
| **ICPI** | π (in-context) | Trial-and-error | Policy iteration via prompt updates |
| **Auto-DP** | All (S,A,T,R) | Natural language | Specialized 7B LLM auto-formulates DP/MDP |
| **LESR** | S (state repr.) | Raw observations | LLM generates feature extraction code |
| **RLHF (token MDP)** | R,π | Human feedback | Token-level MDP for alignment |

### Key Takeaways

1. **LLMs are most impactful at the "interface layer"** — translating human intent (NL) into formal MDP components. This is where their world knowledge provides inductive bias that pure data-driven methods lack.

2. **Code generation is the killer mechanism.** Whether it's WorldCoder's Python world models, Eureka's reward functions, or POMDP Coder's probabilistic programs — generating executable code is more effective than trying to learn continuous function approximators when the LLM has relevant prior knowledge.

3. **The dream-wake cycle is emerging as a pattern.** DECKARD, WorldCoder, and POMDP Coder all use a variant of: LLM hypothesizes model → agent acts in environment → observations correct/refine the model. This is fundamentally different from either pure LLM planning or pure RL.

4. **For distillation**, the trend is moving from output mimicry toward structure-aware RL — learning the reasoning process and reward structure, not just the surface-level outputs.

5. **The open problem** is learning compact, verified MDPs from LLMs for truly complex, high-dimensional environments. Current successes are in structured domains (gridworlds, Minecraft, web navigation, robot manipulation with known objects). Scaling to open-ended real-world domains remains unsolved.
