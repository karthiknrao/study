# Experience Memory for Policy Improvement in LLM Agents

## The Core Problem

LLM agents are fundamentally stateless — each inference call has no memory of past interactions. When an agent interacts with an environment (web browser, codebase, game, API), it generates **trajectories**: sequences of observations, actions, and rewards. The naive approach (ReAct) stuffs the entire trajectory into the context window, which (1) hits context limits, (2) becomes expensive, and (3) degrades performance due to "lost in the middle" effects.

Worse, most trajectories are **failures** — GPT-4o succeeds on fewer than 15% of WebArena tasks. These failed trajectories are routinely discarded, wasting the dominant source of collected experience.

Experience memory addresses both problems: **compress past interactions into reusable knowledge** and **convert failures into training signal**.

---

## The Three-Stage Abstraction

A recent survey formalizes the evolution into three stages:

1. **Storage** — Preserve raw trajectories (what happened, when, in what sequence)
2. **Reflection** — Refine trajectories into higher-quality variants or verbal self-critiques
3. **Experience** — Abstract trajectories into transferable knowledge (heuristics, strategies, rules)

Most research lives in stages 2–3. The key question throughout: *how does stored experience actually change the agent's policy (its decision function)?*

---

## Major Approaches, Ordered by Sophistication

### 1. Reflexion — Verbal Reinforcement Learning (NeurIPS 2023)

The foundational work. The agent:
- Executes a task, gets a reward signal (success/failure + environment feedback)
- Generates a **verbal self-reflection** — natural language describing what went wrong and what to do differently
- Stores reflections in an **episodic memory buffer**
- On the next episode, retrieves relevant reflections and injects them into the prompt

**Policy change mechanism:** purely in-context. The LLM weights never change. The "policy" is `π(a|s, memory)` where memory is the set of retrieved reflections. Reflexion formalizes this as iterative policy optimization in natural language space.

Limitations: reflections are ephemeral and task-specific. They don't generalize well across different tasks.

### 2. ExpeL — Experiential Learners (2023)

Builds on Reflexion by extracting **insights** from contrasting successful and failed trajectories:
- Runs multiple attempts per task
- Uses an LLM to compare success vs failure trajectories and extract generalizable insights
- Stores insights in a persistent pool
- At test time, concatenates *all* insights into every prompt

**Policy change:** in-context via injected insights.

Limitation: concatenating everything scales poorly. No relevance filtering. Requires multiple rollouts per task (impractical for real deployment).

### 3. ERL — Experiential Reflective Learning (2026)

Addresses ExpeL's weaknesses:
- Extracts **heuristics** from *single-attempt* trajectories (no retries needed)
- Heuristics capture both effective strategies and failure modes
- At test time, an LLM **scores heuristics for relevance** and injects only the top-k
- Achieves +7.8% over ReAct baseline on Gaia2

Key insight: heuristics provide more transferable abstractions than raw trajectories or few-shot examples. Selective retrieval is essential — dumping everything in context hurts performance.

### 4. CER — Contextual Experience Replay (ACL 2025)

Directly applies the RL experience replay concept to in-context learning:
- Accumulates past experiences into a **dynamic memory buffer**
- Synthesizes experiences into natural language summaries + concrete trajectory examples
- Retrieves relevant experiences for new tasks using **semantic similarity**
- Training-free: works entirely within the context window

**Policy change:** in-context augmentation with retrieved experience summaries.

### 5. MemRL — Runtime RL on Episodic Memory (2026)

The most principled RL-inspired approach. Key innovation: **decouples model weights from memory policy**.

Memory is organized as **Intent-Experience-Utility triplets**:
- **Intent**: what the agent was trying to do
- **Experience**: the actual trajectory
- **Utility**: a learned Q-value reflecting expected usefulness

Two core operations:
- **Two-Phase Retrieval**: Selects experiences based on **learned Q-values** (expected utility) rather than semantic similarity alone. Uses Q-learning to estimate which memories will actually help.
- **Utility-Driven Update**: After each task, updates Q-values via Monte Carlo returns based on actual outcome. Experiences that led to success get boosted; noise gets suppressed.

The MDP is defined over the memory retrieval action space, not the language model's weights. The frozen LLM is the "backbone"; the memory policy is what gets optimized. This avoids catastrophic forgetting while enabling continuous improvement.

### 6. MemPO — Self-Memory Policy Optimization (2026)

Takes a different angle: makes memory management an **intrinsic agent capability** rather than an external module.

The agent has three actions: `<mem>` (compress/organize history), `<think⟩` (reason), `<act>` (use tools). The `<mem>` action proactively compresses and reorganizes past interactions for the next step.

MemPO optimizes the `<mem>` action via RL with a novel advantage estimation:
- Trajectory-level advantages for all tokens
- Additional **memory-level advantages** for `<mem>` tokens, measuring how much the compressed memory helps solve the task (conditional probability of answer given memory content)

Results: 25.98% F1 gain over base model, 67% reduction in token usage.

### 7. AgentHER — Hindsight Experience Replay for Agents (2026)

Adapts the classic HER idea from deep RL: a trajectory that fails goal A is often a correct demonstration for some achievable goal B.

Pipeline:
1. **Classify** failures
2. **Extract** what the trajectory actually achieved
3. **Relabel** the prompt to reflect the achieved goal (with LLM-guided confidence gating)
4. **Package** as SFT/DPO training data

Converts discarded failures into training signal. Matches success-only SFT performance with only 50% of successful demonstrations. Gains compound over iterative redeployment.

### 8. AgentRR — Record & Replay (2025)

A practical engineering approach:
1. **Record** the agent's full interaction trace + internal decision process
2. **Summarize** into structured experiences with multi-level abstraction
3. **Replay** on similar tasks with a **check function** as a trust anchor

Supports user-recorded demonstrations, large-small model collaboration, and privacy-aware execution.

### 9. Trainable Graph Memory — From Experience to Strategy (2025)

Abstracts raw trajectories into a **graph-structured state machine** with nodes as states and edges as decision paths. A reinforcement-based weight optimization procedure estimates the utility of each meta-cognitive strategy based on reward feedback. Provides consistent benefits during RL training, especially for smaller models.

---

## Key Design Dimensions

| Dimension | Options | Tradeoff |
|-----------|---------|----------|
| **What to store** | Raw trajectories, reflections, heuristics, strategies, graph structures | More abstraction = more transfer, less specificity |
| **Retrieval signal** | Semantic similarity, recency, importance, learned Q-values | Semantic similarity is cheap but passive; Q-values are principled but need exploration |
| **Policy update mechanism** | In-context injection, SFT, DPO, RL | In-context is training-free but limited by context window; weight updates are powerful but risk catastrophic forgetting |
| **Experience source** | Successes only, failures only, both, hindsight-relabeled | Failures are the dominant signal; ignoring them wastes 85%+ of data |
| **Single vs multi-attempt** | Per-task retries allowed vs single-attempt only | Multi-attempt enables contrastive learning but impractical in production |
| **Memory granularity** | Per-step, per-episode, per-task, cross-task | Finer = more precise; coarser = more transferable |

---

## The Central Tension: Stability vs Plasticity

This mirrors the classic RL problem. You want the agent to:
- **Adapt** quickly to new environments (plasticity)
- **Retain** knowledge from past environments (stability)
- **Not forget** previously learned skills (catastrophic forgetting)

MemRL resolves this by decoupling: frozen weights = stability, optimized memory policy = plasticity. MemPO resolves it by making memory management an explicit learned skill within the agent's action space.

---

## Practical Takeaways for Building Systems

1. **Start with Reflexion-style verbal feedback** in the prompt. It's the simplest approach and gives measurable gains immediately.

2. **Add relevance-filtered retrieval** (like ERL) rather than dumping all experience into context. Quality > quantity.

3. **Don't discard failures.** Use hindsight relabeling (AgentHER) or reflection-on-failure (ERL) to extract signal from the majority of your trajectory data.

4. **Consider utility-based retrieval** (MemRL's Q-values) over pure semantic similarity once your experience pool grows large. Semantic similarity retrieves things that *look* related, not things that *help*.

5. **If you can do RL training**, MemPO's approach of making memory management an intrinsic action with dedicated advantage estimation is the current frontier — it jointly optimizes memory, reasoning, and tool use.

6. **For production agents where you can't do weight updates**, the training-free path is: record trajectories → reflect/abstract into heuristics → retrieve top-k relevant ones → inject into prompt. This is the ERL/CER recipe.

---

## Sources

- [MemRL: Self-Evolving Agents via Runtime RL on Episodic Memory](https://arxiv.org/abs/2601.03192)
- [ERL: Experiential Reflective Learning for Self-Improving LLM Agents](https://arxiv.org/abs/2603.24639)
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [ExpeL: LLM Agents Are Experiential Learners](https://arxiv.org/abs/2308.10144)
- [CER: Contextual Experience Replay for Self-Improvement of Language Agents](https://aclanthology.org/2025.acl-long.694.pdf)
- [AgentHER: Hindsight Experience Replay for LLM Agent Trajectories](https://arxiv.org/abs/2603.21357)
- [AgentRR: Get Experience from Practice](https://arxiv.org/abs/2505.17716)
- [MemPO: Self-Memory Policy Optimization for Long-Horizon Agents](https://arxiv.org/abs/2603.00680)
- [From Experience to Strategy: Trainable Graph Memory](https://arxiv.org/abs/2511.07800)
- [How Memory Management Impacts LLM Agents](https://arxiv.org/abs/2505.16067)
- [Evo-Memory: Benchmarking Test-time Learning with Self-Evolving Memory](https://www.marktechpost.com/2025/12/02/google-deepmind-researchers-introduce-evo-memory-benchmark-and-remem-framework-for-experience-reuse-in-llm-agents/)
- [Memory for Autonomous LLM Agents: Survey](https://arxiv.org/abs/2603.07670)
- [Prompt Engineering Guide — Reflexion](https://www.promptingguide.ai/techniques/reflexion)
