# Search Algorithms and Reinforcement Learning for LLM Fine-Tuning: A Deep Research Report

**Date**: May 2026
**Scope**: 2024-2026 papers, methods, and implementations at the intersection of search algorithms, tree search, and RL for LLM reasoning/fine-tuning.

---

## Table of Contents

1. [DeepSeek-R1: Pure RL for Reasoning](#1-deepseek-r1)
2. [Process Reward Models (PRM) and Tree Search](#2-process-reward-models-and-tree-search)
3. [AlphaGo-Style Approaches Applied to LLMs](#3-alphago-style-approaches-applied-to-llms)
4. [Tree Search Rollouts for Training Data Generation](#4-tree-search-rollouts-for-training-data)
5. [Self-Play and Self-Improvement Loops](#5-self-play-and-self-improvement-loops)
6. [GRPO, PPO, DAPO: RL Method Comparison](#6-grpo-ppo-dapo-comparison)
7. [Search-Guided RL for Tool Use and Retrieval](#7-search-guided-rl-for-tool-use)
8. [Key Insights and Open Problems](#8-key-insights-and-open-problems)
9. [Complete Paper Index](#9-complete-paper-index)

---

## 1. DeepSeek-R1: Pure RL for Reasoning

### The Breakthrough

**DeepSeek-R1** (Jan 2025, published in Nature Sep 2025) demonstrated that reasoning abilities in LLMs can be incentivized through **pure reinforcement learning**, without human-labeled reasoning trajectories. This was a landmark result.

**Paper**: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)

### Key Variants

| Model | Training Method | Notes |
|---|---|---|
| **DeepSeek-R1-Zero** | Pure RL (GRPO + rule-based rewards) | No SFT warmup; reasoning emerges spontaneously |
| **DeepSeek-R1** | SFT + RLVR + RLHF (alternating) | Best performance; flagship model |
| **DeepSeek-Distill** | SFT only (distilled from R1 outputs) | No RL for reasoning; for smaller deployable models |

### Core Technical Details

- **Algorithm**: GRPO (Group Relative Policy Optimization) -- a variant of PPO that eliminates the critic/value model by using group-relative advantages
- **Rewards**: Rule-based, not neural reward models:
  - **Accuracy rewards**: Binary correct/incorrect from symbolic verifiers (calculators for math, compilers for code)
  - **Format rewards**: Enforces `<think/>` tags around reasoning
- **Deliberately avoided** process reward models (PRMs) due to reward hacking concerns at scale
- **RLVR** (Reinforcement Learning with Verifiable Rewards): Replaces the expensive reward model with deterministic verifiers

### Emergent Behaviors

The paper documents spontaneous emergence of:
- **Self-reflection**: The model revisits and corrects its own reasoning steps
- **Verification**: The model checks its intermediate results
- **Dynamic strategy adaptation**: The model switches approaches mid-problem
- **"Aha moments"**: Sudden jumps in reasoning capability during training

### Key Finding: SFT + RL > Pure RL

Despite R1-Zero's impressive results, the full R1 model (with SFT warmup) significantly outperforms the pure RL variant. This is a crucial insight: **RL amplifies reasoning but benefits enormously from a strong SFT foundation**.

### Connection to Search

While DeepSeek-R1 does not explicitly use tree search during training, the paper references Monte Carlo tree search and proof assistant feedback as related approaches. The GRPO algorithm itself can be seen as a form of **implicit search**: by sampling multiple completions per prompt (the "group" in GRPO), it explores a solution space and selects based on reward signals -- a softer version of best-of-N search.

---

## 2. Process Reward Models (PRM) and Tree Search

### ORM vs PRM: The Fundamental Distinction

- **Outcome Reward Models (ORM)**: Score the final answer only. Simple but provides no signal about *where* reasoning went wrong.
- **Process Reward Models (PRM)**: Score *each step* of the reasoning chain. Provides granular feedback that can guide tree search.

### Key Papers

#### OmegaPRM (Google, Jun 2024)
**Paper**: [Improve Mathematical Reasoning in Language Models by Automated Process Supervision](https://arxiv.org/abs/2406.06592)

- Proposes a **divide-and-conquer MCTS algorithm** for automated process supervision data collection
- Uses binary search within MCTS to swiftly identify the **first error** in a chain-of-thought
- Collected **1.5 million** process supervision annotations fully automatically
- Trained PRMs on this data, achieving:
  - Gemini Pro: 51% -> 69.4% on MATH500
  - Gemma2 27B: 42.3% -> 58.2% on MATH500
- **Key insight**: Human annotation for PRMs is prohibitively expensive; MCTS can automate the entire pipeline

#### ReST-MCTS* (Tsinghua/THUDM, Jun 2024, NeurIPS 2024)
**Paper**: [ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search](https://arxiv.org/abs/2406.03816)
**Code**: [github.com/THUDM/ReST-MCTS](https://github.com/THUDM/ReST-MCTS)

- **Core idea**: Use MCTS guided by a PRM to collect high-quality reasoning traces, then use those traces for self-training
- Given oracle final answers, MCTS infers per-step process rewards by estimating the probability each step leads to the correct answer
- **Iterative improvement loop**:
  1. Use MCTS + PRM to search for high-quality reasoning traces
  2. Use those traces to train a better policy model
  3. Use the trace values to train a better PRM
  4. Repeat
- Outperforms ReST-EM, Self-Rewarding LM, and vanilla Best-of-N
- **Key insight**: Tree search rollouts serve double duty -- they produce better training data *and* better reward signals

#### ThinkPRM (Apr 2025)
**Paper**: [Process Reward Models That Think](https://arxiv.org/abs/2504.16828)

- Treats the PRM itself as a "thinking" model that verifies steps via chain-of-thought
- A 1.5B ThinkPRM model achieves strong performance on MATH-500 under guided beam search
- Demonstrates that PRMs can be made more capable by giving them computation time

#### Boosting Policy and PRM with MCTS (ACL Findings 2025)
**Paper**: [Boosting Policy and Process Reward Models with Monte Carlo Tree Search](https://aclanthology.org/2025.findings-acl.388.pdf)

- Directly combines MCTS with PRM scoring for both training and inference
- Shows MCTS + PRM outperforms vanilla beam search (where candidate scoring uses prior probability rather than a reward model)

#### AgentPRM (Feb 2025)
**Paper**: [Process Reward Models for LLM Agents: Practical Framework](https://arxiv.org/abs/2502.10325)

- Extends PRM framework to LLM agents that interact with environments
- Focuses on continual improvement through agent-environment interactions

### How PRMs Connect to Tree Search

The connection is fundamental and works in both directions:

1. **PRM guides tree search** (inference): During MCTS or beam search, the PRM scores each node/step, providing the value function that guides exploration. This is directly analogous to AlphaGo's value network guiding MCTS rollouts.

2. **Tree search trains PRMs** (training): MCTS rollouts generate per-step reward annotations by backpropagating final-answer correctness through the search tree. This creates training data for PRMs without human annotation.

3. **Tree search generates training data** (distillation): The best paths found by PRM-guided MCTS become high-quality SFT training data for the policy model.

---

## 3. AlphaGo-Style Approaches Applied to LLMs

### The AlphaGo-LLM Connection

The analogy between AlphaGo and LLM reasoning is one of the most productive conceptual frameworks in recent research:

| AlphaGo Component | LLM Analog |
|---|---|
| Board position | Current reasoning state (partial CoT) |
| Legal moves | Next reasoning step / token |
| Game outcome (win/loss) | Final answer correct/incorrect |
| MCTS | Tree search over reasoning paths |
| Policy network | The LLM itself (generates candidate steps) |
| Value network | PRM (estimates probability of reaching correct answer) |
| Self-play | Model generates its own training problems and solutions |

### Key Papers and Implementations

#### AlphaZero-Like Tree Search for LLM Decoding and Training (NeurIPS 2024)
**Paper**: [Improving Chain-of-Thought Reasoning in LLMs with Refined Values](https://papers.nips.cc/paper_files/paper/2024/file/00d80722b756de0166523a87805dd00f-Paper-Conference.pdf)

- Proposes using AlphaZero-style tree search to guide both LLM decoding and training
- Introduces a value refinement process that improves the quality of the value estimates used during search

#### OmegaPRM as AlphaGo Zero Analog (Google, Jun 2024)
- OmegaPRM's MCTS algorithm is explicitly inspired by **AlphaGo Zero** (Silver et al.)
- The divide-and-conquer search for error detection mirrors AlphaGo Zero's self-play for discovering optimal moves
- The automated data collection loop (search -> annotate -> train PRM -> better search) mirrors the AlphaGo Zero self-improvement cycle

#### KBQA-o1: AlphaGo-Style Search for Knowledge Base QA (ICML 2025)
**Paper**: [KBQA-o1: Agentic Knowledge Base Question Answering with Monte Carlo Tree Search](https://icml.cc/virtual/2025/poster/45313)

- Uses AlphaGo-style MCTS to guide step-by-step knowledge base exploration
- The model learns when and how to traverse a knowledge graph during reasoning

#### RASPberry: Retrieval-Augmented MCTS Self-Play (ACL Findings 2025)
**Paper**: [RASPberry: Retrieval-Augmented Monte Carlo Tree Self-Play](https://aclanthology.org/2025.findings-acl.587.pdf)

- Combines retrieval augmentation with MCTS in a self-play framework
- Uses PRM-guided search to improve both retrieval and reasoning

#### Survey: Large Reasoning Models with Self-Play Deep RL (ACM 2025)
**Paper**: [A Survey on Large Reasoning Models with Self-Play Deep Reinforcement Learning](https://dl.acm.org/doi/full/10.1145/3784013.3784042)

- Comprehensive survey covering the integration of self-play deep RL with chain-of-thought generation
- Covers preference trees, prospective reasoning path planning via MCTS, and iterative self-improvement

### Why the Analogy Works (and Where It Breaks Down)

**Works**:
- Both involve sequential decision-making under uncertainty
- Both benefit from search at test time (MCTS improves both Go play and LLM reasoning)
- Both can use self-play / self-generated data for improvement
- Value functions / PRMs serve the same role as Go value networks

**Breaks down**:
- LLM reasoning spaces are vastly larger and more open-ended than Go's 19x19 board
- Correctness verification is often harder (no clear "win condition" for open-ended reasoning)
- The "opponent" in LLM self-play is the difficulty of the problem, not an adversarial agent
- Go has perfect information; LLM reasoning often deals with ambiguity

---

## 4. Tree Search Rollouts for Training Data Generation

This is one of the most practically impactful areas: using tree search at training time to create better SFT/RL data.

### ReST-MCTS* (NeurIPS 2024) -- The Blueprint

**Paper**: [ReST-MCTS*](https://arxiv.org/abs/2406.03816)

The canonical example of using tree search rollouts for training:

1. Start with a base policy model and seed PRM
2. For each training problem with a known correct answer:
   - Run MCTS guided by the PRM to explore reasoning paths
   - Backpropagate the final correctness signal through the tree
   - Infer per-step process rewards
3. Extract the best traces as SFT data for the policy model
4. Use the per-step reward annotations to train a better PRM
5. Iterate: improved policy + improved PRM -> better search -> better data

**Results**: Continuously improves across iterations; outperforms ReST-EM, Self-Rewarding LM, Best-of-N, and Tree-of-Thought baselines.

### Enhancing LLM Reasoning with Reward-Guided Tree Search (Nov 2024)
**Paper**: [Enhancing LLM Reasoning with Reward-guided Tree Search](https://arxiv.org/abs/2411.11694)

- Uses reward-guided tree search during inference to generate high-quality traces
- These traces are then distilled back into the model via SFT
- Creates a positive feedback loop: better model -> better search -> better training data -> even better model

### Best-of-N as Implicit Search

A simpler but widely used approach:
- Sample N completions from the model
- Score each with a reward model or verifier
- Use only the highest-scoring ones as training data

This is "flat" search (no tree structure) but serves the same purpose. Papers like **Exploring the Limit of Outcome Reward** (Feb 2025, [arxiv](https://arxiv.org/abs/2502.06781)) show that even this simple approach, combined with behavior cloning on positive examples, is theoretically sufficient to optimize the policy.

### Key Insight: Search Quality Caps Training Data Quality

The quality of training data generated by tree search is directly limited by:
1. The quality of the PRM / value function guiding the search
2. The search budget (how many nodes are explored)
3. The base model's ability to generate plausible candidates

This creates a chicken-and-egg problem that iterative methods (like ReST-MCTS*) solve through bootstrapping.

---

## 5. Self-Play and Self-Improvement Loops

### Absolute Zero Reasoner (May 2025, NeurIPS 2025)
**Paper**: [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2505.03335)
**Code**: [github.com/LeapLabTHU/Absolute-Zero-Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner)

Perhaps the most radical self-improvement framework proposed to date:
- A single model learns to **propose tasks** that maximize its own learning progress
- It then solves those tasks, using a code executor as the unified reward source
- **Zero external data** required -- the model creates its own curriculum
- Achieves SOTA on coding and math reasoning despite being trained without any external data
- Compatible with different model scales and classes

### SPIN: Self-Play Fine-Tuning (ICLR 2024)
**Paper**: [Self-play fine-tuning converts weak language models to strong language models](https://dl.acm.org/doi/10.5555/3692070.3692326)

- Frames LLM fine-tuning as a two-player game between the model and its previous version
- Progressively elevates the LLM by having it compete against itself
- Unlocks the full potential of human-annotated demonstration data for SFT

### SeRL: Self-play RL for Large Reasoning Models (NeurIPS 2025)
**Paper**: [SeRL: Self-play Reinforcement Learning for Large Reasoning Models](https://neurips.cc/virtual/2025/poster/117365)

- Demonstrates effectiveness of self-play RL for improving reasoning
- Combines self-play with RLVR for scalable reasoning improvement

### Evolving Self-Play Critic via Adversarial Games (NeurIPS 2025)
**Paper**: [Evolving Self-Play Critic via Adversarial Games for LLM Reasoning](https://neurips.cc/virtual/2025/poster/118706)

- Uses adversarial self-play to train a critic that can guide test-time search
- The critic significantly improves mathematical reasoning on MATH500 when used to guide diverse LLMs

### AceSearcher: Cooperative Self-Play (NeurIPS 2025)
**Paper**: [AceSearcher: Bootstrapping Reasoning and Search for LLMs via Cooperative Self-Play](https://neurips.cc/virtual/2025/poster/116458)

- Trains a single LLM to alternate between two roles: reasoner and searcher
- The reasoner generates solutions; the searcher finds relevant information
- Through cooperative self-play, both capabilities improve simultaneously

### SAGE: Multi-Agent Self-Evolution (2025)
**Paper**: [SAGE: Multi-Agent Self-Evolution for LLM Reasoning](https://arxiv.org/abs/2603.15255)

- Extends self-play to multi-agent settings
- References SPIRAL framework showing self-play on turn-based games improves general reasoning

### The Self-Improvement Pattern

These methods share a common architecture:

```
while not converged:
    1. Generate tasks/problems (or curate them)
    2. Attempt solutions (with optional search)
    3. Verify solutions (code executor, symbolic verifier, etc.)
    4. Use verified solutions as training data
    5. Update model (via SFT, RL, or both)
    6. Optionally: update the task generator, reward model, or verifier
```

The key variables are:
- Who generates the tasks? (Humans, the model itself, a separate task model)
- How are solutions verified? (Code execution, symbolic math, learned reward model)
- How is the model updated? (SFT on correct traces, RL with rewards, DPO on preference pairs)

---

## 6. GRPO, PPO, DAPO: RL Method Comparison

### The Evolution of RL for LLM Reasoning

```
PPO (2017) --> GRPO (2024, DeepSeekMath) --> DAPO (2025, ByteDance)
```

### PPO (Proximal Policy Optimization)
**Requires**: Policy model + Reward model + Critic (value model) + Reference model

- The original RLHF workhorse
- Clips policy updates to prevent destabilizing changes
- KL penalty keeps the policy close to the reference
- **Downside for LLM reasoning**: Requires 4 full-size models in memory simultaneously

### GRPO (Group Relative Policy Optimization)
**Introduced in**: [DeepSeekMath (2024)](https://arxiv.org/abs/2402.03300)
**Requires**: Policy model + Reference model (no critic, no reward model with RLVR)

Key innovations:
- **Eliminates the critic**: Instead of training a separate value model, GRPO samples multiple completions per prompt and computes advantages relative to the group mean
- **Group advantage**: `advantage = (reward - mean(group_rewards)) / std(group_rewards)`
- Combined with RLVR, eliminates both the reward model and critic

**Known issues** (identified by subsequent research):
- **Length bias**: Dividing advantage by response length means long incorrect answers get smaller penalties, encouraging verbose wrong answers
- **Difficulty bias**: Normalizing by per-question reward std overweights easy/hard questions with low variance

### DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)
**Paper**: [DAPO: An Open-Source LLM RL System at Scale](https://arxiv.org/abs/2503.14476)
**Code**: [github.com/BytedTsinghua-SIA/DAPO](https://github.com/BytedTsinghua-SIA/DAPO)

Improvements over GRPO:
1. **Clip-higher**: Increases the upper bound of the PPO clipping range to encourage exploration and prevent entropy collapse
2. **Dynamic sampling**: Filters out prompts where all responses are correct or all wrong (though the Dec 2025 comparative study found this doesn't help in practice)
3. **Token-level policy gradient loss**: Moves from sample-level to token-level loss, so longer responses have proportionally more influence
4. **Overlong reward shaping**: Soft penalty for truncated responses

### Comparative Analysis (Dec 2025)
**Paper**: [Comparative Analysis and Parametric Tuning of PPO, GRPO, and DAPO](https://arxiv.org/abs/2512.07611)

Key findings from controlled comparison:
- **All RL methods outperform their base models** on reasoning benchmarks
- **Increasing group size** in GRPO/DAPO leads to more stable training and higher accuracy
- **KL penalty impact is non-monotonic** -- too much or too little hurts
- **DAPO without Dynamic Sampling achieves the best results** (contradicting the original DAPO paper's claim)
- PPO remains competitive, especially with proper tuning

### How These Compare to Search-Based Approaches

| Aspect | RL Methods (PPO/GRPO/DAPO) | Search Methods (MCTS/Beam Search) |
|---|---|---|
| **When applied** | Training time | Inference time (or both) |
| **Computational cost** | High (model updates) | High (multiple rollouts) |
| **Training data needed** | Prompts + verifiers | Known answers + PRM |
| **Exploration mechanism** | Policy sampling | Tree expansion |
| **Reward signal** | Outcome or process | Process (per-step) |
| **Scalability** | Proven at scale (DeepSeek-R1) | Proven for inference (AlphaGo) |
| **Combination** | Can use search-generated data for SFT before RL | Can use RL-trained models as better search policies |

**The emerging consensus**: The most powerful approaches **combine both**. Tree search generates high-quality training data, which is used for SFT. Then RL (GRPO/DAPO) further optimizes the model. The cycle repeats.

---

## 7. Search-Guided RL for Tool Use and Retrieval

### R1-Searcher (Mar 2025)
**Paper**: [R1-Searcher: Incentivizing the Search Capability in LLMs via RL](https://arxiv.org/abs/2503.05592)

- Teaches reasoning models to use **external search systems** during reasoning
- Two-stage RL: first learn search format, then learn when/how to search
- Improves performance on knowledge-intensive tasks that require current information

### ReSearch (Mar 2025)
**Paper**: [ReSearch: Learning to Reason with Search for LLMs via RL](https://arxiv.org/abs/2503.19470)

- Integrates search directly into the reasoning chain
- End-to-end RL training without supervised data on reasoning steps
- Emergent behaviors: self-correction of incorrect queries, reflection on search results

### Reward-Guided Speculative Decoding (ICML 2025)
**Paper**: [Reward-Guided Speculative Decoding for Efficient LLM Reasoning](https://icml.cc/virtual/2025/poster/46166)

- Uses reward models to guide speculative decoding for more efficient inference
- Combines the speed of speculative decoding with the quality of reward-guided search

---

## 8. Key Insights and Open Problems

### Established Insights

1. **RL with verifiable rewards works**: DeepSeek-R1 proved that rule-based rewards (calculators, compilers) are sufficient to induce reasoning, eliminating the need for expensive reward models.

2. **SFT + RL >> pure RL**: The full DeepSeek-R1 (with SFT warmup) dramatically outperforms R1-Zero (pure RL). Distillation followed by RL is even better.

3. **Process supervision matters**: Per-step rewards from PRMs provide richer training signal than outcome-only rewards, especially for long reasoning chains.

4. **Tree search bootstraps both data and rewards**: MCTS guided by PRMs generates both high-quality SFT traces and per-step reward annotations, creating a virtuous cycle.

5. **Reasoning may emerge from pre-training**: Recent evidence (Understanding R1-Zero-Like Training, Apr 2025) suggests that self-reflection and "aha moments" may already be present in base models from pre-training on CoT data, not solely created by RL.

6. **Length bias is real and problematic**: Both PPO and GRPO inadvertently encourage verbose responses. Explicit length control (LCPO, Dr. GRPO) helps.

7. **Self-play can work without external data**: Absolute Zero shows models can generate their own curricula and improve reasoning with zero human data.

### Open Problems

1. **Search efficiency at scale**: MCTS over reasoning steps is expensive. How to make it practical for very large models and long reasoning chains?

2. **Generalization beyond math/code**: Most PRM and search work focuses on verifiable domains. How to extend to open-ended reasoning where correctness is ambiguous?

3. **Search-training integration**: What is the optimal allocation of compute between search (inference) and training? When should you search more vs. train more?

4. **Reward hacking**: DeepSeek explicitly avoided PRMs due to reward hacking at scale. How to build robust PRMs that don't get exploited?

5. **Self-play stability**: Self-improvement loops can be unstable or converge to degenerate solutions. What guarantees can we provide?

6. **The "pre-training already does it" question**: If reasoning behaviors emerge from pre-training, what exactly does RL add? Is RL just eliciting latent capabilities?

7. **Token-level vs. step-level search**: Should search operate over tokens (like AlphaGo over board positions) or over reasoning steps? The token-level approach is more general but much more expensive.

---

## 9. Complete Paper Index

### Core Papers

| Paper | Date | Venue | Key Contribution |
|---|---|---|---|
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | Jan 2025 | Nature (Sep 2025) | Pure RL induces reasoning; GRPO + RLVR |
| [ReST-MCTS*](https://arxiv.org/abs/2406.03816) | Jun 2024 | NeurIPS 2024 | PRM-guided MCTS for self-training |
| [OmegaPRM](https://arxiv.org/abs/2406.06592) | Jun 2024 | -- | Divide-and-conquer MCTS for automated PRM data |
| [Absolute Zero / AZR](https://arxiv.org/abs/2505.03335) | May 2025 | NeurIPS 2025 | Self-play reasoning with zero external data |
| [DAPO](https://arxiv.org/abs/2503.14476) | Mar 2025 | NeurIPS 2025 | Improved GRPO with clip-higher, token-level loss |

### RL Algorithm Comparisons

| Paper | Date | Key Finding |
|---|---|---|
| [Comparative Analysis: PPO, GRPO, DAPO](https://arxiv.org/abs/2512.07611) | Dec 2025 | DAPO without dynamic sampling is best; group size matters; KL non-monotonic |
| [Open-Reasoner-Zero](https://arxiv.org/abs/2503.24290) | Mar 2025 | Vanilla PPO + binary reward sufficient; outperforms R1-Zero with 1/10 steps |
| [Dr. GRPO](https://arxiv.org/abs/2503.20783) | Mar 2025 | Identifies length and difficulty bias in GRPO; proposes fixes |

### PRM and Search Methods

| Paper | Date | Key Contribution |
|---|---|---|
| [ThinkPRM](https://arxiv.org/abs/2504.16828) | Apr 2025 | PRM that "thinks" via CoT verification |
| [Boosting PRM with MCTS](https://aclanthology.org/2025.findings-acl.388.pdf) | ACL 2025 | Direct MCTS + PRM integration |
| [Reward-Guided Tree Search](https://arxiv.org/abs/2411.11694) | Nov 2024 | Reward-guided search for training data generation |
| [AlphaZero-like LLM Decoding](https://papers.nips.cc/paper_files/paper/2024/file/00d80722b756de0166523a87805dd00f-Paper-Conference.pdf) | NeurIPS 2024 | AlphaZero-style search for LLM reasoning |
| [AgentPRM](https://arxiv.org/abs/2502.10325) | Feb 2025 | PRM for LLM agents |

### Self-Play and Self-Improvement

| Paper | Date | Key Contribution |
|---|---|---|
| [SPIN](https://dl.acm.org/doi/10.5555/3692070.3692326) | ICLR 2024 | Self-play fine-tuning weak-to-strong |
| [SeRL](https://neurips.cc/virtual/2025/poster/117365) | NeurIPS 2025 | Self-play RL for reasoning |
| [Self-Play Critic](https://neurips.cc/virtual/2025/poster/118706) | NeurIPS 2025 | Adversarial self-play for critic training |
| [AceSearcher](https://neurips.cc/virtual/2025/poster/116458) | NeurIPS 2025 | Cooperative self-play: reasoner + searcher |
| [SAGE](https://arxiv.org/abs/2603.15255) | 2025 | Multi-agent self-evolution |

### Search + RL Integration for Tool Use

| Paper | Date | Key Contribution |
|---|---|---|
| [R1-Searcher](https://arxiv.org/abs/2503.05592) | Mar 2025 | RL for learning to use external search |
| [ReSearch](https://arxiv.org/abs/2503.19470) | Mar 2025 | End-to-end search-reasoning integration via RL |
| [RASPberry](https://aclanthology.org/2025.findings-acl.587.pdf) | ACL 2025 | Retrieval-augmented MCTS self-play |
| [KBQA-o1](https://icml.cc/virtual/2025/poster/45313) | ICML 2025 | AlphaGo-style MCTS for knowledge base QA |

### Surveys and Analysis

| Paper | Date | Scope |
|---|---|---|
| [The State of RL for LLM Reasoning](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training) | Apr 2025 | Comprehensive analysis of RL methods for reasoning |
| [Survey: Large Reasoning Models with Self-Play DRL](https://dl.acm.org/doi/full/10.1145/3784013.3784042) | ACM 2025 | Self-play deep RL for reasoning survey |
| [Survey: Reinforced Reasoning](https://www.sciencedirect.com/science/article/pii/S2666389925002181) | 2025 | Survey of reinforced reasoning approaches |

### Additional Notable Papers (from Sebastian Raschka's analysis)

| Paper | Date | Key Insight |
|---|---|---|
| [Kimi k1.5](https://arxiv.org/abs/2501.12599) | Jan 2025 | Long context (128k) helps reasoning; long2short distillation |
| [Competitive Programming with o-models](https://arxiv.org/abs/2502.06807) | Feb 2025 | o3 learns its own test-time strategies via RL |
| [Logic-RL](https://arxiv.org/abs/2502.14768) | Feb 2025 | Logic puzzles train generalizable reasoning |
| [L1: Length Control](https://arxiv.org/abs/2503.04697) | Mar 2025 | LCPO for controlling reasoning length |
| [Concise Reasoning via RL](https://arxiv.org/abs/2504.05185) | Apr 2025 | PPO mathematically favors longer wrong answers |
| [Sober Look at Reasoning](https://arxiv.org/abs/2504.07086) | Apr 2025 | Many RL reasoning gains may be noise |
| [Rethinking Reflection in Pre-Training](https://arxiv.org/abs/2504.04022) | Apr 2025 | Self-correction emerges during pre-training |
| [Crossing the Reward Bridge](https://arxiv.org/abs/2503.23829) | Mar 2025 | Extends RLVR to medicine, chemistry, psychology |

---

## Resource Collections

- **[Awesome-LLM-Post-training](https://github.com/mbzuai-oryx/Awesome-LLM-Post-training)**: Comprehensive list of post-training methods for reasoning
- **[self-improvement-llm](https://github.com/Zesearch/self-improvement-llm)**: Technical reading list for LLM self-improvement methods
- **[learning-from-rewards-llm-papers](https://github.com/bobxwu/learning-from-rewards-llm-papers)**: Curated list of reward-based LLM training papers
- **[Sebastian Raschka's LLM Papers 2025 List](https://magazine.sebastianraschka.com/p/llm-research-papers-2025-list-one)**: Continuously updated list of key papers

---

*Report compiled from 40+ papers, surveys, and analyses spanning 2024-2026.*
