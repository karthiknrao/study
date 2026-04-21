# Deep Dive: Markov Decision Processes (MDPs) and Large Language Models (LLMs)

## Executive Summary

This document provides a comprehensive exploration of how MDPs and LLMs intersect across six key dimensions: (1) LLM agents *as* MDPs/POMDPs, (2) MDP-based training of LLMs (RLHF, GRPO, Bellman methods), (3) LLMs as MDP solvers/planners, (4) LLM world models and transition dynamics, (5) hierarchical MDP decomposition for chain-of-thought reasoning, and (6) value-guided decoding as optimal MDP policy selection.

---

## Table of Contents

1. [Foundations: MDPs Refresher](#1-foundations-mdps-refresher)
2. [LLM Agents AS MDPs: The POMDP Framing](#2-llm-agents-as-mdps-the-pomdp-framing)
3. [MDP-Based Training of LLMs](#3-mdp-based-training-of-llms)
4. [LLMs as MDP Solvers and Planners](#4-llms-as-mdp-solvers-and-planners)
5. [LLM World Models and Transition Dynamics](#5-llm-world-models-and-transition-dynamics)
6. [Hierarchical MDPs and Chain-of-Thought](#6-hierarchical-mdps-and-chain-of-thought)
7. [Value-Guided Decoding](#7-value-guided-decoding)
8. [Key Research Papers (2024-2026)](#8-key-research-papers-2024-2026)
9. [Open Problems and Future Directions](#9-open-problems-and-future-directions)
10. [References](#10-references)

---

## 1. Foundations: MDPs Refresher

### 1.1 Formal Definition

A **Markov Decision Process** is defined as a tuple **M = ⟨S, A, T, R, γ⟩**:

| Component | Symbol | Description |
|-----------|--------|-------------|
| State Space | S | Set of all possible environment configurations |
| Action Space | A | Set of all actions available to the agent |
| Transition Function | T(s,a,s') | P(s' \| s, a) — probability of reaching s' from s after action a |
| Reward Function | R(s,a) | Immediate scalar feedback for taking action a in state s |
| Discount Factor | γ ∈ [0,1] | Weight on future vs. immediate rewards |

### 1.2 The Bellman Equations

The **Bellman Optimality Equation** is the foundational recursive relationship:

```
V*(s) = max_a [ R(s,a) + γ · Σ_{s'} T(s'|s,a) · V*(s') ]

Q*(s,a) = R(s,a) + γ · Σ_{s'} T(s'|s,a) · max_{a'} Q*(s',a')
```

Where:
- **V\*(s)** = optimal value of being in state s
- **Q\*(s,a)** = optimal value of taking action a in state s
- The optimal policy: **π\*(s) = argmax_a Q\*(s,a)**

### 1.3 POMDP Extension

When the agent cannot fully observe the state:

**POMDP = ⟨S, A, T, R, Ω, O, γ⟩**

Additional components:
- **Ω**: Observation space (what the agent perceives)
- **O(s',a,o)**: P(o | s', a) — probability of observing o after action a leads to s'
- **Belief state b(s)**: Distribution over possible true states

Belief update (Bayesian filtering):
```
b'(s') = η · O(o | s', a) · Σ_s T(s' | s, a) · b(s)
```

### 1.4 Solution Methods at a Glance

| Method | Type | Complexity | Use Case |
|--------|------|------------|----------|
| Value Iteration | Dynamic Programming | O(\|S\|²\|A\|) per iteration | Small, known MDPs |
| Policy Iteration | Dynamic Programming | O(\|S\|³ + \|S\|²\|A\|) | Small, known MDPs |
| Q-Learning | Model-free RL | O(\|S\|·\|A\|) per step | Unknown dynamics |
| PPO/GRPO | Policy Gradient | O(batch size) per step | LLM fine-tuning |
| MCTS | Tree Search | O(b^d) | Large state spaces |
| POMCP | Monte Carlo POMDP | Sample-based | Partial observability |

---

## 2. LLM Agents AS MDPs: The POMDP Framing

### 2.1 The Core Insight

> "For the sake of LLM optimization, LLM applications are better framed as POMDPs — **not as agents**."
> — TensorZero (Mehta & Bianconi, 2025)

This is one of the most influential recent framings. The key insight: **the LLM itself is the policy**, not the agent. The "agent" is the entire application (code + LLM + environment).

### 2.2 The POMDP Mapping for LLM Applications

```
┌──────────────────────────────────────────────────────────────────────┐
│          LLM APPLICATION AS POMDP (TensorZero Framing)               │
├──────────────────┬───────────────────────────────────────────────────┤
│ POMDP Component  │ LLM Application Equivalent                       │
├──────────────────┼───────────────────────────────────────────────────┤
│ State S          │ User state, DB contents, external world state     │
│                  │ (never fully observed)                            │
│ Action A         │ LLM function outputs Y (text, tool calls, etc.)   │
│ Transition T     │ All application code that is NOT the LLM          │
│ Reward R         │ Feedback signal (CSAT, resolution, edits, prefs)  │
│ Observation Ω    │ Choice of LLM function f_i + inputs X_i          │
│ Observation fn O │ How the app state determines what to ask the LLM  │
│ Policy π         │ Implementation of LLM functions (prompt+model)   │
│ Horizon H        │ Max LLM calls per session/episode                │
└──────────────────┴───────────────────────────────────────────────────┘
```

### 2.3 Why This Matters Practically

**1. The Interface Problem**: The boundary between app and LLM should be **variables** (X → Y), not prompts/generations (P → G). This enables:
- Model-agnostic optimization
- Clean observability
- Proper offline policy evaluation

**2. Trajectory Storage**: You need (observation, action, reward) trajectories — not just chat logs. This means storing X and Y, not prompts and completions.

**3. Policy Improvement Loop**:
```
Deploy policy → Collect trajectories → Generate new policy variants
     ↑                                          │
     └──────────── Offline evaluation ──────────┘
```

### 2.4 Agent Memory as POMDP Belief State

From recent work on autonomous agent memory (2026):

```
Agent's Internal State ≈ POMDP Belief State

┌─────────────────────────────────────────────────┐
│  Agent Memory Types → Belief State Components    │
├─────────────────┬───────────────────────────────┤
│ Working Memory  │ Current task context           │
│ Episodic Memory │ Past (obs, action, reward)     │
│ Semantic Memory │ Learned facts about the world  │
│ Procedural      │ Learned skills/policies        │
└─────────────────┴───────────────────────────────┘
```

The memory update is itself a belief update — the agent refines its distribution over possible world states as it gathers observations.

---

## 3. MDP-Based Training of LLMs

### 3.1 The LLM-as-Policy Paradigm

LLM fine-tuning with RL is fundamentally an MDP:

```
┌──────────────────────────────────────────────────────────────┐
│              LLM FINE-TUNING AS MDP                           │
│                                                              │
│  State s_t    = Conversation context (tokens w_1,...,w_t)    │
│  Action a_t   = Next token w_{t+1} from vocabulary V         │
│  Transition   = Deterministic append: s_{t+1} = s_t ∘ a_t    │
│  Reward r     = From reward model (RLHF) or rules (GRPO)     │
│  Policy π_θ   = LLM parameterized by θ                       │
│  γ            = Typically 1.0 (undiscounted) or close to 1    │
│                                                              │
│  Objective: max_θ E_{τ~π_θ}[Σ_t R(s_t, a_t)]                │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 RLHF: Reward Models as MDP Reward Functions

**Pipeline**:
```
Human preferences → Reward model R_φ(s,a) → PPO/RL optimizes π_θ against R_φ
```

The reward model R_φ is trained on human preference data:
```
L_RM = -E[log σ(R_φ(x, y_w) - R_φ(x, y_l))]
```
where y_w ≻ y_l means y_w is preferred over y_l.

**MDP Interpretation**:
- The reward model is the MDP reward function R(s,a)
- PPO updates the policy π_θ using policy gradient:
  ```
  ∇_θ J ≈ E[∇_θ log π_θ(a|s) · Â(s,a)]
  ```
  where Â(s,a) is the advantage function = Q(s,a) - V(s)

### 3.3 GRPO: Group Relative Policy Optimization

GRPO (used in DeepSeek-R1, etc.) simplifies PPO by eliminating the value function:

```
GRPO Objective:
L_GRPO = E[Σ_i min(r_i(θ) · Â_i, clip(r_i(θ), 1-ε, 1+ε) · Â_i)]

Where:
  r_i(θ) = π_θ(y_i | x) / π_θ_old(y_i | x)
  Â_i = (r_i - mean(r_group)) / std(r_group)  [normalized group rewards]
```

**Key difference from PPO**: No separate critic network. The advantage is computed relative to the group of sampled completions. This is a **bandit-style** MDP (single-step) rather than a full sequential MDP.

### 3.4 DPO: Direct Preference Optimization

DPO eliminates the reward model entirely by deriving a closed-form optimal policy:

```
π*(y|x) ∝ π_ref(y|x) · exp(R(x,y)/β)

DPO Loss:
L_DPO = -E[log σ(β · log(π_θ(y_w|x)/π_ref(y_w|x)) - β · log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

This is equivalent to implicit RL on the MDP where the reward function is implicit in the preference data.

### 3.5 ShiQ: Bringing Bellman Back to LLMs (NeurIPS 2025)

One of the most exciting recent developments. ShiQ adapts **Q-learning** to LLMs:

**The Problem**: Naively applying Q-learning to LLM logits doesn't work because:
- Logits are not calibrated Q-values
- The token-level MDP has a massive action space (vocabulary size ~100K)
- Standard Bellman updates diverge on softmax outputs

**ShiQ's Solution**: Derive theoretically grounded loss functions from Bellman equations:

```
Standard Bellman:
Q(s,a) = R(s,a) + γ · max_{a'} Q(s',a')

ShiQ's Shifted-Q Update:
L_ShiQ = E[(Q_θ(s,a) - (r + γ · V_target(s')) - shift)²]

Where shift accounts for the logit-to-Q-value mapping peculiar to LLMs.
```

**Results**: Works in both single-turn and multi-turn settings, supports off-policy learning (sample efficient), validated on UltraFeedback and BFCL-V3.

### 3.6 SPPD: Process Preference with Bellman Optimality (EMNLP 2025)

SPPD formulates reasoning as a **process-based MDP**:

```
State s_t  = Partial reasoning chain (tokens 1..t)
Action a_t = Next reasoning step
Reward r_t = Process reward (step correctness)

Bellman optimality for dynamic value margin:
V*(s_t) = max_{a_t} [r(s_t, a_t) + γ · V*(s_{t+1})]
```

This is a direct application of the Bellman optimality equation to the chain-of-thought process, using process reward models (PRMs) as intermediate rewards.

---

## 4. LLMs as MDP Solvers and Planners

### 4.0 The Definitive Survey

**"The Landscape of Agentic RL for LLMs"** (Zhang et al., 2025, 500+ works) formalizes the critical distinction:

```
LLM-RL (single-step):    One prompt → one response → one reward (degenerate MDP, H=1)
Agentic RL (sequential):  Multi-turn POMDP with tool use, memory, planning

Agentic RL optimizes: planning, tool use, memory, reasoning, self-improvement, perception
```

### 4.1 LLM-Generated Policies

LLMs can approximate MDP solutions without explicit dynamic programming:

```
┌────────────────────────────────────────────────────────────┐
│           LLM AS APPROXIMATE MDP SOLVER                     │
│                                                            │
│  Input: MDP description in natural language                 │
│         (states, actions, transitions, rewards)             │
│                                                            │
│  LLM generates:                                            │
│  • Policy: π(s) → a for each state                         │
│  • Value estimates: Approximate V(s) or Q(s,a)             │
│  • Action sequences: Near-optimal trajectories              │
│  • Heuristics: For guiding classical solvers                │
└────────────────────────────────────────────────────────────┘
```

### 4.2 A-LAMP: Automated MDP Modeling (Dec 2025)

Uses multiple specialized LLM agents to automatically translate free-form natural language task descriptions into formal MDP tuples (S, A, T, R) and train RL policies:

```
A-LAMP Pipeline:
1. Modeling Agent: NL description → formal MDP components
2. Coding Agent:  Formal MDP → executable RL environment code
3. Training Agent: Environment code → trained policy

Each stage is independently verifiable — catch errors early
```

### 4.3 LLM-Guided POMDP Model Estimation (MIT CSAIL, 2025)

**POMDP Coder**: Uses LLMs to guide construction of interpretable probabilistic programs for POMDP models.

```
Workflow:
1. LLM generates candidate POMDP model components (T, R, O)
2. Probabilistic program induction refines these components
3. Result: interpretable, sample-efficient POMDP model

Key Result: LLM-guided POMDP construction outperforms:
  • Tabular POMDP learning
  • Behavior cloning
  • Direct LLM planning (without structured model)
```

**Why this matters**: The LLM provides commonsense priors about transition dynamics T and observation models O, dramatically reducing the data needed to learn the POMDP.

### 4.4 Tree Search + LLM MDP Solvers

Multiple breakthrough papers combine tree search with LLMs over the reasoning MDP:

```
┌──────────────────────────────────────────────────────────────────┐
│           TREE SEARCH METHODS FOR LLM REASONING MDP              │
├──────────────────┬───────────────────────────────────────────────┤
│ Method           │ Key Idea                                     │
├──────────────────┼───────────────────────────────────────────────┤
│ Q* (2024)        │ A* search with learned Q-value heuristic      │
│                  │ for multi-step reasoning                      │
│ TS-LLM (NeurIPS  │ AlphaZero-style MCTS with learned value fn   │
│ '23/ICML '24)    │ and LLM action priors                         │
│ ReST-MCTS*       │ MCTS over reasoning steps with process        │
│ (NeurIPS '24)    │ rewards as value estimates; auto-labeling     │
│ LATS (ICML '24)  │ MCTS unifying reasoning, acting, planning.   │
│                  │ LLM = policy + value function                 │
│ PGTS (ICML '25)  │ Learned policy guides tree search paths       │
│ Scaling Test-Time│ Test-time compute as MDP: decide when to      │
│ Compute (ICLR'25)│ search more vs. output (ICLR Oral)            │
│ Gumbel SH (2026) │ Shows AlphaZero-style methods plateau;        │
│                  │ Gumbel-based search scales better              │
└──────────────────┴───────────────────────────────────────────────┘
```

### 4.5 RL of Thoughts (RLoT) — Tsinghua, 2025

Models reasoning as an MDP with 5 logic-block actions:

```
Actions: {Deduce, Abduce, Induce, Analogize, Verify}

A <3K parameter RL navigator dynamically selects logic blocks.
Result: Sub-10B LLMs become comparable to 100B models on reasoning tasks.
```

### 4.6 Formal Verification of LLM MDP Policies

**"Verifying Memoryless Sequential Decision-Making of LLMs"** (2025):
Given an MDP and LLM policy, incrementally constructs the reachable MDP portion for formal verification against PCTL safety requirements. This is practical because it only explores reachable states rather than the full state space.

### 4.7 LLM + Classical Planners

The most effective approaches combine LLM commonsense with classical optimality guarantees:

```
┌───────────────┐    ┌──────────────┐    ┌───────────────┐
│ LLM provides  │───▶│ Classical    │───▶│ Optimal or    │
│ • Heuristics  │    │ MDP/POMDP    │    │ near-optimal  │
│ • Abstraction │    │ Solver       │    │ policy        │
│ • World know. │    │ (VI, MCTS,   │    │               │
│ • Reward spec │    │  POMCP, etc) │    │               │
└───────────────┘    └──────────────┘    └───────────────┘
```

Examples:
- **Code as Policies**: LLM generates code that defines robot policies
- **SayCan**: LLM scores actions, low-level controllers execute
- **LLM-Planner**: LLM generates high-level plans, classical planners refine

### 4.4 Dec-POMDP for Multi-LLM Collaboration (AAAI 2026)

Multiple LLM agents cooperating is formalized as a **Decentralized POMDP**:

```
Dec-POMDP = ⟨I, S, {A_i}, T, R, {Ω_i}, O, γ⟩

Where:
  I       = Set of LLM agents
  {A_i}   = Each agent's action space (independent)
  {Ω_i}   = Each agent's observation space (partial)
  Centralized training, decentralized execution (CTDE)
```

This frames multi-agent LLM cooperation as a well-studied RL problem, enabling principled solutions.

---

## 5. LLM World Models and Transition Dynamics

### 5.1 What is a World Model?

In RL, a **world model** is a learned approximation of the environment's transition dynamics:

```
World Model: Ŝ, Â → Ŝ'

Learns: T̂(s'|s,a) ≈ P(s'|s,a)
```

### 5.2 LLMs AS World Models

LLMs implicitly learn world dynamics from training data:

```
┌─────────────────────────────────────────────────────────────┐
│           LLM AS WORLD MODEL                                 │
│                                                             │
│  "If I push the glass off the table, it will..."            │
│                    ↓                                        │
│  LLM predicts: "...fall and break"                          │
│                    ↓                                        │
│  This is essentially T̂(s'|s,a): predicting next state       │
│  given current state and action                             │
│                                                             │
│  State s    = "glass on table"                              │
│  Action a   = "push glass"                                  │
│  Next s'    = "glass broken on floor"                       │
│  LLM computes: P(s' | s, a) via next-token prediction      │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 RWML: Reinforcement World Model Learning (2026)

**Paper**: "Reinforcement World Model Learning for LLM-based Agents" (Yu et al., Feb 2026)

Key innovation: Train LLMs to predict **action-conditioned state transitions** using self-supervised RL:

```
RWML Training:
1. Agent takes action a in state s, observes real next state s'_real
2. LLM predicts next state: s'_pred = LLM(s, a)
3. Reward = similarity(s'_pred, s'_real) in embedding space ("sim-to-real gap")
4. Update LLM to minimize sim-to-real gap

Key insight: Use embedding-space similarity, NOT token-level prediction
- Token-level: "The cup is to the left" ≠ "The cup sits leftward" (different tokens, same meaning)
- Embedding-level: These map to similar representations → correct reward signal
```

**Results**: +6.9 points on ALFWorld, +5.7 points on τ²-Bench over direct task-success RL. Matches expert-data training without requiring any expert data.

### 5.4 RLVR-World: Training World Models with RL (NeurIPS 2025)

Treats world modeling as autoregressive prediction of tokenized state sequences, trained with RL:

```
Standard: Next-token prediction (cross-entropy loss)
RLVR-World: World model prediction (RL loss against environment metrics)

Evaluates on both language-based and video-based world models
across domains: text games, web navigation, embodied tasks
```

### 5.5 WALL-E: World Alignment for LLM Agents (NeurIPS 2025)

Bridges the gap between LLM prior knowledge and actual environment dynamics:

```
Problem: LLM's internal T̂(s'|s,a) ≠ true T(s'|s,a)
         (LLM thinks pushing a button always works; real button is broken)

Solution: Neurosymbolic learning to align LLM world model
          with actual environment dynamics
```

### 5.6 Affordances Enable Partial World Modeling (2026)

Uses LLMs as transition models but acknowledges their limitations:

```
Strategy:
- Use LLM for T̂(s'|s,a) when confident (commonsense domains)
- Fall back to learned dynamics model for uncertain transitions
- Explore T̂(s) as function of state alone (cheaper than T̂(s,a))
```

---

## 6. Hierarchical MDPs and Chain-of-Thought

### 6.1 The Options Framework

**Options** extend the MDP action space to include temporally extended actions:

```
Option o = ⟨I_o, π_o, β_o⟩
  I_o ⊆ S      : Initiation set (where can this option start?)
  π_o : S → A   : Internal policy (what does this option do?)
  β_o : S → [0,1]: Termination condition (when does it stop?)

Semi-MDP: M = ⟨S, O, T̃, R̃, γ̃⟩
  O     = Set of options (replaces primitive actions)
  T̃     = Option-level transition function
  R̃     = Option-level cumulative reward
  γ̃     = Adjusted discount factor
```

### 6.2 Chain-of-Thought as Hierarchical MDP

CoT reasoning can be formalized as a two-level MDP:

```
┌──────────────────────────────────────────────────────────────┐
│         CHAIN-OF-THOUGHT AS HIERARCHICAL MDP                  │
│                                                              │
│  HIGH-LEVEL MDP (Planning):                                  │
│  State:  Task description + partial solution outline          │
│  Action: Choose next sub-problem / reasoning step             │
│  Policy: LLM with "think step by step" prompting             │
│                                                              │
│  LOW-LEVEL MDP (Execution):                                  │
│  State:  Current reasoning step context                       │
│  Action: Generate next token                                  │
│  Policy: LLM token-level generation                           │
│                                                              │
│  High-level chooses "options" (reasoning steps)               │
│  Low-level executes each option (generates the step text)     │
│                                                              │
│  Reward: Outcome reward (correct answer)                      │
│          + Process reward (correct intermediate steps)         │
└──────────────────────────────────────────────────────────────┘
```

### 6.3 LLM-Augmented Hierarchical RL (Nature Scientific Reports, 2025)

Combines LLMs with hierarchical RL for robotic manipulation:

```
Architecture:
  High-level: LLM planner decomposes task into subgoals
  Low-level:  RL policies execute each subgoal

Benefits:
  • LLM provides commonsense task decomposition (no training needed)
  • RL policies handle precise motor control
  • Dramatically reduces RL training time (fewer sub-problems)
```

### 6.4 Option Discovery Using LLMs (2025)

LLMs discover meaningful options/skills automatically:

```
LLM-Guided Semantic Hierarchical Abstraction:
1. LLM identifies semantically meaningful sub-tasks from language descriptions
2. These become options in the Semi-MDP
3. RL learns low-level policies for each option
4. High-level planner selects among options using LLM commonsense
```

### 6.5 AgentOwl: Joint Learning of Options and World Models (Cornell, 2026)

Learns hierarchical neural options and abstract world models jointly:

```
Joint optimization:
  {o_i} = set of options
  T̂_abstract = abstract transition model

  Simultaneously learn:
  • Which options to have (option discovery)
  • What each option does (option policy)
  • How options chain together (abstract world model)
```

---

## 7. Value-Guided Decoding

### 7.1 Token Generation as Sequential Decision Making

Each token generation step in an LLM is an action in an MDP:

```
┌──────────────────────────────────────────────────────────────┐
│          TOKEN-LEVEL MDP FOR DECODING                         │
│                                                              │
│  State s_t  = Context [x, y_1, ..., y_t] (all tokens so far) │
│  Action a_t = Choose token y_{t+1} from vocabulary V         │
│  Transition = Deterministic append: s_{t+1} = s_t ∘ a_t      │
│  Reward r_t = 0 for t < T, R(s_T) at end (outcome reward)    │
│  Policy π_θ = LLM's token distribution P(y_{t+1} | s_t)      │
│                                                              │
│  Goal: Find π* that maximizes E[R(s_T)]                      │
│       = Best possible token sequence for the given task       │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 Process Reward Models (PRMs) as Value Functions

**PRMs** estimate V(s_t) — the value of a partial solution:

```
Outcome Reward Model (ORM): R(final_answer) — only at the end
Process Reward Model (PRM): V(s_t) at each step — intermediate values

PRM as Bellman Value Function:
  V(s_t) = E[R(s_T) | s_t]
         = Σ_{s_T} P(s_T | s_t) · R(s_T)

This is exactly the definition of a value function in an MDP!
```

**PRM with Q-value Rankings** (OpenReview, 2025):
Reframes PRM as a Q-value ranking problem:
```
Q(s_t, a_t) = expected future reward if we take step a_t from state s_t
```
This enables step-level credit assignment using Bellman-style reasoning.

### 7.3 Value-Guided Decoding Strategies

```
┌──────────────────────────────────────────────────────────────┐
│        DECODING STRATEGIES ON THE VALUE FUNCTION SPECTRUM     │
│                                                              │
│  Greedy/Top-k      │ max P(y|x) — no value guidance          │
│  ──────────────────┼────────────────────────────────────────  │
│  Best-of-N          │ Sample N, pick max V(s_T)               │
│  ──────────────────┼────────────────────────────────────────  │
│  Beam Search + PRM  │ Beam search with V(s_t) re-ranking      │
│  ──────────────────┼────────────────────────────────────────  │
│  MCTS + Value       │ Tree search guided by V(s_t)            │
│  ──────────────────┼────────────────────────────────────────  │
│  Reward-Guided      │ Speculative decoding with PRM reward    │
│  Speculative       │ for candidate verification               │
│  ──────────────────┼────────────────────────────────────────  │
│  Optimal π*        │ Exact Bellman-optimal token selection    │
│                     │ (intractable for real LLMs)              │
└─────────────────────┴──────────────────────────────────────────┘
```

### 7.4 MCTS for LLM Reasoning

Monte Carlo Tree Search applied to the token-level MDP:

```
MCTS + LLM:
1. Selection: UCB1 chooses which reasoning path to explore
   UCB(y_t) = Q(y_t) + c · √(log(N(parent)) / N(y_t))

2. Expansion: LLM generates possible next steps

3. Simulation: Roll out to completion, get reward R

4. Backpropagation: Update Q-values along the path
   Q(s,a) ← running average of rewards passing through (s,a)
```

### 7.5 Trajectory Bellman Residual Minimization (NeurIPS 2025)

A simple value-based method for LLM reasoning:

```
Instead of policy gradient (PPO/GRPO), directly minimize:
  L = E[(V_θ(s_t) - (r_t + γ · V_target(s_{t+1})))²]

Bellman residual = TD error
Minimizing it learns V*(s) — the optimal value function

Then use V* to guide decoding: prefer tokens that lead to high-value states
```

### 7.6 Q#: Provably Optimal Distributional RL for LLM Post-Training (NeurIPS 2025)

Goes beyond expected value to learn the full **return distribution**:

```
Standard Q-learning: learns E[return]
Distributional RL:   learns full distribution of returns

Q# adapts distributional RL to LLMs:
  - Models uncertainty in reasoning outcomes
  - More robust than point estimates
  - Provably optimal under certain conditions
```

---

## 8. Key Research Papers (2024-2026)

### 8.1 LLM Agents as MDPs/POMDPs

| Paper | Venue | Key Contribution |
|-------|-------|------------------|
| **Think of LLM Applications as POMDPs** | TensorZero Blog, 2025 | Formal mapping of LLM apps to POMDPs; practical engineering implications |
| **LLM-Guided POMDP Model Estimation** (POMDP Coder) | arXiv 2505.02216, 2025 | LLM guides probabilistic program induction for POMDP models |
| **Dec-POMDP for Multi-LLM Collaboration** | AAAI 2026 | Multi-agent LLM cooperation as Dec-POMDP with CTDE |
| **VAGEN: RL for VLM Agents** | NeurIPS 2025 | POMDP formulation for vision-language agents with world models |
| **Augmenting LLMs with Psychologically Grounded Memory** | Frontiers AI, 2026 | Memory as POMDP belief state for interactive agents |

### 8.2 MDP-Based LLM Training

| Paper | Venue | Key Contribution |
|-------|-------|------------------|
| **ShiQ: Bringing Back Bellman to LLMs** | NeurIPS 2025 | Q-learning adapted to LLMs via Bellman-derived loss functions |
| **SPPD: Process Preference with Bellman** | EMNLP Findings 2025 | Reasoning as process MDP with Bellman optimality |
| **Trajectory Bellman Residual Minimization** | NeurIPS 2025 | Value-based RL for LLM reasoning via TD error minimization |
| **Q#: Distributional RL for LLM Post-Training** | NeurIPS 2025 | Distributional Q-learning for LLM alignment |
| **DAPO: Multi-Step Reasoning** | NeurIPS 2025 | Step-level RL for multi-step reasoning improvement |
| **Offline RL for LLM Multi-Step Reasoning** | ACL Findings 2025 | Soft Bellman equation for offline policy + value optimization |
| **Self-Guided Process Reward Optimization** | arXiv 2507.01551, 2025 | Redefined step-wise MDP for process rewards |

### 8.3 LLM World Models

| Paper | Venue | Key Contribution |
|-------|-------|------------------|
| **RWML: Reinforcement World Model Learning** | arXiv 2602.05842, 2026 | Self-supervised world model learning with sim-to-real gap rewards |
| **RLVR-World** | NeurIPS 2025 | RL for training world models across domains |
| **WALL-E: World Alignment** | NeurIPS 2025 | Neurosymbolic alignment of LLM world models with environment |
| **Affordances Enable Partial World Modeling** | arXiv 2602.10390, 2026 | LLM as partial transition model with learned fallbacks |
| **Compositional World Modeling (PoPE)** | NeurIPS 2025 | Products of programmatic experts for compositional world models |

### 8.4 Hierarchical Planning and Options

| Paper | Venue | Key Contribution |
|-------|-------|------------------|
| **Option Discovery Using LLMs** | arXiv 2503.19007, 2025 | LLM-guided semantic hierarchical abstraction |
| **LLM-Augmented Hierarchical RL** | Nature Sci. Reports, 2025 | LLM planner + RL executors for robotic tasks |
| **AgentOwl: Joint Options + World Models** | Cornell, 2026 | Joint learning of hierarchical options and abstract world models |
| **Divide and Conquer: LLM as Decision-Making** | ICML 2025 | Grounding LLMs as efficient hierarchical decision-makers |

---

### 7.7 Thought MDPs (NeurIPS 2025)

A minimal extension of classical MDPs to include abstract "thought" states:

```
Thought MDP = ⟨S, S_thought, A, A_thought, T, R, γ⟩

S_thought : Internal reasoning states (not externally observable)
A_thought : Thinking actions (e.g., "analyze", "verify", "backtrack")

Key insight: The model can take "thinking actions" that change its
internal state without changing the external environment state.
This formalizes what chain-of-thought actually IS in MDP terms.
```

This connects to when model-free RL can be model-based — the thought process IS a form of internal planning/model-based reasoning within an otherwise model-free policy.

### 7.8 RL with Promising Tokens (2026)

Formulates LLM generation explicitly as an MDP and selects "promising tokens":

```
MDP Formulation:
  State s_t  = Generated tokens so far + prompt
  Action a_t = Next token from vocabulary V
  Reward     = Outcome reward at end of generation

"Promising token" = token whose selection keeps the trajectory
in high-value regions of the state space

Implementation:
  1. Use value function V(s) to estimate future return
  2. At each decoding step, prefer tokens that maximize V(s_{t+1})
  3. Balance exploitation (high V) with exploration (policy diversity)
```

### 7.9 The Q* Hypothesis

The rumored OpenAI Q* project likely combines:

```
Q* ≈ Process Reward Models + Tree Search + Q-Learning

Components:
  1. Process Reward Model → Q(s_t, a_t) for each reasoning step
  2. Tree Search (MCTS) → Explore reasoning paths
  3. Q-Learning → Learn optimal token/step selection policy

This is exactly the MDP framework:
  States    = Partial solutions
  Actions   = Next reasoning step/token
  Q-values  = Process rewards (PRM)
  Policy    = LLM + tree search guided by Q-values
```

### 7.10 RL in Name Only? Structural Critique of Current LLM-MDP Framings

An important critique (OpenReview, 2025):

```
The Problem: Current LLM post-training treats each generation as a
single-step bandit problem, NOT a true sequential MDP.

Why it matters:
- PPO/GRPO assign ONE reward per episode (the outcome)
- No temporal credit assignment within the sequence
- The "MDP" has H=1 horizon in practice
- True sequential RL would require step-level rewards (PRMs)

Implication: We're leaving performance on the table by not doing
proper sequential credit assignment with Bellman backups.
```

---

## 9. Open Problems and Future Directions

### 9.1 Bridging the Gap Between Theory and Practice

| Problem | Status | Outlook |
|---------|--------|---------|
| Exact Bellman optimality for LLMs | Intractable (vocab size ~100K) | Approximate methods (MCTS, beam search) improving rapidly |
| True POMDP solving for LLM agents | Too expensive for real-time | Hybrid: LLM priors + fast online planners |
| Sample efficiency of LLM RL | Still expensive (thousands of rollouts) | ShiQ/Q# style off-policy methods promising |
| Multi-step credit assignment | Hard with outcome rewards only | PRMs + Bellman decomposition address this |

### 9.2 Key Open Questions

1. **Can we build LLMs that natively reason in Bellman equations?** Current LLMs don't have built-in value functions. Can we architect LLMs with explicit V(s) computation?

2. **What is the right abstraction level for LLM-MDP integration?** Token-level MDPs are too fine-grained; step-level MDPs require process rewards; episode-level is too coarse.

3. **How do we handle non-stationarity?** LLM agents change the environment through their actions, and the environment may change independently. The MDP's T and R are non-stationary in practice.

4. **Multi-agent POMDPs at scale**: As LLM agents collaborate, we need scalable Dec-POMDP solutions. Current methods (CTDE) are promising but limited in number of agents.

5. **Formal verification of LLM policies**: Can we verify safety properties of LLM-derived policies? Rice's theorem says no for general TC agents, but constrained architectures (FSMs, bounded automata) may be verifiable.

### 9.3 Emerging Trends (2025-2026)

```
┌─────────────────────────────────────────────────────────────────┐
│                  CONVERGENCE MAP                                 │
│                                                                 │
│   Classical RL              LLM Research              Products   │
│   ──────────                ────────────              ────────   │
│                                                                 │
│   Bellman Eq.  ──────────▶ ShiQ, SPPD, TBRM  ────▶  Better     │
│                                                     alignment   │
│                                                                 │
│   World Models ──────────▶ RWML, RLVR-World   ────▶  Smarter    │
│                                                     agents      │
│                                                                 │
│   Options/SMDPs ─────────▶ CoT as hier. MDP  ────▶  Better      │
│                                                     reasoning   │
│                                                                 │
│   POMDPs ────────────────▶ LLM apps as       ────▶  Principled  │
│                            POMDPs                    engineering │
│                                                                 │
│   Value Functions ───────▶ PRMs, value-guided ────▶  Optimal     │
│                            decoding                   decoding   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. Unifying Synthesis: The Six-Layer MDP Stack

All the research converges on a single insight: **the Bellman equation connects every level of the LLM-MDP stack**.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    THE SIX-LAYER MDP STACK FOR LLMs                  │
│                                                                      │
│  Layer 6: WORLD MODEL       LLMs approximate T(s,a,s')              │
│           ↑                  (RWML, WorldCoder, RLVR-World)          │
│  Layer 5: PLANNING          MCTS, A*, tree search over reasoning MDP │
│           ↑                  (TS-LLM, ReST-MCTS*, LATS, Q*)         │
│  Layer 4: AGENT             Multi-turn POMDP with tools/memory       │
│           ↑                  (TensorZero, Dec-POMDP, VAGEN)          │
│  Layer 3: REASONING         Step-level MDP for chain-of-thought      │
│           ↑                  (SPPD, RLoT, Thought MDPs)             │
│  Layer 2: TRAINING          RLHF/GRPO/DPO as policy optimization    │
│           ↑                  (PPO, ShiQ, SPPD, SACPO)               │
│  Layer 1: DECODING          Token-level MDP: s_t, a_t=w_{t+1}       │
│                              (value-guided decoding, PRMs as V(s))   │
│                                                                      │
│  ═══════════════════════════════════════════════════════════════      │
│  THE BELLMAN EQUATION THREADS THROUGH ALL SIX LAYERS                 │
│  V*(s) = max_a [R(s,a) + γ · Σ_{s'} T(s'|s,a) · V*(s')]            │
│  ═══════════════════════════════════════════════════════════════      │
└──────────────────────────────────────────────────────────────────────┘
```

**Key takeaway**: Whether you're training an LLM with RLHF, guiding its decoding with process rewards, or building multi-turn agents that plan and use tools — you are always, implicitly or explicitly, solving a Bellman equation. The research of 2024-2026 is making this connection increasingly explicit and mathematically rigorous.

---

## 11. References

### Primary Papers

1. **ShiQ: Bringing back Bellman to LLMs** — Clavier et al., NeurIPS 2025. [arXiv:2505.11081](https://arxiv.org/abs/2505.11081)
2. **SPPD: Self-training with Process Preference Learning** — EMNLP Findings 2025. [PDF](https://aclanthology.org/2025.findings-emnlp.19.pdf)
3. **LLM-Guided Probabilistic Program Induction for POMDP Model Estimation** — Curtis & Tang, 2025. [arXiv:2505.02216](https://arxiv.org/abs/2505.02216)
4. **Reinforcement World Model Learning for LLM-based Agents** — Yu et al., 2026. [arXiv:2602.05842](https://arxiv.org/abs/2602.05842)
5. **Think of LLM Applications as POMDPs — Not Agents** — Mehta & Bianconi, TensorZero, 2025. [Blog](https://www.tensorzero.com/blog/think-of-llm-applications-as-pomdps-not-agents/)
6. **Trajectory Bellman Residual Minimization** — NeurIPS 2025. [NeurIPS Poster](https://neurips.cc/virtual/2025/poster/119725)
7. **Q#: Provably Optimal Distributional RL for LLM Post-Training** — NeurIPS 2025. [NeurIPS Poster](https://neurips.cc/virtual/2025/poster/118428)
8. **DAPO: Improving Multi-Step Reasoning** — NeurIPS 2025. [NeurIPS Poster](https://neurips.cc/virtual/2025/poster/119738)
9. **RLVR-World: Training World Models with RL** — NeurIPS 2025. [NeurIPS Poster](https://neurips.cc/virtual/2025/poster/116433)
10. **WALL-E: World Alignment by NeuroSymbolic Learning** — NeurIPS 2025. [NeurIPS Poster](https://neurips.cc/virtual/2025/poster/119191)
11. **VAGEN: Reinforcing World Model Reasoning** — NeurIPS 2025. [OpenReview](https://openreview.net/forum?id=xpjWEgf8zi)
12. **LLM Collaboration with Multi-Agent RL** — AAAI 2026. [arXiv:2508.04652](https://arxiv.org/html/2508.04652v7)
13. **Option Discovery Using LLM-guided Semantic Hierarchical Abstraction** — 2025. [arXiv:2503.19007](https://arxiv.org/html/2503.19007v1)
14. **LLM-Augmented Hierarchical RL** — Nature Scientific Reports, 2025. [Nature](https://www.nature.com/articles/s41598-025-20653-y)
15. **AgentOwl: Joint Learning of Hierarchical Neural Options** — Cornell, 2026. [PDF](https://www.cs.cornell.edu/~wp237/papers/piriyakulkij2026agentowl.pdf)
16. **Offline RL for LLM Multi-Step Reasoning** — ACL Findings 2025. [PDF](https://aclanthology.org/2025.findings-acl.464.pdf)
17. **Affordances Enable Partial World Modeling** — 2026. [arXiv:2602.10390](https://arxiv.org/pdf/2602.10390)
18. **Self-Guided Process Reward Optimization** — 2025. [arXiv:2507.01551](https://arxiv.org/html/2507.01551v2)
19. **Augmenting LLMs with Psychologically Grounded Memory** — Frontiers AI, 2026. [Frontiers](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1730614/full)
20. **Thought MDPs: When Can Model-Free RL Be Model-Based?** — NeurIPS 2025. [NeurIPS Poster](https://neurips.cc/virtual/2025/poster/116888)
21. **RL with Promising Tokens for LLMs** — arXiv 2602.03195, 2026. [PDF](https://arxiv.org/pdf/2602.03195)
22. **RL in Name Only? Analyzing Structural Assumptions** — OpenReview, 2025. [PDF](https://openreview.net/pdf?id=6hnjZMWICE)
23. **Process Reward Models for LLM Agents: Practical Framework** — arXiv 2502.10325, 2025. [HTML](https://arxiv.org/html/2502.10325v1)
24. **VinePPO: Refining Credit Assignment in RL Training of LLMs** — ICML 2025. [ICML Poster](https://icml.cc/virtual/2025/poster/45526)
25. **High-Level Planning Guidance RL for LLM Agents** — arXiv 2510.01833, 2025. [HTML](https://arxiv.org/html/2510.01833v1)
26. **A-LAMP: Agentic LLM Framework for Automated MDP Modeling** — arXiv 2512.11270, Dec 2025. [arXiv](https://arxiv.org/abs/2512.11270)
27. **The Landscape of Agentic RL for LLMs: A Survey** — Zhang et al., 2025. [arXiv:2509.02547](https://arxiv.org/abs/2509.02547)
28. **Q*: Improving Multi-Step Reasoning via Deliberative Planning** — 2024. [arXiv:2406.14283](https://arxiv.org/abs/2406.14283)
29. **TS-LLM: AlphaZero-Like Tree Search for LLM Decoding** — NeurIPS 2023/ICML 2024. [arXiv:2309.17179](https://arxiv.org/abs/2309.17179)
30. **ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search** — NeurIPS 2024. [arXiv:2406.03816](https://arxiv.org/abs/2406.03816)
31. **LATS: Language Agent Tree Search** — ICML 2024. [arXiv:2310.04406](https://arxiv.org/abs/2310.04406)
32. **Scaling LLM Test-Time Compute Optimally** — Snell et al., ICLR 2025 Oral. [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)
33. **RL of Thoughts (RLoT)** — Hao et al., Tsinghua, 2025. [arXiv:2505.14140](https://arxiv.org/abs/2505.14140)
34. **Verifying Memoryless Sequential Decision-Making of LLMs** — 2025. [arXiv:2510.06756](https://arxiv.org/abs/2510.06756)
35. **Revisiting Tree Search for LLMs: Gumbel and Sequential Halving** — 2026. [arXiv:2603.21162](https://arxiv.org/abs/2603.21162)
36. **Adaptive Decoding via Test-Time Policy Learning** — 2026. [arXiv:2603.18428](https://arxiv.org/abs/2603.18428)
37. **SACPO: Stepwise Alignment for Constrained LLM Policy Optimization** — NeurIPS 2024. [Project](https://akifumi-wachi-4.github.io/website/)
38. **WorldCoder: A Model-Based LLM Agent** — NeurIPS 2024. [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/820c61a0cd419163ccbd2c33b268816e-Paper-Conference.pdf)
39. **Zero-Shot Model-Based RL with LLMs** — ICLR 2025. [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/file/5fc1e662bd63c4a70b95088ba5d08cb8-Paper-Conference.pdf)
40. **What Are Step-Level Reward Models Rewarding?** — AAAI 2025. [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34663)

### Foundational Texts

- Sutton & Barto, **Reinforcement Learning: An Introduction** (2018) — Ch. 3-4 for MDPs and Bellman Equations
- Puterman, **Markov Decision Processes: Discrete Stochastic Dynamic Programming** (1994)
- Oliehoek & Amato, **A Concise Introduction to Decentralized POMDPs** (2016)
- Sutton, Precup & Singh, "Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in RL" (1999)

### Surveys and Books

- RLHF Book — Nathan Lambert. [rlhfbook.com](https://rlhfbook.com/)
- GRPO++: Tricks for Making RL Actually Work — Cameron Wolfe. [Substack](https://cameronrwolfe.substack.com/p/grpo-tricks)
- Cutting-Edge Advancements in RLHF (2023-2025). [Medium](https://medium.com/foundation-models-deep-dive/cutting-edge-advancements-in-rlhf-2023-2025-fe814c770e88)

---

*Document compiled April 2026. Complements the earlier MDP_Automata_Planning_DeepDive.md from March 2026.*
