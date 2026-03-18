# Deep Dive: MDPs and LLM/Agents

**Date:** March 18, 2026

## Table of Contents
1. [Introduction](#1-introduction)
2. [MDP Fundamentals](#2-mdp-fundamentals)
3. [MDP vs Bandit Formulations for LLM Training](#3-mdp-vs-bandit-formulations-for-llm-training)
4. [POMDPs and LLM Agents](#4-pomdps-and-llm-agents)
5. [Key Frameworks and Papers](#5-key-frameworks-and-papers)
6. [Value-Based Methods for LLM Agents](#6-value-based-methods-for-llm-agents)
7. [Memory-Augmented MDPs](#7-memory-augmented-mdps)
8. [Hierarchical MDP Structures](#8-hierarchical-mdp-structures)
9. [Open Problems and Future Directions](#9-open-problems-and-future-directions)
10. [Key Papers Reference](#10-key-papers-reference)

---

## 1. Introduction

The intersection of Markov Decision Processes (MDPs) and Large Language Models (LLMs) represents one of the most active research areas in AI. This deep dive explores how classical decision-making frameworks from reinforcement learning are being adapted and extended for LLM agents, covering theoretical foundations, practical frameworks, and emerging research directions.

### Why MDPs Matter for LLM Agents

- **Sequential Decision Making**: LLM agents must make sequences of decisions (tool calls, reasoning steps, actions) rather than single-shot responses
- **Environmental Interaction**: Agents interact with external environments (APIs, databases, physical systems) where actions have consequences
- **Long-Horizon Planning**: Complex tasks require planning over multiple steps with delayed rewards
- **Theoretical Grounding**: MDPs provide a principled mathematical framework for analyzing agent behavior

---

## 2. MDP Fundamentals

### 2.1 Standard MDP Definition

A **Markov Decision Process** is defined as a tuple $(S, A, P, R, \gamma)$ where:

- **$S$**: State space (set of all possible states)
- **$A$**: Action space (set of all possible actions)
- **$P(s'|s,a)$**: Transition probability (probability of reaching state $s'$ from state $s$ after taking action $a$)
- **$R(s,a,s')$**: Reward function (immediate reward received)
- **$\gamma \in [0,1]$**: Discount factor (weight of future rewards)

**Key Property - Markov Assumption**: The next state depends only on the current state and action, not on the history of previous states:
$$P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)$$

### 2.2 MDP for LLM Agents

When adapting MDPs for LLM agents, the components map as follows:

| MDP Component | LLM Agent Interpretation |
|---------------|--------------------------|
| **State ($s_t$)** | Current context: conversation history, environment observations, task description, memory state |
| **Action ($a_t$)** | Token generation, tool call, API invocation, or complete response |
| **Transition ($P$)** | Environment dynamics: how the world changes after an action (deterministic for most digital environments) |
| **Reward ($R$)** | Task success signal, human feedback, process rewards, intermediate milestones |
| **Policy ($\pi$)** | The LLM itself - maps states to action distributions |

### 2.3 The Critical Distinction: State Representations

For LLM agents, the **state representation** is crucial:

```
State = {
    task_description,      # What needs to be done
    conversation_history,  # Past interactions
    environment_context,   # Current environment state
    memory_bank,          # Retrieved experiences
    internal_state        # Agent's "working memory"
}
```

The Markov property requires that this state representation be **sufficient** - it must contain all information needed to predict future outcomes without requiring additional history.

---

## 3. MDP vs Bandit Formulations for LLM Training

### 3.1 Two Dominant Formulations

There are **two primary RL formulations** used for LLM training:

#### Bandit Formulation (Contextual Bandit)
- **Single-step episodes**: Generate response → Receive reward → Episode ends
- **State**: The prompt/context (fixed throughout)
- **Action**: The entire generated response
- **Reward**: Outcome-based (e.g., human preference, task success)

```
Bandit Episode:
[Prompt] → [Full Response] → [Reward]
```

**Used by**: REINFORCE, RLOO, DPO (implicitly)

#### MDP Formulation (Token-Level)
- **Multi-step episodes**: Each token is a step
- **State**: Current prompt + previously generated tokens
- **Action**: Next token to generate
- **Reward**: Per-token or sparse (mostly zero until completion)

```
MDP Episode:
[Prompt] → [token_1] → [token_2] → ... → [token_n] → [Reward]
           ↑           ↑
         (r₁=0)     (r₂=0)
```

**Used by**: PPO with KL penalties, GRPO

### 3.2 When to Use Which Formulation?

| Criterion | Bandit Formulation | MDP Formulation |
|-----------|-------------------|-----------------|
| **Reward structure** | Outcome-only rewards | Dense/token-level rewards possible |
| **Credit assignment** | Hard (entire response) | Finer-grained (per-token) |
| **Training stability** | Generally more stable | Can be unstable |
| **Exploration** | Response-level diversity | Token-level diversity |
| **Use case** | RLHF, preference learning | Agents with tool use, multi-turn |

### 3.3 Key Insight from Practice

From the research community:

> "LLM training is on contextual bandits, not full MDPs" - Reddit r/reinforcementlearning discussion

The bandit formulation dominates because:
1. LLMs typically receive **outcome rewards** rather than step-by-step feedback
2. It's **simpler** and more stable for training
3. Credit assignment across tokens is fundamentally difficult

However, for **agentic tasks** (tool use, multi-turn interaction), the MDP formulation becomes essential.

---

## 4. POMDPs and LLM Agents

### 4.1 Why POMDPs?

**Partially Observable Markov Decision Processes** extend MDPs to handle incomplete state information - critical for real-world agents.

A **POMDP** is defined as $(S, A, P, R, \Omega, O, \gamma)$ where:
- Standard MDP components $(S, A, P, R, \gamma)$
- **$\Omega$**: Observation space
- **$O(o|s',a)$**: Observation probability (probability of observing $o$ after taking action $a$ and reaching state $s'$)

### 4.2 LLM Agents as POMDPs

**TensorZero's Key Insight**:
> "Think of LLM Applications as POMDPs — Not Agents"

LLM agents operate in partially observable environments:
- **Hidden environment state**: Database contents, API responses, user intent
- **Limited observations**: Only what tools/APIs return
- **Belief states**: Agents maintain implicit beliefs about the world

```
True World State (Hidden)
        ↓
    [Environment]
        ↓
Observations (Partial) → [LLM Agent] → Actions
        ↑                       ↓
        └── [Belief State] ←────┘
```

### 4.3 POMDP Research Highlights

#### InfoSeeker (arXiv 2510.01531)
- **Framework**: Information seeking for robust decision-making under partial observability
- **Key insight**: Integrates planning with information gathering
- **Approach**: LLM agents learn when to ask clarifying questions vs. act

#### LLM-Guided POMDP Model Estimation (arXiv 2505.02216)
- Uses LLMs to **construct POMDP models** from natural language descriptions
- Combines probabilistic programming with LLM guidance
- Applications: Robotics, mobile-base navigation

#### Augmenting LLMs with Causal Models for POMDP Planning (PMC 2025)
- Integrates human mental causal models with LLMs
- Focus on object assembly and troubleshooting domains
- Causal reasoning improves performance under partial observability

---

## 5. Key Frameworks and Papers

### 5.1 Agent-R1 (arXiv 2511.14460) - *Most Comprehensive MDP Extension*

**Agent-R1** is the most systematic framework for extending MDPs to LLM agents.

#### Core Contribution: Extended MDP Framework

The framework extends the standard MDP tuple to capture multi-turn agent interactions:

```
Extended MDP = (S, A, P, R, γ, H, T, M)

Where:
- S: State space (includes environment state + conversation history)
- A: Action space (tool calls + text generation)
- P: Transition probability
- R: Reward function (process + outcome rewards)
- γ: Discount factor
- H: Horizon (maximum turns)
- T: Tool set (available tools/APIs)
- M: Memory mechanism
```

#### Key Design Principles

1. **Multi-turn Tool Calling**: End-to-end RL on complete interaction trajectories
2. **Multi-tool Coordination**: Agents learn to coordinate multiple tools
3. **Process Rewards**: Rewards for each tool call based on effectiveness
4. **Outcome Balance**: Process rewards normalized with outcome rewards

#### Results
- Significantly outperforms RAG baselines on multi-hop QA
- Demonstrates that extended MDP framework enables better tool-use policies
- Open-source framework for community research

### 5.2 A-LAMP (arXiv 2512.11270) - *Automated MDP Modeling*

**A-LAMP**: Agentic LLM-Based Framework for Automated MDP Modeling

#### Problem Addressed
Applying RL to real-world tasks requires:
1. Converting informal descriptions to formal MDPs
2. Implementing executable environments
3. Training policy agents

#### Solution
Uses LLM agents to **automate MDP construction**:
- Natural language → Formal MDP specification
- Automatic code generation for environments
- Policy training pipeline

### 5.3 STRIDE (Cowles Foundation 2024)

**STRIDE**: Tool-Assisted LLM Agent Framework for Strategic Decision Making

- Integrates MDP solving with LLM reasoning
- Uses pretrained LLMs (GPT-4, Claude) as reasoning modules
- Designed for strategic planning in complex environments

### 5.4 Training Task Reasoning LLM Agents (arXiv 2509.20616)

Formulates multi-turn task planning as **two interconnected MDPs**:
1. **Multi-turn MDP**: Complete long-horizon task planning
2. **Single-turn MDP**: Based on expert trajectories for training

This dual-MDP approach enables efficient training while preserving multi-turn capabilities.

---

## 6. Value-Based Methods for LLM Agents

### 6.1 The Challenge of Value Estimation

Traditional RL uses value functions to guide decision-making. For LLM agents:
- **Q(s,a)**: Value of taking action $a$ in state $s$
- **V(s)**: Value of being in state $s$

Challenge: How to define states and actions in language space?

### 6.2 Step-Level Q-Value Models (arXiv 2409.09345)

**Paper**: "Enhancing Decision-Making for LLM Agents via Step-Level Q-Value Models"

#### Key Innovation
Trains a separate **Q-value model** to evaluate actions at each decision step:

```
At each step:
1. LLM generates candidate actions
2. Q-value model scores each action
3. Select action with highest Q-value
4. Execute and observe outcome
```

#### Training the Q-Value Model
- Collect trajectories with (state, action, reward) tuples
- Train Q-network using TD learning
- Use Q-values to guide action selection via **Direct Policy Optimization**

#### Results
- Significant improvement on multi-step tasks
- Better credit assignment across decision steps
- Works with both open-source and API-based LLMs

### 6.3 MLAQ: Model-based LLM Agent with Q-Learning (NeurIPS 2024)

Combines zero-shot LLM capabilities with optimal RL decision-making:

1. **Memory-based transitions**: Store and retrieve past experiences
2. **Q-learning on memories**: Derive optimal policies from stored transitions
3. **Zero-shot capability**: No training required on new tasks

### 6.4 Converting LLMs to Value-Based Planners (arXiv 2505.06987)

**Approach**: Define a "strategy-level MDP" and train a Q-network:
- Input: Textual state representation
- Output: Q-values over possible strategies
- Use Q-network as "plug-and-play strategic planner"

---

## 7. Memory-Augmented MDPs

### 7.1 Memento (arXiv 2508.16153) - *Memory-Based Learning Without Fine-Tuning*

**Memento** is a paradigm-shifting approach that enables LLM agents to learn **without updating model weights**.

#### Core Idea: Memory-Augmented MDP (M-MDP)

```
M-MDP = (S, A, P, R, γ, M)

Where M is the memory:
- Case Bank: Episodic memory of past trajectories
- Neural Case-Selection Policy: Learns which cases to retrieve
- Read/Write operations: Efficient memory access
```

#### How It Works

```
1. Agent faces new situation
2. Query Case Bank for similar past cases
3. Retrieve relevant trajectories
4. Use retrieved experience to guide decisions
5. Store new trajectory in Case Bank
6. Update case-selection policy (small network)
```

#### Key Advantages
- **No LLM fine-tuning**: Only trains small case-selection network
- **Continual learning**: Improves from experience
- **Low cost**: Fraction of the compute of full fine-tuning
- **Up to 9.6% accuracy improvement** on unseen tasks

### 7.2 Memento 2: Stateful Reflective Decision Process (arXiv 2512.22716)

Extends Memento with theoretical foundations:

#### Stateful Reflective Decision Process (SRDP)

```
SRDP extends MDP with:
- Reflective memory updates
- State-dependent retrieval
- Convergence guarantees
```

**Theoretical Contribution**: Proves that as memory grows and provides denser coverage of state space, the composite policy converges to optimal.

### 7.3 A-Mem: Agentic Memory (arXiv 2502.12110)

Framework for evolving memory in LLM agents:
- Memory evolution as new memories integrate
- Contextual representation updates
- Continuous refinement of memory network

---

## 8. Hierarchical MDP Structures

### 8.1 Why Hierarchical MDPs?

Long-horizon tasks require **decomposition**:
- High-level: Task planning, goal setting
- Mid-level: Subgoal generation
- Low-level: Action execution

### 8.2 Key Hierarchical Frameworks

#### EPO: Hierarchical LLM Agents (EMNLP 2024)
- **High-level LLM**: Subgoal prediction
- **Low-level LLM**: Action generation
- **Training**: Environment Preference Optimization

#### DEPART (OpenReview)
- **Planning Agent**: Task decomposition
- **Execution Agent**: Action execution
- **Visual Agent**: Environment understanding

#### HALO: Hierarchical Autonomous Logic-Oriented Orchestration
- **High-level**: Planning agent for task decomposition
- **Mid-level**: Role-design agents for subtask instantiation
- **Low-level**: Inference agents for subtask execution

### 8.3 Multi-Agent MDP Formulations

#### Dec-POMDP for Multi-Agent LLM Systems
For N agents, the decentralized POMDP (Dec-POMDP) formulation:
- Each agent has partial observations
- Joint action space: $A_1 \times A_2 \times ... \times A_N$
- Shared or individual rewards
- Coordination required for optimal behavior

#### LLM-based Multi-Agent RL Survey (arXiv 2405.11106)
Comprehensive review of multi-agent LLM systems:
- Communication protocols between agents
- Credit assignment in multi-agent settings
- Emergent coordination behaviors

---

## 9. Open Problems and Future Directions

### 9.1 Theoretical Challenges

| Challenge | Description | Current Status |
|-----------|-------------|----------------|
| **Credit Assignment** | How to attribute success/failure across long trajectories | Active research (process rewards, turn-level rewards) |
| **Exploration** | How should LLM agents explore efficiently | Thompson Sampling, PSRL approaches |
| **Non-stationarity** | Environments change as agents learn | Partially addressed in multi-agent settings |
| **Scalability** | MDP methods don't scale to language action spaces | Hierarchical approaches, abstractions |

### 9.2 Practical Challenges

1. **Reward Specification**: How to design rewards for complex tasks?
   - Solution directions: LLM-as-reward-model, process rewards, human feedback

2. **Sample Efficiency**: RL is notoriously sample-inefficient
   - Solution directions: Offline RL, demonstration learning, transfer learning

3. **Safety and Alignment**: Ensuring agents behave safely
   - Solution directions: Constrained MDPs, safe exploration, constitutional AI

4. **Evaluation**: How to benchmark MDP-based agents?
   - Emerging benchmarks: LLF-Bench, TravelPlanner, custom MDP environments

### 9.3 Emerging Research Directions

#### 9.3.1 Test-Time Compute as Meta-RL (CMU ML Blog)
Optimizing LLM test-time compute is fundamentally a **meta-RL problem**:
- Each test instance is a new MDP
- Agent must adapt using test-time computation
- Leverages pre-trained priors for efficient adaptation

#### 9.3.2 Turn-Level vs Trajectory-Level Rewards
Recent work (arXiv 2505.11821) investigates:
- **Trajectory-level rewards**: Single reward at end
- **Turn-level rewards**: Reward at each decision point
- Finding: Turn-level rewards improve multi-turn reasoning

#### 9.3.3 Process Reward Models for Agents (arXiv 2502.10325)
**AgentPRM**: Actor-critic framework for LLM agents:
- Monte Carlo rollouts for reward targets
- Lightweight critic network
- Continual improvement through interaction

### 9.4 Future Directions

1. **Unified MDP/Bandit Frameworks**: Approaches that smoothly interpolate between formulations based on task structure

2. **Neuro-symbolic MDPs**: Combining neural LLMs with symbolic planners for more interpretable, verifiable agent behavior

3. **Curriculum Learning for Agents**: Progressively complex MDPs to train generalist agents

4. **Inverse RL for Agent Alignment**: Learning reward functions from demonstrations of desired behavior

5. **Multi-modal MDPs**: Integrating vision, language, and action in unified decision-making frameworks

---

## 10. Key Papers Reference

### Foundational MDP/POMDP
| Paper | Venue | Key Contribution |
|-------|-------|------------------|
| Sutton & Barto RL Book | 2018 | Standard MDP formulation |
| Kaelbling et al. POMDP Survey | AIJ 1998 | POMDP foundations |
| Spaan POMDP Tutorial | 2012 | Practical POMDP methods |

### MDP Extensions for LLM Agents
| Paper | arXiv ID | Key Contribution |
|-------|----------|------------------|
| Agent-R1 | 2511.14460 | Extended MDP framework for agents |
| Memento | 2508.16153 | Memory-augmented MDP (M-MDP) |
| Memento 2 | 2512.22716 | SRDP with convergence guarantees |
| A-LAMP | 2512.11270 | Automated MDP modeling with LLMs |
| Task Reasoning LLM Agents | 2509.20616 | Dual MDP formulation |

### Value-Based Methods
| Paper | arXiv ID | Key Contribution |
|-------|----------|------------------|
| Step-Level Q-Value Models | 2409.09345 | Q-value guidance for LLM agents |
| MLAQ | NeurIPS 2024 | Q-learning with LLM memory |
| LLM as Value-Based Planner | 2505.06987 | Strategy-level MDP with Q-network |

### POMDP and LLM Agents
| Paper | arXiv ID | Key Contribution |
|-------|----------|------------------|
| InfoSeeker | 2510.01531 | Information seeking under partial observability |
| LLM-Guided POMDP Estimation | 2505.02216 | LLM constructs POMDP models |
| TensorZero Blog | 2024 | LLM applications as POMDPs |

### Surveys and Taxonomies
| Paper | Venue | Key Contribution |
|-------|-------|------------------|
| Evolving Landscape of LLM-RL | IJCAI 2025 | Comprehensive taxonomy: Agent/Planner/Reward roles |
| LLM-based Multi-Agent RL | arXiv 2405.11106 | Multi-agent LLM systems survey |
| RL for LLM Fine-Tuning | ResearchGate 2025 | Systematic literature review |

### Hierarchical and Multi-Agent
| Paper | arXiv ID | Key Contribution |
|-------|----------|------------------|
| EPO | EMNLP 2024 | Hierarchical agents with EPO |
| HALO | arXiv 2505.13516 | Three-level hierarchical orchestration |
| HiPlan | arXiv | Hierarchical planning framework |

---

## Summary: Key Takeaways

1. **MDP vs Bandit**: Bandit formulation dominates LLM fine-tuning (simpler, more stable), but MDP formulation is essential for agentic tasks with multi-turn interaction

2. **POMDP is the Right Model**: Real-world LLM agents operate with partial observability; POMDP formulations capture this reality better than full MDPs

3. **Memory Augmentation**: M-MDP and similar approaches enable continual learning without expensive LLM fine-tuning

4. **Value-Based Methods**: Q-value models can guide LLM agent decisions, improving credit assignment in multi-step tasks

5. **Hierarchical Decomposition**: Complex tasks require hierarchical MDP structures with specialized agents at each level

6. **Process Rewards Matter**: Turn-level and step-level rewards improve learning compared to trajectory-level-only rewards

7. **Theoretical Gaps Remain**: Convergence guarantees, sample complexity, and optimal exploration for LLM-based MDPs are active research areas

---

*Last updated: March 18, 2026*
