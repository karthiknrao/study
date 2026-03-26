# Deep Dive: MDPs and Automata for Planning in LLMs and Agents

## Executive Summary

This document explores the intersection of **Markov Decision Processes (MDPs)**, **automata theory**, and **Large Language Models (LLMs)** for building intelligent planning agents. The fusion of these formal frameworks with LLMs enables more reliable, verifiable, and efficient AI systems.

---

## 1. Foundations

### 1.1 Markov Decision Processes (MDPs)

An MDP is a mathematical framework for sequential decision-making under uncertainty, formally defined as a tuple **M = ⟨S, A, T, R, γ⟩**:

| Component | Description |
|-----------|-------------|
| **S** | Finite set of states (environment configurations) |
| **A** | Finite set of actions available to the agent |
| **T** | Transition function: T(s, a, s') = P(s' \| s, a) |
| **R** | Reward function: R(s, a) → ℝ |
| **γ** | Discount factor (0 ≤ γ ≤ 1) |

**Key Properties:**
- **Markov Property**: Future states depend only on the current state, not the history
- **Goal**: Find optimal policy π* that maximizes expected cumulative reward
- **Solution Methods**: Value Iteration, Policy Iteration, Q-Learning

### 1.2 Partially Observable MDPs (POMDPs)

When the agent cannot directly observe the true state, we extend MDPs to POMDPs: **⟨S, A, T, R, Ω, O, γ⟩**

| Additional Components | Description |
|----------------------|-------------|
| **Ω** | Set of observations |
| **O** | Observation function: O(s', a, o) = P(o \| s', a) |
| **Belief State** | Distribution over possible states: b(s) |

**Critical for LLM Agents**: POMDPs model the inherent uncertainty in:
- Ambiguous human instructions
- Hidden or unknown object locations
- Open-vocabulary object types

### 1.3 Automata Theory Fundamentals

The **Chomsky Hierarchy** classifies computational models by their memory capacity:

| Level | Automaton | Memory | Language Class | Decidability |
|-------|-----------|--------|----------------|--------------|
| 1 | Finite Automaton (FA) | None (finite states) | Regular | Fully decidable |
| 2 | Pushdown Automaton (PDA) | Stack (LIFO) | Context-Free | Partially decidable |
| 3 | Linear Bounded Automaton (LBA) | Bounded tape | Context-Sensitive | Decidable |
| 4 | Turing Machine (TM) | Unbounded tape | Recursively Enumerable | Undecidable |

---

## 2. The Automata-Agent Framework

### 2.1 Formal Equivalence Between Agents and Automata

Recent research establishes a **direct correspondence** between agentic AI architectures and the Chomsky hierarchy based on memory architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT-TO-AUTOMATA MAPPING                    │
├─────────────────┬────────────────────┬─────────────────────────┤
│ Agent Type      │ Memory Model       │ Automaton Equivalent    │
├─────────────────┼────────────────────┼─────────────────────────┤
│ Regular Agent   │ Finite/No memory   │ Finite Automaton (FA)   │
│ Context-Free    │ Stack (LIFO)       │ Pushdown Automaton      │
│ Context-Sensitive│ Bounded tape      │ Linear Bounded Auto.    │
│ TC Agent        │ Unbounded R/W      │ Turing Machine          │
└─────────────────┴────────────────────┴─────────────────────────┘
```

### 2.2 Agent Class Definitions

#### Regular Agent (AR)
```
AR = (S, s₀, Σ, δ, F)
```
- **S**: Finite set of internal agent states
- **s₀**: Initial state
- **Σ**: Finite set of perceptions (input alphabet)
- **δ**: State transition function: δ: S × Σ → S
- **F**: Set of accepting states

**Examples**: Simple voice assistants, command-response systems, FSM-controlled robots

#### Context-Free Agent (ACF)
```
ACF = (S, s₀, Σ, Z, δ, F)
```
- **Z**: Finite stack alphabet (LIFO memory)
- **δ**: δ: S × Σε × Zε → P(S × Zε*)

**Examples**: Project management agents with hierarchical task decomposition, planning agents with nested subgoals

#### TC Agent (ATC)
```
ATC = (S, s₀, Σ, Γ, δ, F)
```
- **Γ**: Tape alphabet with unbounded R/W memory
- **δ**: δ: S × Γ → S × Γ × {L, R}

**Examples**: Research agents with web browsing, agents with arbitrary scratchpads, ReAct-style agents

### 2.3 The Right-Sizing Principle

**Key Engineering Insight**: Choose the minimal computational class sufficient for the task.

```
                    ┌──────────────────┐
                    │  Task Analysis   │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │Bounded   │  │Hierarchical│  │Unbounded │
        │States?   │  │Planning?   │  │Memory?   │
        └────┬─────┘  └─────┬──────┘  └────┬─────┘
             │              │              │
             ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │Regular   │  │Context-  │  │TC Agent  │
        │Agent (FA)│  │Free (PDA)│  │(TM)      │
        └──────────┘  └──────────┘  └──────────┘
```

**Benefits of Right-Sizing**:
1. **Cost & Latency Savings**: Constrained paths limit execution steps
2. **Regulatory Compliance**: Verifiable state transitions
3. **Improved Reliability**: Predictable behavior

---

## 3. POMDP Framework for LLM Agents

### 3.1 Tru-POMDP: A State-of-the-Art Framework

**Tru-POMDP** combines LLM reasoning with principled POMDP planning:

```
┌──────────────────────────────────────────────────────────────┐
│                     Tru-POMDP Architecture                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────┐  │
│  │ Task Input  │───▶│ Tree of          │───▶│ Hybrid     │  │
│  │ - Human     │    │ Hypotheses (TOH) │    │ Belief     │  │
│  │   instruct. │    │ - LLM generates  │    │ Update     │  │
│  │ - Scene     │    │   weighted       │    │ - Bayesian │  │
│  │   graph     │    │   particles      │    │   filtering│  │
│  └─────────────┘    └──────────────────┘    └─────┬──────┘  │
│                                                     │        │
│                                                     ▼        │
│                                          ┌────────────────┐  │
│                                          │ Online POMDP   │  │
│                                          │ Planning       │  │
│                                          │ - Belief tree  │  │
│                                          │ - Dynamic      │  │
│                                          │   branching    │  │
│                                          └────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Key Components

#### Tree of Hypotheses (TOH)
- Systematically queries LLM to construct particle beliefs
- Generates weighted hypotheses over:
  - Target objects
  - Target areas
  - Initial locations

#### Hybrid Belief Update
```
Belief Update Process:
1. Particle Prediction (from dynamics)
2. Particle Elimination (based on observations)
3. LLM Particle Augmentation (commonsense reasoning)
```

#### Open-Ended POMDP Model
- Handles boundlessly large planning spaces
- Open-vocabulary object types
- Ambiguous instructions

### 3.3 Mathematical Formulation

**POMDP for LLM Agents**:

```
POMDP = ⟨S, A, T, R, Ω, O, γ, B⟩

Where:
- S: World states (object positions, robot state)
- A: Actions (search, pick, place, navigate)
- T: Transition dynamics
- R: Reward function (task completion)
- Ω: Observations (visible objects, scene graph)
- O: Observation model
- B: Belief state = distribution over S
```

**Belief Update (Bayes Rule)**:
```
b'(s') = η · O(o | s', a) · Σ_s T(s' | s, a) · b(s)
```
Where η is a normalization constant.

---

## 4. Formal Methods + LLM Integration

### 4.1 The Fusion Roadmap

```
┌──────────────────────────────────────────────────────────────┐
│         LLM-FM BIDIRECTIONAL ENHANCEMENT                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   FMs for LLMs                    LLMs for FMs               │
│   ────────────                    ───────────                │
│   • Reliability                   • Autoformalization        │
│   • Certification                 • Theorem Proving          │
│   • Logical Reasoning             • Model Checking           │
│   • Behavior Analysis             • Specification Gen        │
│                                                              │
│                      ┌────────────┐                          │
│                      │ TRUSTWORTHY│                          │
│                      │ AI AGENT   │                          │
│                      └────────────┘                          │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 SMT Solvers for LLMs

**Integration Strategy**:
1. LLM generates formal constraints (SMT-LIB)
2. Solver provides precise solutions
3. LLM interprets results back to natural language

**Example Workflow**:
```python
# Natural Language Input
"Can you help plan a meeting for David, Emma, and Alex?
 David is free Monday or Tuesday, Emma is free Tuesday or Wednesday,
 Alex is free Tuesday or Thursday. Find a common day."

# LLM generates Z3 constraints
days = ["Monday", "Tuesday", "Wednesday", "Thursday"]
David_free = [Bool(f"David_free_{day}") for day in days]
Emma_free = [Bool(f"Emma_free_{day}") for day in days]
Alex_free = [Bool(f"Alex_free_{day}") for day in days]

solver = Solver()
solver.add(Or(David_free[0], David_free[1]))
solver.add(Or(Emma_free[1], Emma_free[2]))
solver.add(Or(Alex_free[1], Alex_free[3]))

# Add mutual exclusivity constraints
common_day = [And(David_free[i], Emma_free[i], Alex_free[i])
              for i in range(len(days))]
solver.add(Or(common_day))

# Solve and interpret
if solver.check() == sat:
    # Output: "Tuesday is the common day"
```

### 4.3 Model Checking Agents (PAT Agent)

**Workflow**:
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Natural     │────▶│ LLM         │────▶│ Formal      │
│ Language    │     │ Planning    │     │ Code        │
│ Description │     │             │     │ Generation  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌─────────────┐             │
                    │ Model       │◀────────────┘
                    │ Checker     │
                    │ (PAT)       │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐     ┌─────────────┐
                    │ Verification│────▶│ Refinement  │
                    │ Feedback    │     │ Loop (APE)  │
                    └─────────────┘     └─────────────┘
```

---

## 5. Key Research Papers and Frameworks

### 5.1 POMDP-Based Approaches

| Paper/Project | Key Contribution | URL |
|---------------|------------------|-----|
| **Tru-POMDP** | LLM + POMDP for household tasks | https://tru-pomdp.github.io/ |
| **LLM-Guided POMDP** | Probabilistic program induction | https://arxiv.org/html/2505.02216v1 |
| **From Words to Actions** | Hierarchical RL with LLM Planner | https://arxiv.org/html/2405.19883v1 |
| **Agentic RL Survey** | POMDP formalization of agents | https://www.alphaxiv.org/overview/2509.02547v3 |

### 5.2 Automata-Based Approaches

| Paper/Project | Key Contribution | URL |
|---------------|------------------|-----|
| **Are Agents Just Automata?** | Chomsky hierarchy mapping | https://arxiv.org/html/2510.23487v1 |
| **Formal-LLM** | FSM constraints for agents | https://arxiv.org/html/2402.00798 |
| **StateFlow** | State-driven workflows | https://arxiv.org/html/2403.11322v1 |
| **MetaAgent** | FSM-based multi-agent systems | https://openreview.net/forum?id=vOxaD3hhPt |

### 5.3 Formal Methods + LLM

| Paper/Project | Key Contribution | URL |
|---------------|------------------|-----|
| **Fusion of LLM and FM** | Comprehensive roadmap | https://arxiv.org/html/2412.06512v1 |
| **VeriPlan** | Formal verification + LLM planning | https://dl.acm.org/doi/10.1145/3706598.3714113 |
| **AgentGuard** | Runtime verification | https://arxiv.org/html/2509.23864v1 |

---

## 6. Practical Implementation Patterns

### 6.1 Pattern 1: FSM-Constrained LLM Agent

```python
class FSMConstrainedAgent:
    """
    Regular Agent: LLM as transition oracle within fixed state graph
    """
    def __init__(self, states, transitions, llm):
        self.states = states  # Finite set S
        self.transitions = transitions  # δ: S × Σ → S
        self.llm = llm
        self.current_state = states[0]

    def act(self, perception):
        # LLM selects among predeclared edges only
        valid_transitions = self.transitions[self.current_state]
        action = self.llm.select_transition(perception, valid_transitions)
        self.current_state = action.next_state
        return action
```

### 6.2 Pattern 2: POMDP Planning Agent

```python
class POMDPAgent:
    """
    Context-Free Agent with belief tracking
    """
    def __init__(self, pomdp_model, llm):
        self.model = pomdp_model
        self.llm = llm
        self.belief = self.initialize_belief()
        self.plan_stack = []  # LIFO for hierarchical planning

    def plan(self, goal):
        # Generate hypotheses using LLM
        hypotheses = self.llm.generate_hypotheses(goal)

        # Update belief with LLM-generated particles
        self.belief = self.hybrid_belief_update(hypotheses)

        # POMDP planning
        action = self.pomdp_solve()
        return action

    def hybrid_belief_update(self, llm_particles):
        # Bayesian filtering + LLM augmentation
        predicted = self.model.predict(self.belief)
        filtered = self.model.update(predicted, self.observation)
        return self.augment(filtered, llm_particles)
```

### 6.3 Pattern 3: Verifiable TC Agent

```python
class VerifiableTCAgent:
    """
    TC Agent with runtime verification guards
    """
    def __init__(self, llm, memory, safety_properties):
        self.llm = llm
        self.memory = memory  # Unbounded R/W
        self.safety_monitor = SafetyMonitor(safety_properties)
        self.verification_oracle = VerificationOracle()

    def act(self, perception):
        # Generate action plan
        plan = self.llm.plan(perception, self.memory.read())

        # Verify before execution
        if self.verification_oracle.verify(plan, self.safety_properties):
            return self.execute(plan)
        else:
            return self.safe_fallback()
```

---

## 7. Decidability and Verification Implications

### 7.1 What Can Be Verified?

| Agent Class | Reachability | Safety | Liveness | Halting |
|-------------|--------------|--------|----------|---------|
| Regular (FA) | ✓ Decidable | ✓ Decidable | ✓ Decidable | ✓ Decidable |
| Context-Free (PDA) | ✓ Decidable | ✓ Decidable | Partial | Partial |
| Context-Sensitive (LBA) | ✓ Decidable | Partial | Partial | ✓ Decidable |
| TC (TM) | ✗ Undecidable | ✗ Undecidable | ✗ Undecidable | ✗ Undecidable |

### 7.2 Rice's Theorem Implications

For **TC Agents** (Turing-complete):
> Any non-trivial semantic property is undecidable.

This means:
- **Safety verification**: Impossible in general
- **Halting problem**: Cannot determine if agent terminates
- **Behavioral equivalence**: Cannot compare two agents

**Solution**: Use runtime monitors and weaker agent classes where possible.

---

## 8. Probabilistic Extensions

### 8.1 From Deterministic to Probabilistic

LLM transition functions are inherently **non-deterministic**. The framework extends to:

**Probabilistic Finite Automata (PFA)**:
```
δ: S × Σ → Distribution(S)
```

**Implications**:
- Verification shifts from "Can it happen?" to "What's the probability?"
- Enables **quantitative risk analysis**
- Supports policy and compliance frameworks

### 8.2 Belief-Based Planning

```
POMDP Policy: π: B → A
Where B = distribution over states

Expected Utility:
EU(π, b) = Σ_s b(s) · R(s, π(b)) + γ · Σ_{s',o} P(s',o|s,π(b)) · EU(π, b')
```

---

## 9. Research Directions and Open Problems

### 9.1 Near-Term Challenges

1. **Scalability**: State-space explosion in real-world agents
2. **Abstraction Fidelity**: Accurate modeling of LLM behavior
3. **Probabilistic Verification**: Handling non-deterministic LLM outputs
4. **Memory Management**: Enforcing bounded memory in practice

### 9.2 Future Directions

1. **Minimal-Class Synthesis**: Automatic compilation of task specs to minimal automata
2. **Hybrid Architectures**: FA/DPDA cores with TM components and runtime guards
3. **Calibrated Risk**: Probabilistic verification for safety-critical domains
4. **Cross-Session Learning**: Persistent memory with verifiable updates

### 9.3 Benchmarking Needs

- Shared conformance testing suites
- Class boundary probes under realistic constraints
- Standard task domains for right-sizing evaluation

---

## 10. Key Takeaways

1. **Memory Architecture = Computational Power**: The agent's memory model directly determines its computational class and verifiability.

2. **Right-Size Your Agents**: Don't default to Turing-complete agents when simpler architectures suffice.

3. **POMDPs for Uncertainty**: Use POMDP frameworks when dealing with partial observability, ambiguous instructions, and hidden states.

4. **Formal Methods Enable Trust**: Integration of SMT solvers, model checkers, and automata theory provides mathematical guarantees.

5. **Undecidability Trade-offs**: Higher computational power comes at the cost of verifiability—choose wisely.

6. **LLM as Oracle, Architecture as Constraint**: Let LLMs provide intelligence within formally defined architectural bounds.

---

## References

### Primary Sources
- [Are Agents Just Automata?](https://arxiv.org/html/2510.23487v1) - Formal equivalence between agents and Chomsky hierarchy
- [Tru-POMDP](https://tru-pomdp.github.io/) - LLM + POMDP for task planning
- [Fusion of LLM and Formal Methods](https://arxiv.org/html/2412.06512v1) - Comprehensive roadmap

### Foundational Theory
- [Markov Decision Process - Wikipedia](https://en.wikipedia.org/wiki/Markov_decision_process)
- [Partially Observable MDP - Wikipedia](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process)
- [MIT 6.390 - MDPs](https://introml.mit.edu/notes/mdp.html)

### Practical Frameworks
- [StateFlow](https://arxiv.org/html/2403.11322v1) - State-driven LLM workflows
- [Formal-LLM](https://arxiv.org/html/2402.00798) - Integrating formal language and natural language
- [GitHub: LLM-State-Machine](https://github.com/jsz-05/LLM-State-Machine) - FSM + LLM framework

---

*Document compiled from research as of March 2026*
