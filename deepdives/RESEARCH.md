# Deep Dive: Multi-Armed Bandits & LLMs/Agents

A comprehensive investigation into the intersection of bandit algorithms and large language models.

---

## Table of Contents
1. [Overview](#overview)
2. [Why This Intersection Matters](#why-this-intersection-matters)
3. [Two Main Directions](#two-main-directions)
4. [Key Use Cases](#key-use-cases)
5. [Practical Implementations](#practical-implementations)
6. [Key Algorithms](#key-algorithms)
7. [Important Papers](#important-papers)
8. [Tools & Frameworks](#tools--frameworks)
9. [Getting Started](#getting-started)
10. [Open Problems & Future Directions](#open-problems--future-directions)

---

## Overview

Multi-armed bandits (MAB) and Large Language Models (LLMs) have emerged as complementary tools in AI:
- **Bandits** provide a mathematical framework for sequential decision-making under uncertainty, balancing exploration vs exploitation
- **LLMs** provide semantic understanding, reasoning, and natural language capabilities

Their intersection enables powerful new capabilities for both fields.

---

## Why This Intersection Matters

### Problems with Traditional LLM Experimentation
1. **Fixed stopping rules** - You must pre-determine sample size (risks under/over-sampling)
2. **Suboptimal sample allocation** - Traffic isn't adapted based on performance
3. **P-hacking problem** - Repeatedly checking significance inflates error rates
4. **Manual prompt optimization** - Labor-intensive and doesn't scale

### What Bandits Provide
1. **Adaptive sampling** - Dynamically allocate resources to promising variants
2. **Anytime-valid inference** - Check results anytime without inflating error rates
3. **Statistical guarantees** - Rigorous bounds on misidentification probability
4. **Efficient exploration** - Systematically explore the space of possibilities

---

## Two Main Directions

### Direction 1: Bandits Enhancing LLMs

| Application Area | Description |
|-----------------|-------------|
| **Prompt Optimization** | Dynamically select/evolve prompts to maximize performance |
| **Model Selection** | Choose between different LLMs based on task/context |
| **Hyperparameter Tuning** | Optimize temperature, top_p, etc. automatically |
| **RLHF & Alignment** | Improve preference learning with bandit methods |
| **RAG Optimization** | Select retrieval strategies adaptively |
| **Inference Routing** | Route queries to appropriate models (cost vs quality) |
| **Tool/Function Calling** | Select tools dynamically for agents |
| **Personalization** | Adapt responses to user preferences over time |

### Direction 2: LLMs Enhancing Bandits

| Application Area | Description |
|-----------------|-------------|
| **Reward Prediction** | Use LLMs to predict rewards in bandit environments |
| **Context Understanding** | Leverage semantic understanding for contextual bandits |
| **Environment Modeling** | LLMs as simulators/environment models |
| **Prior Generation** | Bootstrap bandits with LLM-generated priors |
| **Action Selection** | LLMs as decision-makers in bandit settings |
| **Non-stationary Environments** | Adapt to changing environments using LLM reasoning |

---

## Key Use Cases

### 1. Prompt Optimization with Bandits

**Problem**: Which prompt variant produces the best results?

**Approach**:
- Each prompt variant = an "arm"
- Reward = task success metric (accuracy, user satisfaction, etc.)
- Use bandit algorithm to efficiently identify best prompt

**Key Insight**: Direct arm selection by LLMs is often suboptimal. Better approach: Use LLM for reward prediction, bandit algorithm for exploration/exploitation.

```
Arms = [prompt_v1, prompt_v2, prompt_v3, ...]
At each step:
  1. Bandit selects which prompt to use
  2. Get reward (e.g., user click, accuracy)
  3. Update bandit statistics
  4. Eventually converge to best prompt
```

### 2. LLM Gateway / Model Routing

**Problem**: Which LLM should handle each request?

**Approach**:
- Arms = different models (GPT-4, Claude, Gemini, smaller models)
- Context = query characteristics
- Reward = quality/cost/latency tradeoff

**TensorZero Implementation**:
```toml
[functions.extract_entities.experimentation]
type = "track_and_stop"
candidate_variants = ["gpt-5-mini-good-prompt", "gpt-5-mini-bad-prompt"]
metric = "exact_match"
```

### 3. RLHF as Contextual Bandit

**Key Insight**: RLHF can be viewed as a contextual bandit problem:
- Context = prompt
- Action = response
- Reward = human preference

**Simplification**: Some research shows single-step contextual bandit algorithms can work as well as PPO for RLHF, with much simpler implementation.

### 4. Adaptive RAG

**Problem**: When and how to retrieve for RAG?

**Approach**:
- Arms = different retrieval strategies (no retrieval, semantic search, hybrid, etc.)
- Context = query complexity/type
- Reward = answer quality

### 5. Agent Tool Selection

**Problem**: Which tool should an agent use?

**Approach**:
- Arms = available tools
- Context = task description, conversation history
- Reward = task success, efficiency

---

## Key Algorithms

### 1. Thompson Sampling (TS)
- Bayesian approach
- Sample from posterior distribution of each arm
- Select arm with highest sample
- Good for regret minimization

### 2. Upper Confidence Bound (UCB)
- Optimistic approach
- Select arm with highest upper confidence bound
- Balances exploration and exploitation
- Good for regret minimization

### 3. Track-and-Stop (for Best-Arm Identification)
- **Best for LLM experimentation**
- Learns optimal sampling proportions on-the-fly
- Uses generalized likelihood ratio test (GLRT) for stopping
- Provides anytime-valid statistical guarantees
- Can add new variants mid-experiment

### 4. LinUCB / NeuralUCB
- Contextual bandit variants
- Linear/neural function approximation
- Adapts to context (e.g., query features)

### 5. Dueling Bandits
- Preference-based feedback (A vs B)
- Useful when absolute rewards are hard to define
- LLMs can help with preference prediction

---

## Practical Implementations

### TensorZero (Recommended)

**What it is**: Open-source LLMOps platform with built-in bandit experimentation

**Features**:
- LLM Gateway (unified API for all providers)
- Adaptive A/B testing with Track-and-Stop algorithm
- Observability and feedback collection
- Written in Rust for performance (<1ms p99 latency)

**Key Benefits**:
- 37% faster experiment completion vs uniform sampling
- Anytime-valid stopping (no p-hacking)
- Automatic winner detection

**Getting Started**:
```bash
# Docker
docker run -p 3000:3000 tensorzero/tensorzero

# Or deploy with ClickHouse
docker-compose up -d
```

### EVOLvE Framework

**What it is**: Framework for evaluating LLMs for in-context reinforcement learning

**Features**:
- Compare LLMs against classic bandit algorithms
- Contextual and non-contextual bandit scenarios
- LinUCB implementations

**GitHub**: https://github.com/allenanie/EVOLvE

### contextualbandits (Python)

**What it is**: Python package with contextual bandit implementations

**Algorithms**:
- LinUCB, LinTS
- Neural variants
- Various policy wrappers

**GitHub**: https://github.com/david-cortes/contextualbandits

---

## Important Papers

### Surveys & Tutorials

| Paper | Description |
|-------|-------------|
| [Multi-Armed Bandits Meet Large Language Models](https://arxiv.org/abs/2505.13355) (AAAI 2026) | Comprehensive survey on the intersection |
| [Large Language Model-Enhanced Multi-Armed Bandits](https://arxiv.org/abs/2502.01118) (ICLR 2025) | LLM-based reward prediction for bandits |
| [A Tutorial on Multi-Armed Bandit Applications for LLMs](https://dl.acm.org/doi/10.1145/3637528.3671440) | ACM tutorial on the topic |

### Prompt Optimization

| Paper | Key Contribution |
|-------|-----------------|
| Prompt Optimization with Logged Bandit Data | Use historical data for prompt optimization |
| Efficient Prompt Optimization through Best Arm Identification | TRIPLE algorithm for prompt selection |
| INSTINCT | Neural bandits for instruction optimization |
| Bandit-Based Prompt Design Strategy Selection | Select prompt design strategies with bandits |

### LLM-Enhanced Bandits

| Paper | Key Contribution |
|-------|-----------------|
| Large Language Model-Enhanced Multi-Armed Bandits | LLM for reward prediction in Thompson Sampling |
| Beyond Numeric Rewards: In-Context Dueling Bandits | LLM agents in dueling bandit settings |
| Jump Starting Bandits with LLM-Generated Prior Knowledge | Bootstrap bandits with LLM priors |
| LLM-Informed Multi-Armed Bandit Strategies for Non-Stationary Environments | Handle non-stationary bandits with LLMs |

### RLHF & Alignment

| Paper | Key Contribution |
|-------|-----------------|
| Provably Efficient RLHF Pipeline: A Unified View from Contextual Bandits | RLHF as contextual bandit |
| Sharp Analysis for KL-Regularized Contextual Bandits and RLHF | Theoretical analysis |

### Inference & Routing

| Paper | Key Contribution |
|-------|-----------------|
| LLM Bandit: Cost-Efficient LLM Generation | Dynamic routing based on preferences |
| Learning to Route LLMs from Bandit Feedback | One policy, multiple trade-offs |
| BanditSpec: Adaptive Speculative Decoding | Bandit for speculative decoding |

---

## Tools & Frameworks

### Production-Ready

| Tool | Purpose | Link |
|------|---------|------|
| **TensorZero** | LLM Gateway + Bandit Experimentation | [GitHub](https://github.com/tensorzero/tensorzero) |
| **contextualbandits** | Python contextual bandit library | [GitHub](https://github.com/david-cortes/contextualbandits) |
| **Pearl (Meta)** | Deep contextual bandits | [GitHub](https://github.com/facebookresearch/Pearl) |

### Research Frameworks

| Tool | Purpose | Link |
|------|---------|------|
| **EVOLvE** | Evaluate LLMs for in-context RL | [GitHub](https://github.com/allenanie/EVOLvE) |
| **Awesome-LLM-Bandit-Interaction** | Curated paper list | [GitHub](https://github.com/bucky1119/Awesome-LLM-Bandit-Interaction) |

---

## Getting Started

### Step 1: Understand Your Problem

Ask yourself:
1. What are my "arms"? (prompts, models, configurations)
2. What is my "reward"? (accuracy, user satisfaction, cost)
3. Do I have context? (query features, user attributes)
4. What is my goal? (best-arm identification vs regret minimization)

### Step 2: Choose Your Approach

| Scenario | Recommended Approach |
|----------|---------------------|
| A/B testing prompts | Track-and-Stop (TensorZero) |
| Personalization | Contextual bandit (LinUCB/LinTS) |
| Model routing | Contextual bandit with cost constraints |
| RLHF | Contextual bandit (simpler than PPO) |
| Exploration in agents | Thompson Sampling or UCB |

### Step 3: Implement

**Option A: TensorZero (Recommended for Production)**
```toml
# tensorzero.toml
[functions.my_function]
type = "chat"

[functions.my_function.variants.prompt_v1]
type = "chat_completion"
model = "openai::gpt-4o-mini"
system_template = "prompts/v1.txt"

[functions.my_function.variants.prompt_v2]
type = "chat_completion"
model = "openai::gpt-4o-mini"
system_template = "prompts/v2.txt"

[functions.my_function.experimentation]
type = "track_and_stop"
candidate_variants = ["prompt_v1", "prompt_v2"]
metric = "user_satisfaction"
```

**Option B: Python (Custom Implementation)**
```python
# Using contextualbandits
from contextualbandits import LinUCB

# Initialize
policy = LinUCB(nchoices=3)  # 3 prompt variants

# For each query
context = get_query_features(query)
arm = policy.predict(context)
reward = get_user_feedback(arm)
policy.update(context, arm, reward)
```

### Step 4: Define Metrics

Good metrics for LLM experimentation:
- **Exact match**: For structured outputs
- **F1 score**: For extraction tasks
- **User clicks/ratings**: For recommendations
- **Task completion**: For agents
- **Cost/latency**: For routing decisions

---

## Open Problems & Future Directions

### Research Challenges

1. **Multi-objective optimization**: Balancing quality, cost, and latency simultaneously
2. **Non-stationary environments**: Handling concept drift in user preferences
3. **Cold start**: Efficient initialization with limited data
4. **Large action spaces**: When there are thousands of prompt variants
5. **Safety constraints**: Ensuring bandits don't select harmful outputs

### Emerging Directions

1. **Self-improving systems**: LLMs that generate and test their own prompts
2. **Multi-agent bandits**: Bandits in multi-agent LLM systems
3. **Offline-to-online**: Leveraging historical data for online bandits
4. **Hierarchical bandits**: Nested bandit structures for complex decisions
5. **Causal bandits**: Incorporating causal reasoning

---

## Quick Reference

### When to Use Bandits for LLMs

| Use Case | Bandit Type | Algorithm |
|----------|------------|-----------|
| Prompt A/B testing | Stochastic MAB | Track-and-Stop |
| Model routing | Contextual MAB | LinUCB |
| Personalization | Contextual MAB | LinTS |
| RLHF | Contextual MAB | REINFORCE-style |
| Tool selection | Contextual MAB | NeuralUCB |

### Key Hyperparameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| delta (δ) | Error probability | 0.05 |
| epsilon (ε) | Minimum detectable difference | 0.0 (strict) |
| Temperature (TS) | Exploration in Thompson Sampling | Decaying schedule |
| Alpha (UCB) | Confidence level | 1.0 |

---

## Resources

### Papers
- [Awesome-LLM-Bandit-Interaction](https://github.com/bucky1119/Awesome-LLM-Bandit-Interaction) - Comprehensive paper collection

### Blog Posts
- [Bandits in your LLM Gateway (TensorZero)](https://www.tensorzero.com/blog/bandits-in-your-llm-gateway/)
- [Bandits for Prompts (Medium)](https://medium.com/data-science-collective/bandits-for-prompts-the-practical-rl-trick-that-makes-your-llm-improve-while-its-still-running-ca8da6acf213)

### Tutorials
- [AAAI 2026 Tutorial: Bandits, LLMs, and Agentic AI](https://research.ibm.com/publications/bandits-llms-and-agentic-ai)

---

*Last updated: March 2026*
