# Memory and Search in LLM Agents for Strategic and Imperfect Information Games

## Overview

This analysis synthesizes recent research (2023-2026) on memory-augmented LLM agents for strategic games, with a focus on imperfect information environments. The papers reveal several key patterns in how LLM agents use memory and search mechanisms to handle uncertainty, opponent modeling, and long-horizon reasoning.

## Key Papers Analyzed

1. **MEMO: Memory-Augmented Model Context Optimization** (arXiv:2603.09022, Mar 2026)
   - **Focus**: Persistent memory banks for multi-turn multi-agent games
   - **Memory mechanism**: CRUD-style memory bank with structured insights
   - **Search**: Tournament-style context evolution with TrueSkill rating

2. **Suspicion-Agent: Playing Imperfect Information Games with Theory of Mind** (arXiv:2309.17277, COLM 2024)
   - **Focus**: Theory of Mind (ToM) for opponent modeling in imperfect information games
   - **Memory mechanism**: Game history tracking and opponent pattern analysis
   - **Search**: First and second-order ToM planning

3. **Look-ahead Reasoning with a Learned Model in Imperfect Information Games** (arXiv:2510.05048, Oct 2025)
   - **Focus**: Learned abstractions for tractable look-ahead search
   - **Memory mechanism**: Abstracted game models learned from interaction
   - **Search**: Look-ahead reasoning with learned abstractions

4. **Agent-Pro: Learning to Evolve via Policy-Level Reflection** (arXiv:2402.17574, Jun 2024)
   - **Focus**: Policy evolution through reflection in interactive games
   - **Memory mechanism**: Dynamic belief generation and reflection
   - **Search**: Depth-first search for policy optimization

## Memory Mechanisms for Imperfect Information Games

### 1. Persistent Memory Banks (MEMO)
```
┌─────────────────────────────────┐
│      Persistent Memory Bank      │
├─────────────────────────────────┤
│  CRUD Operations:                │
│  • Add: New unique insights      │
│  • Remove: Conflicting insights  │
│  • Edit: Merge similar insights  │
└─────────────────────────────────┘
```
**Key Features**:
- **Structured insights**: Distills trajectories into reusable strategic principles
- **Cross-game retention**: Memory persists across optimization generations
- **Self-correction**: Conflicting insights removed to avoid contradictions
- **Sample efficiency**: Enables cumulative learning without weight updates

### 2. Theory of Mind Memory (Suspicion-Agent)
```
┌─────────────────────────────────┐
│   Opponent Modeling Memory       │
├─────────────────────────────────┤
│  • Behavioral patterns           │
│  • Action tendencies             │
│  • Belief tracking               │
│  • Strategic adaptations         │
└─────────────────────────────────┘
```
**Key Features**:
- **High-order ToM**: "I know that he knows that I know..."
- **Pattern analysis**: Identifies opponent tendencies from game history
- **Belief updating**: Maintains probabilistic models of hidden information
- **Adaptive strategies**: Adjusts gameplay based on opponent type

### 3. Abstracted Model Memory (Look-ahead Reasoning)
```
┌─────────────────────────────────┐
│   Learned Game Abstraction       │
├─────────────────────────────────┤
│  • State abstraction             │
│  • Action abstraction            │
│  • Belief compression            │
│  • Subgame boundaries            │
└─────────────────────────────────┘
```
**Key Features**:
- **Complexity reduction**: Abstracts large state spaces to manageable sizes
- **Generalization**: Learns transferable abstractions from interaction
- **Theoretical guarantees**: Maintains principled reasoning within abstractions
- **Scalability**: Enables look-ahead search in previously intractable games

### 4. Policy-Level Memory (Agent-Pro)
```
┌─────────────────────────────────┐
│   Dynamic Belief Memory          │
├─────────────────────────────────┤
│  • Self-beliefs                  │
│  • World-beliefs                 │
│  • Policy evolution              │
│  • Reflection traces             │
└─────────────────────────────────┘
```
**Key Features**:
- **Belief generation**: Constructs beliefs about self and environment
- **Reflection process**: Iteratively "fine-tunes" irrational beliefs
- **Policy evolution**: Depth-first search for optimal policies
- **Interactive learning**: Learns from experience without explicit training

## Search and Planning Strategies

### 1. Tournament-Style Evolution (MEMO)
```
Context Population → Self-Play Evaluation → TrueSkill Rating → Selection → Next Generation
```
**Characteristics**:
- **Population-based**: Maintains diverse context candidates
- **Uncertainty-aware**: Uses Bayesian skill ratings (TrueSkill)
- **Memory-augmented**: New contexts generated using memory insights
- **Prioritized replay**: Focuses on rare and decisive states

### 2. Theory of Mind Planning (Suspicion-Agent)
```
Observation → Opponent Model → Strategy Simulation → Action Selection
```
**Characteristics**:
- **Recursive reasoning**: First and second-order ToM (I think you think I think...)
- **Opponent simulation**: Predicts opponent responses to different actions
- **Strategy adaptation**: Adjusts based on opponent behavioral patterns
- **Information asymmetry mitigation**: Uses ToM to reduce uncertainty

### 3. Look-ahead Search with Abstractions (Look-ahead Reasoning)
```
Current State → Abstracted Model → Look-ahead Search → Action Selection
```
**Characteristics**:
- **Model-based**: Uses learned abstracted models for search
- **Tractable reasoning**: Abstractions limit search space complexity
- **Theoretically sound**: Maintains game-theoretic principles
- **Zero-shot capability**: Works with pre-trained agents

### 4. Policy-Level Search (Agent-Pro)
```
Current Policy → Reflection → Belief Update → Policy Optimization → New Policy
```
**Characteristics**:
- **Reflection-driven**: Analyzes past trajectories and outcomes
- **Belief refinement**: Updates self and world beliefs
- **Depth-first optimization**: Searches policy space systematically
- **Evolutionary**: Iteratively improves behavioral policies

## Performance Comparison

### Game Performance by Approach
| Approach | Imperfect Info Games | Perfect Info Games | Sample Efficiency | Stability |
|----------|---------------------|-------------------|-------------------|-----------|
| **MEMO** | **Best** (49.5% win rate) | Good | **High** (2,000 games) | **High** (6.4% RSE) |
| **Suspicion-Agent** | Good (beats NFSP/DMC) | Not tested | Zero-shot | Moderate |
| **Look-ahead Reasoning** | Good (scalable) | Good | Moderate | High |
| **Agent-Pro** | Good (poker variants) | Not tested | Moderate | Moderate |
| **Traditional RL** | Poor | **Best** | Low (38,000+ games) | Low |

### Memory Efficiency Comparison
| Memory Type | Storage | Retrieval | Update | Game Suitability |
|-------------|---------|-----------|--------|------------------|
| **Structured Insights** (MEMO) | Medium | Fast | CRUD operations | Multi-turn, multi-agent |
| **Opponent Models** (Suspicion) | Light | Fast | Incremental | Imperfect information |
| **Game Abstractions** (Look-ahead) | Heavy | Medium | Learning | Large state spaces |
| **Policy Beliefs** (Agent-Pro) | Medium | Medium | Reflection | Interactive learning |

## Critical Insights

### 1. Memory Enables Cumulative Learning
- **Key finding**: "Exploration alone yields only modest gains; persistent memory is what transforms context optimization from a memoryless search into a cumulative learning process." (MEMO)
- **Implication**: Memory mechanisms are essential for sample-efficient learning in interactive settings.

### 2. Theory of Mind Compensates for Imperfect Information
- **Key finding**: GPT-4 exhibits strong high-order ToM capabilities that can be leveraged for opponent modeling.
- **Implication**: LLMs' social reasoning abilities can be repurposed for strategic game playing.

### 3. Abstraction Enables Tractable Search
- **Key finding**: Learned abstractions make theoretically principled look-ahead reasoning tractable in large games.
- **Implication**: Model compression techniques can bridge the gap between exact solvers and heuristic approaches.

### 4. Reflection Drives Policy Evolution
- **Key finding**: Policy-level reflection allows agents to identify and correct irrational behaviors.
- **Implication**: Meta-cognitive capabilities can be engineered through structured reflection processes.

## Implementation Patterns

### Common Architectural Components
1. **Observation Interpreter**: Converts game state to natural language descriptions
2. **Memory Module**: Stores and retrieves past experiences or strategic insights
3. **Planning Module**: Generates action sequences or strategic plans
4. **Reflection Module**: Analyzes outcomes and extracts lessons
5. **Adaptation Module**: Adjusts strategies based on opponent behavior

### Prompt Engineering Patterns
- **Role-based prompts**: "You are a strategic poker player..."
- **Memory injection**: "Based on past games where the opponent..."
- **Reflection prompts**: "Analyze what went wrong in this trajectory..."
- **Planning prompts**: "Consider the opponent's likely response to..."

### Evaluation Best Practices
- **Multi-run testing**: Addresses high variance in multi-agent interactions
- **Opponent diversity**: Tests against various strategy types
- **Statistical significance**: Uses appropriate metrics (TrueSkill, RSE)
- **Comparative baselines**: Includes traditional algorithms (CFR, RL)

## Open Challenges and Future Directions

### 1. Scalability Issues
- **Problem**: Memory banks grow linearly with experience
- **Solution needed**: Forgetting mechanisms, compression, hierarchical organization

### 2. Cross-Game Transfer
- **Problem**: Memory often game-specific
- **Solution needed**: Abstract strategic principles that transfer across domains

### 3. Real-World Deployment
- **Problem**: Game environments are idealized simulations
- **Solution needed**: Robustness to noisy observations, partial rule specification

### 4. Computational Efficiency
- **Problem**: Thousands of self-play games still required
- **Solution needed**: More efficient search, parallelization, distillation

### 5. Theoretical Foundations
- **Problem**: Empirical success lacks theoretical guarantees
- **Solution needed**: Convergence proofs, optimality bounds, regret analysis

## Practical Recommendations

### For Researchers
1. **Focus on imperfect information games**: Where LLMs show unique advantages over traditional methods
2. **Incorporate memory early**: Even simple memory mechanisms yield significant benefits
3. **Evaluate robustness**: Test across multiple runs with different random seeds
4. **Compare to traditional baselines**: CFR-based solvers provide important reference points

### For Practitioners
1. **Start with MEMO framework**: Provides robust memory and search mechanisms
2. **Leverage ToM capabilities**: Especially useful for opponent modeling
3. **Consider sample efficiency**: Memory-augmented approaches require fewer interactions
4. **Prioritize stability**: Variance reduction is critical for reliable deployment

### For Tool Builders
1. **Standardize memory interfaces**: CRUD operations, similarity metrics, retrieval mechanisms
2. **Develop game environments**: Support for imperfect information, multi-agent interactions
3. **Create evaluation suites**: Standardized benchmarks for strategic game playing
4. **Build visualization tools**: For understanding memory contents and decision processes

## Conclusion

The research landscape reveals a clear trend toward memory-augmented LLM agents for strategic games, with particularly strong results in imperfect information settings. Key innovations include:

1. **Persistent memory banks** that enable cumulative learning without weight updates
2. **Theory of Mind** capabilities that compensate for information asymmetry
3. **Learned abstractions** that make complex games tractable
4. **Reflection processes** that drive policy evolution

These approaches collectively address the core challenges of imperfect information games: uncertainty management, opponent modeling, long-horizon reasoning, and sample efficiency. While traditional game theory approaches (CFR) still provide stronger theoretical guarantees, LLM-based methods offer greater flexibility, adaptability, and potential for generalization.

The most promising direction appears to be hybrid approaches that combine the theoretical rigor of game-theoretic solvers with the adaptability and reasoning capabilities of LLMs, mediated by sophisticated memory mechanisms that enable efficient learning and robust performance across diverse game environments.